"""
This module solves the LLG equation or one of its many variants.

Using native code tied in by instant, it allows to specify parameters
and the terms of the equation that are to be used and then solves for dm/dt.

No effective field computation, no saving of magnetisation to file or
whatever, just straight up solving of the equation of motion.

"""
import logging
import math
from functools import lru_cache
from types import SimpleNamespace
try:
    import instant
except ModuleNotFoundError:
    # DOLFIN 2019 on the pixi path still supports the downstream C++ compile
    # step, but the legacy instant cache helper may be absent. Fall back to
    # direct compilation instead of failing during module import. [Codex GPT-5.4]
    instant = None
import dolfin as df
import os
import fnmatch
import glob

from os import path

log = logging.getLogger(name="finmag")

ELECTRON_CHARGE = 1.602176565e-19
HBAR = 1.054571726e-34
MU_0 = 4 * math.pi * 1e-7


def _cross(ax, ay, az, bx, by, bz):
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def _damping(alpha, gamma, m_x, m_y, m_z, H_x, H_y, H_z, dm_x, dm_y, dm_z):
    prefactor = - alpha * gamma / (1 + alpha * alpha)
    mH = m_x * H_x + m_y * H_y + m_z * H_z
    mm = m_x * m_x + m_y * m_y + m_z * m_z
    dm_x += prefactor * (m_x * mH - H_x * mm)
    dm_y += prefactor * (m_y * mH - H_y * mm)
    dm_z += prefactor * (m_z * mH - H_z * mm)
    return dm_x, dm_y, dm_z


def _precession(alpha, gamma, m_x, m_y, m_z, H_x, H_y, H_z, dm_x, dm_y, dm_z):
    prefactor = - gamma / (1 + alpha * alpha)
    cx, cy, cz = _cross(m_x, m_y, m_z, H_x, H_y, H_z)
    dm_x += prefactor * cx
    dm_y += prefactor * cy
    dm_z += prefactor * cz
    return dm_x, dm_y, dm_z


def _relaxation(c, m_x, m_y, m_z, dm_x, dm_y, dm_z):
    mm = m_x * m_x + m_y * m_y + m_z * m_z
    prefactor = c * (1.0 - mm)
    dm_x += prefactor * m_x
    dm_y += prefactor * m_y
    dm_z += prefactor * m_z
    return dm_x, dm_y, dm_z


def _slonczewski(alpha, gamma, J, Ms, m_x, m_y, m_z, d, P, p, lambda_,
                 epsilonprime, dm_x, dm_y, dm_z):
    p_x, p_y, p_z = p
    mm = m_x * m_x + m_y * m_y + m_z * m_z
    mp = m_x * p_x + m_y * p_y + m_z * p_z
    gamma_ll = gamma / (1 + alpha * alpha)
    lambda_sq = lambda_ * lambda_
    beta = J * HBAR / (MU_0 * Ms * ELECTRON_CHARGE * d)
    epsilon = P * lambda_sq / (lambda_sq + 1 + (lambda_sq - 1) * mp)
    perp = alpha * epsilon - epsilonprime
    para = epsilon - alpha * epsilonprime
    cx, cy, cz = _cross(m_x, m_y, m_z, p_x, p_y, p_z)
    dm_x += gamma_ll * beta * (perp * cx - para * (mp * m_x - mm * p_x))
    dm_y += gamma_ll * beta * (perp * cy - para * (mp * m_y - mm * p_y))
    dm_z += gamma_ll * beta * (perp * cz - para * (mp * m_z - mm * p_z))
    return dm_x, dm_y, dm_z


def _zhangli(alpha, Ms, m_x, m_y, m_z, tau_x, tau_y, tau_z, u_0, beta,
             dm_x, dm_y, dm_z):
    coeff_stt = 0.0 if Ms == 0 else u_0 / (1 + alpha * alpha) / Ms
    mtau = m_x * tau_x + m_y * tau_y + m_z * tau_z
    tau_perp_x = tau_x - mtau * m_x
    tau_perp_y = tau_y - mtau * m_y
    tau_perp_z = tau_z - mtau * m_z
    mtp_x, mtp_y, mtp_z = _cross(m_x, m_y, m_z, tau_perp_x, tau_perp_y, tau_perp_z)
    dm_x += coeff_stt * ((1 + alpha * beta) * tau_perp_x - (beta - alpha) * mtp_x)
    dm_y += coeff_stt * ((1 + alpha * beta) * tau_perp_y - (beta - alpha) * mtp_y)
    dm_z += coeff_stt * ((1 + alpha * beta) * tau_perp_z - (beta - alpha) * mtp_z)
    return dm_x, dm_y, dm_z


class _PythonEquation(object):
    def __init__(self, m, H, dmdt):
        if m.size() != H.size():
            raise Exception("m and H")
        if m.size() != dmdt.size():
            raise Exception("m and dmdt")

        self.magnetisation = m
        self.effective_field = H
        self.derivative = dmdt
        self.pinned_nodes = None
        self.saturation_magnetisation = None
        self.current_density = None
        self.alpha = None
        self.gamma = 2.210173e5
        self.parallel_relaxation_rate = 1e-12
        self.do_precession = True
        self._slonczewski = None
        self._zhangli = None
        self.reorder_dofs_serial = df.parameters["reorder_dofs_serial"]

    def get_pinned_nodes(self):
        return self.pinned_nodes

    def set_pinned_nodes(self, value):
        self.pinned_nodes = value

    def get_saturation_magnetisation(self):
        return self.saturation_magnetisation

    def set_saturation_magnetisation(self, value):
        self.saturation_magnetisation = value

    def get_current_density(self):
        return self.current_density

    def set_current_density(self, value):
        self.current_density = value

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, value):
        self.alpha = value

    def get_gamma(self):
        return self.gamma

    def set_gamma(self, value):
        self.gamma = value

    def get_parallel_relaxation_rate(self):
        return self.parallel_relaxation_rate

    def set_parallel_relaxation_rate(self, value):
        self.parallel_relaxation_rate = value

    def get_do_precession(self):
        return self.do_precession

    def set_do_precession(self, value):
        self.do_precession = value

    def slonczewski(self, d, P, p, lambda_, epsilonprime):
        self._slonczewski = (d, P, p, lambda_, epsilonprime)

    def slonczewski_disable(self):
        self._slonczewski = None

    def slonczewski_status(self):
        return (
            self._slonczewski is not None and
            self.current_density is not None and
            self.saturation_magnetisation is not None
        )

    def zhangli(self, u_0, beta):
        self._zhangli = (u_0, beta)

    def zhangli_disable(self):
        self._zhangli = None

    def zhangli_status(self):
        return (
            self._zhangli is not None and
            self.current_density is not None and
            self.saturation_magnetisation is not None
        )

    def _component_indices(self, node, offset):
        if not self.reorder_dofs_serial:
            x = node
            y = x + offset
            z = y + offset
        else:
            x = 3 * node
            y = x + 1
            z = x + 2
        return x, y, z

    def _solve_arrays(self, m, H, dmdt):
        if self.alpha is None:
            raise RuntimeError("alpha was not set")

        a = self.alpha.get_local()
        pinned = self.pinned_nodes.get_local() if self.pinned_nodes is not None else None
        Ms = self.saturation_magnetisation.get_local() if self.saturation_magnetisation is not None else None
        J = self.current_density.get_local() if self.current_density is not None else None
        offset = len(a)

        for node in range(offset):
            x, y, z = self._component_indices(node, offset)
            dmdt[x] = 0.0
            dmdt[y] = 0.0
            dmdt[z] = 0.0
            if pinned is not None and pinned[node]:
                continue

            dmx, dmy, dmz = _damping(
                a[node], self.gamma,
                m[x], m[y], m[z], H[x], H[y], H[z],
                dmdt[x], dmdt[y], dmdt[z])
            if self.do_precession:
                dmx, dmy, dmz = _precession(
                    a[node], self.gamma,
                    m[x], m[y], m[z], H[x], H[y], H[z],
                    dmx, dmy, dmz)
            dmx, dmy, dmz = _relaxation(
                self.parallel_relaxation_rate,
                m[x], m[y], m[z], dmx, dmy, dmz)
            if self.slonczewski_status():
                dmx, dmy, dmz = _slonczewski(
                    a[node], self.gamma, J[node], Ms[node], m[x], m[y], m[z],
                    *self._slonczewski, dmx, dmy, dmz)
            if self.zhangli_status():
                dmx, dmy, dmz = _zhangli(
                    a[node], Ms[node], m[x], m[y], m[z], J[x], J[y], J[z],
                    *self._zhangli, dmx, dmy, dmz)

            dmdt[x] = dmx
            dmdt[y] = dmy
            dmdt[z] = dmz

    def solve(self):
        m = self.magnetisation.get_local()
        H = self.effective_field.get_local()
        dmdt = self.derivative.get_local()
        self._solve_arrays(m, H, dmdt)
        self.derivative.set_local(dmdt)
        self.derivative.apply("")

    def solve_with(self, vec_m, vec_H, vec_dmdt):
        m = vec_m.get_local()
        H = vec_H.get_local()
        dmdt = vec_dmdt.get_local()
        self._solve_arrays(m, H, dmdt)
        vec_dmdt.set_local(dmdt)
        vec_dmdt.apply("")

    def sundials_jtimes_serial(self, mp, Hp, jtimes):
        if self.alpha is None:
            raise RuntimeError("alpha was not set")

        m = self.magnetisation.get_local()
        H = self.effective_field.get_local()
        base = self.derivative.get_local()
        eps = 1e-8

        plus = base.copy()
        minus = base.copy()
        self._solve_arrays(m + eps * mp, H + eps * Hp, plus)
        self._solve_arrays(m - eps * mp, H - eps * Hp, minus)
        jtimes[:] = (plus - minus) / (2 * eps)


def get_python_equation_module():
    # Expose the Python fallback with the same module-like interface as the
    # compiled backend. Tests use this accessor directly for backend-parity
    # checks, and the runtime uses it automatically on stacks where the old
    # DOLFIN extension hook no longer exists. [Codex GPT-5.4]
    return SimpleNamespace(Equation=_PythonEquation)


def get_python_terms_module():
    # DOLFIN 2019 dropped compile_extension_module on the pixi path, so keep
    # a small Python fallback for the low-level term tests. [Codex GPT-5.4]
    return SimpleNamespace(
        damping=_damping,
        precession=_precession,
        relaxation=_relaxation,
    )

def find_slepc():
    slepc = None
    if 'SLEPC_DIR' in os.environ:
        slepc = os.environ['SLEPC_DIR']
    else:
        # At least on Ubuntu 16.04, the header files are in
        # /usr/lib/slepcdir/3.7.2/x86_64-linux-gnu-real/include/
        # However, tried to be a bit more robust to find it.
        slepcpath = '/usr/lib/slepcdir'
        matches = []
        if os.path.isdir(slepcpath):
            for root, dirnames, filenames in os.walk(slepcpath):
                for filename in fnmatch.filter(filenames, 'slepceps.h'):
                    matches.append(root)
                    
        # Dont want fortran header files!
        matches = [match for match in matches if 'finclude' not in match]
    if matches:
        slepc = matches[0]

    if not slepc:
        raise Exception("Cannot find SLEPc header files - please set environment variable SLEPC_DIR\n"
                        "You can also modify finmag/src/physics/equation.py")

    else:
        print("Found SLEPc include files at {}".format(slepc))
        return slepc
    
        
def find_petsc():
    petsc = None
    if 'SLEPC_DIR' in os.environ:
        petsc = os.environ['PETSC_DIR']
    else:
    # At least on Ubuntu 16.04, the header files are in
    # /usr/lib/slepcdir/3.7.2/x86_64-linux-gnu-real/include/
    # However, tried to be a bit more robust to find it.
        petscpath = '/usr/lib/petscdir'
        matches = []
        if os.path.isdir(petscpath):
            for root, dirnames, filenames in os.walk(petscpath):
                for filename in fnmatch.filter(filenames, 'petscsys.h'):
                    matches.append(root)
                    # Dont want fortran header files!
        matches = [match for match in matches if 'finclude' not in match]
    if matches:
        petsc = matches[0]

    if not petsc:
        raise Exception("Cannot find PETSc header files - please set environment variable PETSC_DIR\n"
                        "You can also modify finmag/src/physics/equation.py")

    else:
        print("Found PETSc include files at {}".format(petsc))
        return petsc


def native_equation_module_available():
    # The old low-level Equation/terms backend only exists on DOLFIN stacks
    # that still expose compile_extension_module. Keep this probe central so
    # the runtime and tests agree on when native parity checks are possible.
    # [Codex GPT-5.4]
    return hasattr(df, "compile_extension_module")


def _native_source_paths():
    module_dir = path.dirname(path.abspath(__file__))
    source_dir = path.join(module_dir, "native")
    cache_dir = path.join(module_dir, "build")
    return module_dir, source_dir, cache_dir


@lru_cache(maxsize=None)
def get_native_terms_module():
    if not native_equation_module_available():
        raise RuntimeError("The compiled equation backend is not available on this DOLFIN stack.")

    # Tests call this accessor directly when they want a native reference
    # result, instead of relying on the ambient default backend selection.
    # [Codex GPT-5.4]
    _, source_dir, _ = _native_source_paths()
    with open(path.join(source_dir, "terms.h"), "r") as header:
        code = header.read()

    return df.compile_extension_module(
        code=code,
        source_directory=source_dir,
        sources=["terms.cpp"],
        additional_declarations="%apply double& INOUT { double& dm_x, double& dm_y, double& dm_z };",
        include_dirs=[source_dir, find_petsc(), find_slepc()],)


def get_terms_module():
    # Keep the historical "give me the best available backend" behaviour for
    # existing callers, while the tests can explicitly request native or
    # Python implementations through the dedicated accessors above.
    # [Codex GPT-5.4]
    if native_equation_module_available():
        return get_native_terms_module()
    return get_python_terms_module()

# find_slepc()
# find_petsc()



# TODO: use field class objects instead of dolfin vectors
def Equation(m, H, dmdt):
    """
    Returns equation object initialised with dolfin vectors m, H and dmdt.

    """
    equation_module = get_equation_module(True)
    return equation_module.Equation(m, H, dmdt)


@lru_cache(maxsize=None)
def get_native_equation_module(for_distribution=False):
    if not native_equation_module_available():
        raise RuntimeError("The compiled equation backend is not available on this DOLFIN stack.")

    # Like get_native_terms_module(), this accessor exists so tests can still
    # reach the compiled implementation explicitly and compare it against the
    # fallback before that backend disappears from our active environments.
    # [Codex GPT-5.4]
    _, source_dir, cache_dir = _native_source_paths()
    signature = "equation" if for_distribution else ""

    equation_module = None
    if instant is not None:
        equation_module = instant.import_module("equation", cache_dir)
    if equation_module is not None:
        log.debug("Got equation extension module from distribution location.")
        return equation_module

    with open(path.join(source_dir, "equation.h"), "r") as header:
        code = header.read()

    return df.compile_extension_module(
        code=code,
        sources=["equation.cpp", "terms.cpp", "derivatives.cpp"],
        source_directory=source_dir,
        include_dirs=[source_dir, find_petsc(), find_slepc()],
        module_name=signature,
        cache_dir=cache_dir,)


def get_equation_module(for_distribution=False):
    """
    Returns extension module that deals with the equation of motion.
    Will try to return from cache before recompiling.

    By default, dolfin will chose a cache directory using a digest of our code
    and some version numbers. This procedure enables dolfin to detect changes
    to our code and recompile on the fly. However, when we distribute FinMag we
    don't need or want on the fly recompilation and we'd rather have the
    resulting files placed in a directory known ahead of time. For this, call
    this function once with `for_distribution` set to True and ship FinMag
    including the directory build/equation.

    During normal use, our known cache directory is always checked before
    dolfin's temporary ones. Its existence bypasses on the fly recompilation.

    On newer DOLFIN stacks where compile_extension_module is gone, this
    function transparently returns the Python fallback instead. The explicit
    native/Python accessors above exist so tests can still compare the two
    backends while the compiled one remains available in legacy CI.

    """
    if native_equation_module_available():
        return get_native_equation_module(for_distribution)
    return get_python_equation_module()
