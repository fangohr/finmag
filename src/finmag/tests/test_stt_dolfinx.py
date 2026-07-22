"""Production tests for the direct DOLFINx spin-transfer-torque port (Task 22).

Covers the Slonczewski/Xiao and Zhang-Li torques transcribed into the ported
``LLG`` NumPy right-hand side: coordinate-ordered legacy oracle fixtures (dm/dt
and, for Zhang-Li, the discrete gradient field), analytic direction/scaling
pins, ``Simulation`` pass-throughs, dynamics witnesses, and a scipy-vs-sundials
cross-backend check.
"""

import json
import os
import sys

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag import Simulation
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.field import Field
from finmag.physics.llg import LLG

FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SLON_FIXTURE = os.path.join(FIX_DIR, "slonczewski_rhs.json")
ZL_FIXTURE = os.path.join(FIX_DIR, "zhangli_rhs.json")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _spaces(domain):
    S1 = fem.functionspace(domain, ("Lagrange", 1))
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    return S1, S3


def _macrospin_llg(m, alpha=0.1, Ms=8.6e5, Hz=0.0, do_precession=True):
    """Uniform-state LLG on a single-cell cube (every node identical)."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, do_precession=do_precession, unit_length=1e-9)
    llg.Ms = Ms
    llg.set_alpha(alpha)
    llg.set_m(tuple(m), normalise=True)
    if Hz != 0.0:
        llg.effective_field.add(Zeeman((0.0, 0.0, Hz), name="Zeeman"))
    return llg


def _lexsort_rows(coords, *arrays):
    padded = np.zeros((coords.shape[0], 3))
    padded[:, : coords.shape[1]] = coords
    order = np.lexsort((padded[:, 2], padded[:, 1], padded[:, 0]))
    return (padded[order],) + tuple(a[order] for a in arrays)


def _nodal(dmdt_flat):
    return dmdt_flat.reshape((3, -1))


# --------------------------------------------------------------------------
# import boundary
# --------------------------------------------------------------------------

def test_stt_does_not_load_legacy_dolfin():
    # The ported LLG (STT included) is a pure NumPy transcription: the compiled
    # STT kernels are not rebuilt, and legacy FEniCS must never be imported.
    # (This file *does* opt into ``finmag.native.sundials`` for the cross-backend
    # check below, so it deliberately does not assert ``finmag.native`` absence;
    # ``test_llg_dolfinx.py`` pins that for the core RHS module.)
    assert LLG.__module__ == "finmag.physics.llg"
    assert "dolfin" not in sys.modules


# --------------------------------------------------------------------------
# Slonczewski: analytic direction / scaling pins
# --------------------------------------------------------------------------

def _slon_torque(m, J, alpha=0.1, Ms=8.6e5, P=0.4, d=2e-9, p=(0.0, 0.0, 1.0),
                 Lambda=2.0, epsilonprime=0.1):
    """Pure Slonczewski torque (no external field -> H_eff = 0)."""
    llg = _macrospin_llg(m, alpha=alpha, Ms=Ms)
    llg.use_slonczewski(J, P, d, p, Lambda=Lambda, epsilonprime=epsilonprime)
    return _nodal(llg.solve(0.0))[:, 0]


def test_slonczewski_torque_is_linear_in_current_density():
    m = (0.6, 0.0, 0.8)
    t1 = _slon_torque(m, 1.0e12)
    t2 = _slon_torque(m, 2.0e12)
    assert np.linalg.norm(t1) > 0.0
    assert np.allclose(t2, 2.0 * t1, rtol=1e-10, atol=0.0)


def test_slonczewski_torque_flips_sign_with_current():
    m = (0.6, 0.0, 0.8)
    tpos = _slon_torque(m, 1.0e12)
    tneg = _slon_torque(m, -1.0e12)
    assert np.allclose(tneg, -tpos, rtol=1e-10, atol=0.0)


def test_slonczewski_torque_vanishes_when_m_parallel_to_p():
    # m == p: both m x p and m x (m x p) vanish, so the whole torque is zero
    # (llg.cc:225-227), even with epsilonprime != 0. With H_eff = 0 and |m| = 1
    # the total dm/dt is therefore zero.
    t = _slon_torque((0.0, 0.0, 1.0), 1.0e12)
    assert np.allclose(t, 0.0, atol=1e-6)


# --------------------------------------------------------------------------
# Zhang-Li: analytic direction / scaling pins
# --------------------------------------------------------------------------

def _zhangli_interval(cells=8, x1=8.0, J=(1.0e12, 0.0, 0.0), P=0.5, beta=0.02,
                      alpha=0.1, Ms=8.6e5, add_field=False):
    domain = mesh.create_interval(MPI.COMM_WORLD, cells, [0.0, x1])
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    llg.Ms = Ms
    llg.set_alpha(alpha)
    llg.set_m(
        lambda x: np.vstack(
            (np.cos(0.4 * x[0]), np.sin(0.4 * x[0]), 0.5 * np.ones(x.shape[1]))
        ),
        normalise=True,
    )
    if add_field:
        llg.effective_field.add(Zeeman((0.0, 0.0, 1e4), name="Zeeman"))
    llg.use_zhangli(J_profile=J, P=P, beta=beta)
    return llg


def test_zhangli_stt_vanishes_for_uniform_m():
    # Uniform m -> (J.grad)m = 0 -> the adiabatic + non-adiabatic term is zero.
    # With no external field and |m| = 1 the total dm/dt is zero.
    domain = mesh.create_interval(MPI.COMM_WORLD, 8, [0.0, 8.0])
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    llg.Ms = 8.6e5
    llg.set_alpha(0.1)
    llg.set_m((0.0, 0.0, 1.0), normalise=True)
    llg.use_zhangli(J_profile=(1.0e12, 0.0, 0.0), P=0.5, beta=0.02)
    assert np.allclose(llg.solve(0.0), 0.0, atol=1e-3)


def test_zhangli_stt_is_linear_in_current_density():
    # No external field: dm/dt is purely the STT term, which is linear in J.
    t1 = _nodal(_zhangli_interval(J=(1.0e12, 0.0, 0.0)).solve(0.0))
    t2 = _nodal(_zhangli_interval(J=(2.0e12, 0.0, 0.0)).solve(0.0))
    assert np.linalg.norm(t1) > 0.0
    assert np.allclose(t2, 2.0 * t1, rtol=1e-9, atol=0.0)


def test_zhangli_stt_flips_sign_with_current():
    tpos = _nodal(_zhangli_interval(J=(1.0e12, 0.0, 0.0)).solve(0.0))
    tneg = _nodal(_zhangli_interval(J=(-1.0e12, 0.0, 0.0)).solve(0.0))
    assert np.allclose(tneg, -tpos, rtol=1e-9, atol=0.0)


# --------------------------------------------------------------------------
# oracle fixtures
# --------------------------------------------------------------------------

def test_slonczewski_rhs_matches_legacy_oracle_fixture():
    fixture = json.load(open(SLON_FIXTURE))
    params = fixture["physical_parameters"]
    cells = fixture["mesh"]["parameters"]["cells"]
    x1 = fixture["mesh"]["parameters"]["x1"]

    domain = mesh.create_interval(MPI.COMM_WORLD, cells, [0.0, x1])
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, do_precession=params["do_precession"]["value"],
              unit_length=params["unit_length"]["value"])
    llg.Ms = params["Ms"]["value"]
    llg.set_alpha(params["alpha"]["value"])
    llg.set_m(tuple(params["m_uniform"]["value"]), normalise=True)
    llg.effective_field.add(Zeeman(tuple(params["H_zeeman"]["value"]),
                                   name="Zeeman"))
    llg.use_slonczewski(
        params["J"]["value"], params["P"]["value"], params["d"]["value"],
        tuple(params["p"]["value"]), Lambda=params["Lambda"]["value"],
        epsilonprime=params["epsilonprime"]["value"])

    dmdt = llg.solve(0.0)

    cm, m_vals = llg.m_field.coords_and_values()
    Hf = Field(S3)
    # H_eff is component-blocked (``xxx``, Task 31); invert that ordering to
    # rebuild the Function (raw from_array would scramble the components).
    Hf.set_with_ordered_numpy_array_xxx(llg.effective_field.H_eff)
    _, H_vals = Hf.coords_and_values()
    _, dmdt_vals = llg._dmdt.coords_and_values()

    _, m_s, H_s, dmdt_s = _lexsort_rows(cm, m_vals, H_vals, dmdt_vals)

    by_name = {q["name"]: q for q in fixture["quantities"]}
    for name, got in (("m", m_s), ("effective_field", H_s), ("dmdt", dmdt_s)):
        q = by_name[name]
        ref = np.array(q["values"])
        tol = q["tolerances"]
        assert np.allclose(got, ref, atol=tol["absolute"], rtol=tol["relative"]), (
            "quantity {} mismatch:\n got {}\n ref {}".format(name, got, ref)
        )


def test_zhangli_rhs_and_gradient_match_legacy_oracle_fixture():
    fixture = json.load(open(ZL_FIXTURE))
    params = fixture["physical_parameters"]
    cells = fixture["mesh"]["parameters"]["cells"]
    x1 = fixture["mesh"]["parameters"]["x1"]

    domain = mesh.create_interval(MPI.COMM_WORLD, cells, [0.0, x1])
    S1, S3 = _spaces(domain)
    llg = LLG(S1, S3, do_precession=params["do_precession"]["value"],
              unit_length=params["unit_length"]["value"])
    llg.Ms = params["Ms"]["value"]
    llg.set_alpha(params["alpha"]["value"])
    llg.set_m(
        lambda x: np.vstack(
            (np.cos(0.4 * x[0]), np.sin(0.4 * x[0]), 0.5 * np.ones(x.shape[1]))
        ),
        normalise=True,
    )
    llg.effective_field.add(Exchange(params["A"]["value"], name="Exchange"))
    llg.effective_field.add(Zeeman(tuple(params["H_zeeman"]["value"]),
                                   name="Zeeman"))
    llg.use_zhangli(J_profile=tuple(params["J_profile"]["value"]),
                    P=params["P"]["value"], beta=params["beta"]["value"],
                    using_u0=params["using_u0"]["value"])

    dmdt = llg.solve(0.0)

    # u0 conversion pin (P mu_B / e / (1 + beta**2))
    assert llg.u0 == pytest.approx(params["u0"]["value"], rel=1e-12)

    # the discrete gradient operator, pinned directly against the frozen
    # LLG.compute_gradient_field output
    llg._Ms_node = llg._ms_nodal()
    H_gradm = llg._compute_zhangli_gradient()
    hg_field = Field(S3)
    hg_field.set_with_ordered_numpy_array_xxx(H_gradm.reshape(-1))

    cm, m_vals = llg.m_field.coords_and_values()
    Hf = Field(S3)
    # H_eff is component-blocked (``xxx``, Task 31); invert that ordering.
    Hf.set_with_ordered_numpy_array_xxx(llg.effective_field.H_eff)
    _, H_vals = Hf.coords_and_values()
    _, hg_vals = hg_field.coords_and_values()
    _, dmdt_vals = llg._dmdt.coords_and_values()

    _, m_s, H_s, hg_s, dmdt_s = _lexsort_rows(
        cm, m_vals, H_vals, hg_vals, dmdt_vals)

    by_name = {q["name"]: q for q in fixture["quantities"]}
    for name, got in (("m", m_s), ("effective_field", H_s),
                      ("H_gradm", hg_s), ("dmdt", dmdt_s)):
        q = by_name[name]
        ref = np.array(q["values"])
        tol = q["tolerances"]
        assert np.allclose(got, ref, atol=tol["absolute"], rtol=tol["relative"]), (
            "quantity {} mismatch:\n got {}\n ref {}".format(name, got, ref)
        )


# --------------------------------------------------------------------------
# Simulation pass-throughs
# --------------------------------------------------------------------------

def _box_sim(name="stt", backend=None):
    box = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [1, 1, 1],
        mesh.CellType.tetrahedron)
    kwargs = dict(unit_length=1e-9, name=name)
    if backend is not None:
        kwargs["integrator_backend"] = backend
    sim = Simulation(box, 8.6e5, **kwargs)
    return sim


def test_simulation_set_stt_activates_slonczewski():
    sim = _box_sim("slon_passthrough")
    sim.set_m((0.6, 0.0, 0.8))
    sim.set_stt(1.0e12, 0.4, 2e-9, (0.0, 0.0, 1.0), Lambda=2.0, epsilonprime=0.1)
    assert sim.llg.do_slonczewski is True
    assert sim.llg.do_zhangli is False
    # toggle_stt flips the flag off and on again
    sim.toggle_stt()
    assert sim.llg.do_slonczewski is False
    sim.toggle_stt(True)
    assert sim.llg.do_slonczewski is True


def test_simulation_set_zhangli_activates_zhangli():
    sim = _box_sim("zl_passthrough")
    sim.set_m((0.0, 0.0, 1.0))
    sim.set_zhangli(J_profile=(1.0e12, 0.0, 0.0), P=0.5, beta=0.02)
    assert sim.llg.do_zhangli is True
    assert sim.llg.do_slonczewski is False


# --------------------------------------------------------------------------
# dynamics witnesses
# --------------------------------------------------------------------------

def _domain_wall_m(x):
    delta = np.sqrt(13e-12 / 520e3) * 1e9
    xx = (x[0] - 50.0) / delta
    return np.vstack((-np.tanh(xx), 1.0 / np.cosh(xx), np.zeros_like(xx)))


def test_zhangli_domain_wall_displacement_witness():
    """Port of the legacy ``zhang_li_test.test_zhangli`` invariant: a positive
    current density displaces the domain wall so ``m_average[0]`` decreases from
    ~0 to a resolvably negative value (the wall moves along +u)."""
    domain = mesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 100.0])
    sim = Simulation(domain, Ms=8.6e5, unit_length=1e-9, name="zl_dw")
    sim.set_m(_domain_wall_m)
    sim.add(UniaxialAnisotropy(K1=520e3, axis=[1, 0, 0]))
    sim.add(Exchange(A=13e-12))
    sim.alpha = 0.01
    sim.set_zhangli((1.0e12, 0.0, 0.0), 0.5, 0.02)

    p0 = sim.m_average
    sim.run_until(2e-12)
    p1 = sim.m_average

    assert abs(p0[0]) < 1e-15
    assert p1[0] < p0[0]
    assert abs(p1[0]) > 1e-3


def test_slonczewski_tilt_direction_flips_with_current():
    """A short Slonczewski run tilts m toward or away from p depending on the
    sign of J: for m near +z (== p) the anti-damping/damping character of the
    torque flips with J, so the change in m_z has opposite sign for +J and -J."""
    def _run(J):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        sim = Simulation(domain, Ms=8.6e5, unit_length=1e-9, name="slon_dw")
        # m tilted 20 degrees from p = +z, in the x-z plane
        theta = np.deg2rad(20.0)
        sim.set_m((np.sin(theta), 0.0, np.cos(theta)))
        sim.add(Zeeman((0.0, 0.0, 1e5)))
        sim.alpha = 0.02
        sim.set_stt(J, 0.4, 2e-9, (0.0, 0.0, 1.0))
        mz0 = sim.m_average[2]
        sim.run_until(5e-11)
        return sim.m_average[2] - mz0

    d_pos = _run(5.0e12)
    d_neg = _run(-5.0e12)
    assert d_pos * d_neg < 0.0
    assert abs(d_pos) > 1e-4 and abs(d_neg) > 1e-4


# --------------------------------------------------------------------------
# cross-backend agreement (scipy vs native sundials)
# --------------------------------------------------------------------------

try:
    from finmag.drivers.llg_integrator import SundialsIntegrator
    import finmag.native.sundials as _native_sundials
except Exception:  # pragma: no cover - build absent
    SundialsIntegrator = None
    _native_sundials = None

requires_sundials = pytest.mark.skipif(
    SundialsIntegrator is None or _native_sundials is None,
    reason="native sundials extension is not available in this environment",
)


@requires_sundials
def test_zhangli_scipy_vs_sundials_agree():
    def _run(backend):
        domain = mesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 100.0])
        sim = Simulation(domain, Ms=8.6e5, unit_length=1e-9,
                         name="zl_" + backend, integrator_backend=backend)
        sim.set_m(_domain_wall_m)
        sim.add(UniaxialAnisotropy(K1=520e3, axis=[1, 0, 0]))
        sim.add(Exchange(A=13e-12))
        sim.alpha = 0.01
        sim.set_zhangli((1.0e12, 0.0, 0.0), 0.5, 0.02)
        sim.set_tol(reltol=1e-9, abstol=1e-11)
        sim.run_until(2e-12)
        return sim.m_average

    scipy_avg = _run("scipy")
    sundials_avg = _run("sundials")
    assert np.allclose(scipy_avg, sundials_avg, rtol=1e-4, atol=1e-6)
