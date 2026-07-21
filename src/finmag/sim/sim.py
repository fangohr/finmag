"""Core micromagnetic ``Simulation``, ported directly to DOLFINx.

This is the direct DOLFINx port of the legacy ``finmag.sim.sim.Simulation``. It
preserves the public core surface -- construction on a DOLFINx mesh with scalar
``Ms``/``unit_length``/``name``/scalar ``alpha``/scalar ``gamma``; ``set_m``,
``m``, ``m_field``, ``m_average``, ``t``, ``dmdt``; the interaction registry
(``add``/``get_interaction``/``interactions``/``remove_interaction`` and the
energy accessors); integrator creation/tolerances/``advance_time``/``run_until``/
``reset_time``/``reinit_integrator``; and the ``sim_with`` convenience factory
for Exchange, Zeeman and uniaxial anisotropy -- all driven through the ported
``LLG``/``EffectiveField``/``Field`` stack and the SciPy integrator. No compiled
``finmag.native`` extension, legacy ``dolfin``, scheduler, table writer, or
per-simulation log file is used.

Deliberate deviations from the legacy module (all documented in
``transition-notes.org`` and ``dev/dolfinx/porting_map.md``):

- ``mesh`` is a ``dolfinx.mesh.Mesh``; the CG1 scalar/vector spaces are built
  with ``dolfinx.fem.functionspace`` and there is no ``constrained_domain``.
- ``integrator_backend`` defaults to ``"scipy"`` (was ``"sundials"``) because
  the native Sundials/CVODE extension is unported; an explicit
  ``integrator_backend="sundials"`` still raises ``ImportError`` by name via
  ``llg_integrator``.
- ``m`` and ``dmdt`` return the component-blocked coordinate-ordered ``xxx``
  arrays (matching the ``LLG`` state-vector contract), not raw backend dofs.
- ``t`` reports ``0.0`` until an integrator exists rather than lazily creating
  one just to read the clock.
- No table writer / scheduler / per-simulation log file: NDT/VTK output,
  scheduling, restart, regions, hysteresis, normal modes, STT, PBC, demag, and
  the non-``llg`` kernels all raise ``NotImplementedError`` by name when
  requested, while never breaking import or the core paths.

[Claude Opus 4.8]
"""

import logging

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from finmag.field import Field
from finmag.physics.llg import LLG
from finmag.drivers.llg_integrator import llg_integrator
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman

log = logging.getLogger(name="finmag")


def _deferred(name, detail):
    """Raise a uniform, by-name ``NotImplementedError`` for a deferred surface."""
    raise NotImplementedError(
        "{}: {} is not part of the core DOLFINx Simulation port "
        "(deferred, see dev/dolfinx/porting_map.md).".format(name, detail)
    )


class Simulation(object):
    """Unified interface to finmag's micromagnetic simulation capabilities."""

    # Lightweight instance accounting is retained for API compatibility; it no
    # longer holds any cyclic references (no table writer / scheduler).
    instance_counter_max = 0
    instances = {}

    def __init__(self, mesh, Ms, unit_length=1, name="unnamed", kernel="llg",
                 integrator_backend="scipy", pbc=None, average=False,
                 parallel=False):
        """Create a core micromagnetic simulation.

        *Arguments*

          mesh : a ``dolfinx.mesh.Mesh``

          Ms : scalar saturation magnetisation (A/m)

          unit_length : distance (in metres) associated with mesh distance 1.0

          name : simulation name

          kernel : only ``'llg'`` is ported (``'sllg'``/``'llg_stt'`` deferred)

          integrator_backend : ``'scipy'`` (default) or ``'sundials'`` (deferred)

          pbc : periodic boundaries are deferred (only ``None`` is supported)
        """
        if pbc is not None:
            _deferred("pbc", "periodic boundary conditions ('1d'/'2d')")
        if parallel:
            _deferred(
                "parallel",
                "distributed/parallel (multi-rank) state; run serially",
            )

        self.name = name
        self.mesh = mesh
        self.unit_length = unit_length
        self.integrator_backend = integrator_backend
        self.pbc = None
        self._integrator = None

        # instance booking (no cyclic references retained)
        self.instance_id = Simulation.instance_counter_max
        Simulation.instance_counter_max += 1
        Simulation.instances[self.instance_id] = self

        log.info("Creating Sim object name='{}', instance_id={}.".format(
            self.name, self.instance_id))

        # CG1 scalar / three-component vector spaces (no constrained_domain).
        self.S1 = fem.functionspace(mesh, ("Lagrange", 1))
        self.S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))

        if kernel == "llg":
            self.llg = LLG(
                self.S1, self.S3, average=average, unit_length=unit_length)
        elif kernel == "sllg":
            _deferred("sllg", "the stochastic LLG kernel")
        elif kernel == "llg_stt":
            _deferred("llg_stt", "the spin-transfer-torque LLG kernel")
        else:
            raise ValueError("kernel must be one of llg, sllg or llg_stt.")

        self.kernel = kernel
        self.Ms = Ms

        # Mesh volume (in mesh units); DOLFINx assemble, not util.meshes.
        vol_local = fem.assemble_scalar(
            fem.form(fem.Constant(mesh, 1.0) * ufl.dx))
        self.Volume = mesh.comm.allreduce(vol_local, op=MPI.SUM)

        self.driver = "cvode"
        self.reltol = 1e-6
        self.abstol = 1e-6
        self.parallel = False

    def __str__(self):
        return "finmag.Simulation(name='{}', instance_id={}) with {}".format(
            self.name, self.instance_id, self.mesh)

    # -- magnetisation ------------------------------------------------------

    def __get_m(self):
        """The unit magnetisation as a component-blocked ``xxx`` array."""
        return self.llg._m_field.get_ordered_numpy_array_xxx()

    def set_m(self, value, normalise=True, **kwargs):
        """Set the magnetisation (normalised to unit length by default).

        ``value`` may be a constant tuple/list, a callable, a
        :class:`~finmag.field.Field`, a ``dolfinx.fem.Function``, or a flat
        component-blocked ``xxx`` NumPy array (see ``LLG.set_m``).
        """
        # ``debug`` was a legacy integrator-reinit flag; accepted and ignored.
        kwargs.pop("debug", None)
        self.llg.set_m(value, normalise=normalise, **kwargs)
        if self.has_integrator():
            self.reinit_integrator()

    m = property(__get_m, set_m)

    @property
    def Ms(self):
        return self.llg.Ms

    @Ms.setter
    def Ms(self, value):
        # ``LLG.Ms`` stores a DG0 Field that the EffectiveField/interactions
        # already reference in place, so the update propagates automatically.
        self.llg.Ms = value

    @property
    def m_field(self):
        return self.llg.m_field

    @property
    def m_average(self):
        """Volume-averaged magnetisation, :math:`\\frac{1}{V}\\int m\\,dV`."""
        return self.llg.m_average

    @property
    def dmdt(self):
        """dm/dt for all nodes as a component-blocked ``xxx`` array."""
        return self.llg._dmdt.get_ordered_numpy_array_xxx()

    @property
    def dmdt_max(self):
        """Max-L2-norm dm/dt vector across nodes (component-blocked order)."""
        dmdts = self.dmdt.reshape((3, -1))
        norms = np.sqrt(np.sum(dmdts ** 2, axis=0))
        index = norms.argmax()
        return np.array(
            [dmdts[0][index], dmdts[1][index], dmdts[2][index]])

    # -- damping / precession / gamma --------------------------------------

    @property
    def alpha(self):
        """The scalar Gilbert damping constant :math:`\\alpha`."""
        return self.llg.alpha

    @alpha.setter
    def alpha(self, value):
        self.llg.set_alpha(value)

    @property
    def do_precession(self):
        return self.llg.do_precession

    @do_precession.setter
    def do_precession(self, value):
        self.llg.do_precession = value

    def __get_gamma(self):
        return self.llg.gamma

    def __set_gamma(self, value):
        self.llg.gamma = value

    gamma = property(__get_gamma, __set_gamma)

    def __get_pins(self):
        return self.llg.pins

    def __set_pins(self, nodes):
        if callable(nodes):
            _deferred(
                "pins",
                "callable pin masks (coordinate->dof mapping)",
            )
        self.llg.pins = nodes

    pins = property(__get_pins, __set_pins)

    # -- interactions -------------------------------------------------------

    def add(self, interaction, with_time_update=None):
        """Add an interaction (e.g. Exchange, Zeeman, UniaxialAnisotropy)."""
        self.llg.effective_field.add(interaction, with_time_update)

    def effective_field(self):
        """Compute and return the total effective field (``xxx`` owned array)."""
        return self.llg.effective_field.compute(self.t)

    def total_energy(self):
        """Total energy of all interactions present in the simulation."""
        return self.llg.effective_field.total_energy()

    def compute_energy(self, name="total"):
        """Energy of a named interaction, or ``'total'`` for the whole system."""
        if name.lower() == "total":
            return self.total_energy()
        return self.get_interaction(name).compute_energy()

    def has_interaction(self, interaction_name):
        return self.llg.effective_field.exists(interaction_name)

    def interactions(self):
        """Return the sorted names of the known interactions."""
        return self.llg.effective_field.all()

    def get_interaction(self, interaction_name):
        """Return the interaction object registered under ``interaction_name``."""
        return self.llg.effective_field.get(interaction_name)

    def get_interaction_list(self):
        """Return a list of interaction names."""
        return self.llg.effective_field.all()

    def remove_interaction(self, interaction_type):
        """Remove the interaction registered under ``interaction_type``."""
        log.debug("Removing interaction '{}' from simulation '{}'".format(
            interaction_type, self.name))
        return self.llg.effective_field.remove(interaction_type)

    def set_H_ext(self, H_ext):
        """Convenience: set (or create) the external Zeeman field."""
        if self.has_interaction("Zeeman"):
            self.get_interaction("Zeeman").set_value(H_ext)
        else:
            self.add(Zeeman(H_ext))

    def switch_off_H_ext(self, remove_interaction=False):
        """Convenience: zero (default) or remove the external Zeeman field."""
        if remove_interaction:
            self.remove_interaction("Zeeman")
        else:
            self.get_interaction("Zeeman").set_value([0, 0, 0])

    def get_field_as_dolfin_function(self, field_type, region=None):
        """Return an interaction field (or ``m``) as a DOLFINx ``Function``."""
        if region is not None:
            _deferred(
                "get_field_as_dolfin_function",
                "region-restricted field extraction",
            )
        if field_type == "m":
            return self.llg._m_field.f
        return self.llg.effective_field.get_dolfin_function(field_type)

    # -- integrator ---------------------------------------------------------

    def has_integrator(self):
        return self._integrator is not None

    def _get_integrator(self):
        if not self.has_integrator():
            self.create_integrator()
        return self._integrator

    def _set_integrator(self, value):
        self._integrator = value

    integrator = property(_get_integrator, _set_integrator)

    def create_integrator(self, backend=None, **kwargs):
        if backend is not None:
            self.integrator_backend = backend

        if self.has_integrator():
            log.warning(
                "Cannot create integrator - one exists already: {}".format(
                    self._integrator))
            return self._integrator

        self._integrator = llg_integrator(
            self.llg, self.llg._m_field, backend=self.integrator_backend,
            reltol=self.reltol, abstol=self.abstol, **kwargs)
        return self._integrator

    def set_tol(self, reltol=1e-6, abstol=1e-6):
        """Set the relative/absolute tolerances of the default integrator.

        If an integrator already exists, its tolerances are updated and it is
        reinitialised so the new values take effect (the SciPy driver has no
        in-place ``set_scalar_tolerances`` counterpart to the native one).
        """
        self.reltol = reltol
        self.abstol = abstol
        if self.has_integrator():
            self._integrator.reltol = reltol
            self._integrator.abstol = abstol
            self._integrator.reinit()

    def advance_time(self, t):
        """Advance the integrator to time ``t`` (no schedule)."""
        log.debug("Advancing time to t = {} s.".format(t))
        self.integrator.advance_time(t)
        # The integrator may slightly overshoot; pin the fields to exactly t.
        self.llg.effective_field.update(t)

    def run_until(self, t):
        """Run the simulation until physical time ``t`` is reached.

        Scheduler integration is Task 12 scope; this advances physical time
        directly through the integrator without any scheduler features.
        """
        log.info("Simulation will run until t = {:.2g} s.".format(t))
        self.t_max = t
        self.integrator.advance_time(t)
        self.llg.effective_field.update(t)
        log.info("Simulation has reached time t = {:.2g} s.".format(self.t))

    @property
    def t(self):
        """The current simulation time (``0.0`` until an integrator exists)."""
        if self.has_integrator():
            return self._integrator.cur_t
        return 0.0

    def reset_time(self, t0):
        """Reset the simulation clock to ``t0``, rebuilding the integrator."""
        integrator = llg_integrator(
            self.llg, self.llg._m_field, backend=self.integrator_backend,
            reltol=self.reltol, abstol=self.abstol)
        # The SciPy driver takes no ``t0`` kwarg; reseed its state explicitly.
        integrator.cur_t = t0
        integrator.ode.set_initial_value(
            self.llg._m_field.get_ordered_numpy_array_xxx(), t0)
        self._integrator = integrator
        assert self.t == t0

    def reinit_integrator(self):
        """Reinitialise the integrator from the current field state, if any."""
        if self.has_integrator():
            self._integrator.reinit()
        else:
            log.warning("Integrator reinit requested, but none is present.")

    # -- explicitly deferred surfaces --------------------------------------
    #
    # These preserve the legacy public names but fail by name when requested,
    # so the core import graph stays clean and callers get a clear error
    # instead of a NameError or a silent no-op.

    def schedule(self, *args, **kwargs):
        _deferred("schedule", "the scheduler API (Task 12)")

    def unschedule(self, *args, **kwargs):
        _deferred("unschedule", "the scheduler API (Task 12)")

    def clear_schedule(self, *args, **kwargs):
        _deferred("clear_schedule", "the scheduler API (Task 12)")

    def save_restart_data(self, *args, **kwargs):
        _deferred("save_restart_data", "restart persistence (Task 12)")

    def restart(self, *args, **kwargs):
        _deferred("restart", "restart loading (Task 12)")

    def save_averages(self, *args, **kwargs):
        _deferred("save_averages", "NDT table output")

    def save_ndt(self, *args, **kwargs):
        _deferred("save_ndt", "NDT table output")

    def save_m(self, *args, **kwargs):
        _deferred("save_m", "field snapshot output")

    def save_field(self, *args, **kwargs):
        _deferred("save_field", "field snapshot output")

    def save_vtk(self, *args, **kwargs):
        _deferred("save_vtk", "VTK output")

    def save_field_to_vtk(self, *args, **kwargs):
        _deferred("save_field_to_vtk", "VTK output")

    def snapshot(self, *args, **kwargs):
        _deferred("snapshot", "VTK output (use save_vtk)")

    def render_scene(self, *args, **kwargs):
        _deferred("render_scene", "ParaView rendering")

    def plot_dynamics(self, *args, **kwargs):
        _deferred("plot_dynamics", "NDT plotting")

    def plot_dynamics_3d(self, *args, **kwargs):
        _deferred("plot_dynamics_3d", "NDT plotting")

    def plot_mesh(self, *args, **kwargs):
        _deferred("plot_mesh", "mesh plotting")

    def mark_regions(self, *args, **kwargs):
        _deferred("mark_regions", "region/material machinery")

    def save_m_in_region(self, *args, **kwargs):
        _deferred("save_m_in_region", "region/material machinery")

    def get_submesh(self, *args, **kwargs):
        _deferred("get_submesh", "region/material machinery")

    def probe_field(self, *args, **kwargs):
        _deferred("probe_field", "point probing")

    def probe_field_along_line(self, *args, **kwargs):
        _deferred("probe_field_along_line", "point probing")

    def relax(self, *args, **kwargs):
        _deferred("relax", "the relaxation driver")

    def hysteresis(self, *args, **kwargs):
        _deferred("hysteresis", "the hysteresis driver")

    def hysteresis_loop(self, *args, **kwargs):
        _deferred("hysteresis_loop", "the hysteresis driver")

    def run_normal_modes_computation(self, *args, **kwargs):
        _deferred("run_normal_modes_computation", "normal-mode analysis")

    def set_stt(self, *args, **kwargs):
        _deferred("set_stt", "Slonczewski spin-transfer torque")

    def toggle_stt(self, *args, **kwargs):
        _deferred("toggle_stt", "Slonczewski spin-transfer torque")

    def set_zhangli(self, *args, **kwargs):
        _deferred("set_zhangli", "Zhang-Li spin-transfer torque")


def sim_with(mesh, Ms, m_init, alpha=0.5, unit_length=1,
             integrator_backend="scipy", A=None, K1=None, K1_axis=None,
             H_ext=None, demag_solver="FK", demag_solver_type=None, nx=None,
             ny=None, spacing_x=None, spacing_y=None, demag_solver_params=None,
             D=None, name="unnamed", pbc=None, sim_class=Simulation):
    """Create a :class:`Simulation` and add the requested ported interactions.

    Exchange (``A``), uniaxial anisotropy (``K1`` + ``K1_axis``) and Zeeman
    (``H_ext``) are ported. Demag (``demag_solver``, default ``'FK'``) and DMI
    (``D``) are not ported and raise ``NotImplementedError`` by name when
    requested; pass ``demag_solver=None`` to build a demag-free simulation.
    """
    sim = sim_class(mesh, Ms, unit_length=unit_length,
                    integrator_backend=integrator_backend, name=name, pbc=pbc)

    sim.set_m(m_init)
    sim.alpha = alpha

    if A is not None:
        sim.add(Exchange(A))
    if (K1 is not None and K1_axis is None) or (K1 is None and K1_axis is not None):
        log.warning(
            "Not initialising uniaxial anisotropy because only one of K1, "
            "K1_axis was specified (values given: K1={}, K1_axis={}).".format(
                K1, K1_axis))
    if K1 is not None and K1_axis is not None:
        sim.add(UniaxialAnisotropy(K1, K1_axis))
    if H_ext is not None:
        sim.add(Zeeman(H_ext))
    if D is not None:
        _deferred("sim_with(D=...)", "DMI")
    if demag_solver is not None:
        _deferred(
            "sim_with(demag_solver={!r})".format(demag_solver),
            "demag (pass demag_solver=None for a demag-free simulation)",
        )
    log.debug("Successfully created simulation '{}'".format(sim.name))
    return sim
