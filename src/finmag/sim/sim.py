"""Core micromagnetic ``Simulation``, ported directly to DOLFINx.

This is the direct DOLFINx port of the legacy ``finmag.sim.sim.Simulation``. It
preserves the public core surface -- construction on a DOLFINx mesh with scalar
or spatially varying ``Ms``/``alpha`` (Task 16), scalar ``unit_length``/``name``/
``gamma``; ``set_m``, ``m``, ``m_field``, ``m_average``, ``t``, ``dmdt``; the
interaction registry (``add``/``get_interaction``/``interactions``/
``remove_interaction`` and the
energy accessors); integrator creation/tolerances/``advance_time``/``run_until``/
``reset_time``/``reinit_integrator``; and the ``sim_with`` convenience factory
for Exchange, Zeeman, uniaxial anisotropy and DMI -- all driven through the ported
``LLG``/``EffectiveField``/``Field`` stack and the SciPy integrator. FK demag
(via the compiled ``finmag.native.bem_arrays`` extension), the coordinate-aware
v2 ``.npz`` restart format, scheduler-driven ``run_until``, NDT output (via
``Tablewriter``) and write-only VTK/XDMF field output are all ported and
supported (Tasks 11b/12). No legacy ``dolfin`` is used anywhere in this module.

Deliberate deviations from the legacy module (all documented in
``transition-notes.org`` and ``dev/dolfinx/porting_map.md``):

- ``mesh`` is a ``dolfinx.mesh.Mesh``; the CG1 scalar/vector spaces are built
  with ``dolfinx.fem.functionspace`` and there is no ``constrained_domain``.
- ``integrator_backend`` defaults to ``"scipy"`` (was ``"sundials"``). Task 20
  ported and fully validated the native Sundials/CVODE extension on DOLFINx
  (``integrator_backend="sundials"`` now works end-to-end when the extension
  is built, and ``llg_integrator``'s own factory default was restored to
  ``"sundials"``), but ``Simulation.integrator_backend``'s default is
  deliberately kept at ``"scipy"`` -- USER ACCEPTANCE PENDING, see
  ``transition-notes.org`` ("Native Sundials/CVODE on DOLFINx (Task 20)") and
  ``dev/dolfinx/porting_map.md``. Without the native extension built,
  ``integrator_backend="sundials"`` still raises ``ImportError`` by name via
  ``llg_integrator``.
- ``m`` and ``dmdt`` return the component-blocked coordinate-ordered ``xxx``
  arrays (matching the ``LLG`` state-vector contract), not raw backend dofs.
- ``t`` reports ``0.0`` until an integrator exists rather than lazily creating
  one just to read the clock.
- Scheduling, restart, NDT and VTK/XDMF output are ported and supported (see
  ``Simulation.schedule``/``run_until``, ``Simulation.save_restart_data``/
  ``restart``, and ``Tablewriter``/``FieldSaver``). ``relax``, ``hysteresis``
  and ``hysteresis_loop`` are also ported and supported (Task 15; the
  untouched legacy ``sim_relax.py``/``hysteresis.py`` modules bound the same
  way legacy did). What remains deferred and raises ``NotImplementedError``
  by name when requested: the PBC/treecode/GCR demag variants, STT
  (``set_stt``/``set_zhangli``), the ``sllg``/thermal kernel, normal modes,
  and ``parallel=True`` -- none of these ever break import or the core ``llg``
  paths. ``integrator_backend="sundials"`` is no longer deferred (Task 20): it
  works end-to-end when the native extension is built, and only raises
  ``ImportError`` by name (unchanged failure mode) when it is not. Region
  accounting (``mark_regions`` + per-region energy/magnetisation)
  is ported (Task 16); region-restricted submesh *field output*
  (``save_m_in_region``/``get_submesh``/``get_field_as_dolfin_function
  (region=...)``) remains deferred by name.

[Claude Opus 4.8], [Claude Sonnet 5]
"""

import logging

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI

from finmag.field import Field
from finmag.physics.llg import LLG
from finmag.drivers.llg_integrator import llg_integrator
from finmag.energies import DMI, Exchange, UniaxialAnisotropy, Zeeman
from finmag.sim import sim_helpers
from finmag.sim import sim_relax
from finmag.sim.hysteresis import hysteresis as _hysteresis
from finmag.sim.hysteresis import hysteresis_loop as _hysteresis_loop
from finmag.util.fileio import Tablewriter, FieldSaver
from finmag.scheduler import scheduler

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

          integrator_backend : ``'scipy'`` (default) or ``'sundials'`` (Task 20:
            fully ported and supported when the native extension is built;
            raises ``ImportError`` by name otherwise). The ``"scipy"`` default
            here is temporary -- see the Task 20 "USER ACCEPTANCE PENDING"
            register entry in ``transition-notes.org``.

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
        self.sanitized_name = sim_helpers.clean_filename(name)
        self.ndtfilename = self.sanitized_name + ".ndt"
        self.mesh = mesh
        self.unit_length = unit_length
        self.integrator_backend = integrator_backend
        self.pbc = None
        self._integrator = None

        # Output / scheduling state (Task 12). The table writer is created
        # lazily on first NDT save so a simulation that never writes an .ndt
        # file does not touch the filesystem. [Claude Opus 4.8]
        self._tablewriter = None
        self.field_savers = {}
        self.scheduler = scheduler.Scheduler()
        self.callbacks_at_scheduler_events = []
        self.scheduler_shortcuts = {
            "save_averages": sim_helpers.save_ndt,
            "save_ndt": sim_helpers.save_ndt,
            "save_restart_data": sim_helpers.save_restart_data,
            "save_vtk": None,  # handled specially in schedule()
            "save_field": None,  # handled specially in schedule()
            "eta": sim_helpers.eta,
            "ETA": sim_helpers.eta,
        }

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

        self.driver = "scipy"
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
        """The Gilbert damping constant :math:`\\alpha`.

        Returns a Python ``float`` when uniform (preserving the legacy scalar
        contract) or a per-node array when spatially varying (Task 16); see
        ``finmag.physics.llg.LLG.alpha``.
        """
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

    def compute_energy(self, name="total", region=None):
        """Energy of a named interaction, or ``'total'`` for the whole system.

        With ``region`` (a region id passed earlier to :meth:`mark_regions`)
        the energy is integrated only over that region. Region energies of an
        interaction sum to its whole-mesh energy (the legacy
        ``test_energies_in_regions`` additivity invariant). ``'total'`` with a
        region sums every interaction's region energy.
        """
        if region is not None:
            measure = self.region_measure(region)
            if name.lower() == "total":
                return sum(
                    self.get_interaction(n).compute_energy(dx=measure)
                    for n in self.interactions()
                )
            return self.get_interaction(name).compute_energy(dx=measure)
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

        Any scheduled actions (see :meth:`schedule`) registered on this
        simulation are triggered at their scheduled times as the integrator
        advances through them, matching the legacy scheduler-driven loop. With
        no schedule this reduces to advancing the integrator directly to ``t``.
        """
        log.info("Simulation will run until t = {:.2g} s.".format(t))
        self.t_max = t

        # Ensure the integrator exists before the scheduler drives it.
        self.integrator

        # A stop event terminates the scheduler loop at t. ``at_end=True`` is
        # required because ``t`` can be zero (treated as falsey by ``add``).
        def call_to_end_integration():
            return False

        self.scheduler.add(call_to_end_integration, at=t, at_end=True)
        self.scheduler.run(self.integrator, self.callbacks_at_scheduler_events)

        # The time integrator may slightly overshoot the requested end time;
        # pin the fields to exactly t.
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
        self.scheduler.reset(t0)
        assert self.t == t0

    def reinit_integrator(self):
        """Reinitialise the integrator from the current field state, if any."""
        if self.has_integrator():
            self._integrator.reinit()
        else:
            log.warning("Integrator reinit requested, but none is present.")

    # -- restart persistence (Task 12) -------------------------------------

    def save_restart_data(self, filename=None):
        """Save the current magnetisation, time and metadata to a restart file.

        The magnetisation is stored in the coordinate-aware format defined in
        :mod:`finmag.sim.sim_helpers` (a deliberate deviation from the legacy
        raw-dof npz layout; see that module's docstring).
        """
        sim_helpers.save_restart_data(self, filename)

    def restart(self, filename=None, t0=None):
        """Reload magnetisation and time from a restart file.

        With no ``filename`` the canonical ``<name>-restart.npz`` is used. The
        magnetisation is remapped onto the current mesh by coordinate; a mesh
        mismatch raises ``ValueError`` rather than silently misassigning. The
        restart time is taken from the file unless ``t0`` overrides it.
        """
        if filename is None:
            filename = sim_helpers.canonical_restart_filename(self)
        log.debug("Loading restart data from {}.".format(filename))

        data = sim_helpers.load_restart_data(filename)
        if data.get("driver") not in ("scipy", "cvode"):
            raise NotImplementedError(
                "Unknown driver {!r} for restarting.".format(data.get("driver")))

        sim_helpers.apply_restart_magnetisation(self.llg._m_field, data)
        self.reset_time(data["simtime"] if t0 is None else t0)
        log.info("Reloaded m (<m>=%s) and time=%s from %s." % (
            self.llg.m_average, self.t, filename))

    # -- NDT table output (Task 12) ----------------------------------------

    @property
    def tablewriter(self):
        """The lazily-created NDT :class:`~finmag.util.fileio.Tablewriter`."""
        if self._tablewriter is None:
            self._tablewriter = Tablewriter(
                self.ndtfilename, self, override=True)
        return self._tablewriter

    def save_averages(self, *args, **kwargs):
        """Save the spatial averages (magnetisation etc.) to the .ndt file."""
        sim_helpers.save_ndt(self)

    # ``save_ndt`` is the historical alias of ``save_averages``.
    save_ndt = save_averages

    # -- field snapshot output (.npy) (Task 12) ----------------------------

    def _get_field_saver(self, field_name, filename=None, overwrite=False,
                         incremental=False):
        if filename is None:
            filename = "{}_{}.npy".format(
                self.sanitized_name, field_name.lower())
        if not filename.endswith(".npy"):
            filename += ".npy"
        saver = self.field_savers.get(filename)
        if saver is not None and saver.incremental == incremental:
            return saver
        saver = FieldSaver(filename, overwrite=overwrite, incremental=incremental)
        self.field_savers[filename] = saver
        return saver

    def save_field(self, field_name, filename=None, incremental=False,
                   overwrite=False, region=None):
        """Save a field ('m' or an interaction) to a .npy file.

        The saved array is the stable coordinate-ordered ``xyz`` view (a
        deliberate deviation from the legacy raw-dof ``get_local()`` layout,
        for the same coordinate-stability reason as restart).
        """
        if region is not None:
            _deferred("save_field", "region-restricted field snapshots")
        if field_name == "m":
            _coords, values = self.llg._m_field.coords_and_values()
        else:
            fld = Field(
                self.S3,
                self.llg.effective_field.get_dolfin_function(field_name))
            _coords, values = fld.coords_and_values()
        saver = self._get_field_saver(
            field_name, filename, incremental=incremental, overwrite=overwrite)
        saver.save(np.asarray(values))

    def save_m(self, filename=None, incremental=False, overwrite=False):
        """Convenience wrapper: save the magnetisation to a .npy file."""
        self.save_field(
            "m", filename=filename, incremental=incremental, overwrite=overwrite)

    # -- VTK / XDMF output (Task 12) ---------------------------------------
    #
    # These route through the ported, write-only ``Field`` VTK/XDMF writers.
    # Read-back is explicitly unavailable (``Field.save_hdf5``/read paths raise
    # by name); DOLFINx VTK/XDMF output has no legacy read-back either.

    def save_vtk(self, filename=None, overwrite=False, region=None):
        """Append the magnetisation to a VTK/PVD (or XDMF) time series."""
        self.save_field_to_vtk(
            "m", filename=filename, overwrite=overwrite, region=region)

    def save_field_to_vtk(self, field_name, filename=None, overwrite=False,
                          region=None):
        """Append the named field to a VTK/PVD (or XDMF) time series."""
        if region is not None:
            _deferred("save_field_to_vtk", "region-restricted VTK output")
        if filename is None:
            filename = self.sanitized_name + ".pvd"
        if field_name == "m":
            fld = self.llg._m_field
        else:
            fld = Field(
                self.S3,
                self.llg.effective_field.get_dolfin_function(field_name),
                name=field_name)
        if filename.endswith(".xdmf"):
            fld.save_xdmf(filename, self.t)
        else:
            fld.save_pvd(filename, self.t)

    # -- scheduler (Task 12) -----------------------------------------------

    def schedule(self, func, *args, **kwargs):
        """Register an action to be called during ``run_until``.

        ``func`` may be a callable ``func(sim, *args)`` or one of the supported
        shortcut strings: ``'save_ndt'``/``'save_averages'``, ``'save_vtk'``,
        ``'save_field'``, ``'save_restart_data'``, ``'eta'``/``'ETA'``. Use the
        ``at``/``every``/``after``/``at_end`` keywords to place the action in
        time (see :class:`finmag.scheduler.scheduler.Scheduler`). Unknown
        shortcut strings raise ``KeyError`` by name.
        """
        if isinstance(func, str):
            if func not in self.scheduler_shortcuts:
                raise KeyError(
                    "Scheduling keyword '{}' unknown. Known values are {}".format(
                        func, sorted(self.scheduler_shortcuts.keys())))
            if func == "save_vtk":
                filename = kwargs.pop("filename", None)
                overwrite = kwargs.pop("overwrite", False)
                func = lambda sim: sim.save_field_to_vtk(
                    "m", filename=filename, overwrite=overwrite)
            elif func == "save_field":
                func = lambda sim, *a, **kw: sim.save_field(
                    *a, incremental=True, **kw)
            elif func in ("eta", "ETA"):
                import time as _time

                eta_fn = self.scheduler_shortcuts[func]
                started = _time.time()
                func = lambda sim: eta_fn(sim, when_started=started)
            else:
                func = self.scheduler_shortcuts[func]

        at = kwargs.pop("at", None)
        every = kwargs.pop("every", None)
        after = kwargs.pop("after", self.t if every is not None else None)
        at_end = kwargs.pop("at_end", False)
        realtime = kwargs.pop("realtime", False)

        return self.scheduler.add(
            func, [self] + list(args), kwargs, at=at, at_end=at_end,
            every=every, after=after, realtime=realtime)

    def unschedule(self, item):
        """Remove a previously scheduled item (as returned by ``schedule``)."""
        self.scheduler._remove(item)

    def clear_schedule(self):
        """Remove all scheduled actions and reset the scheduler clock."""
        self.scheduler.clear()
        self.scheduler.reset(self.t)

    # -- explicitly deferred surfaces --------------------------------------
    #
    # These preserve the legacy public names but fail by name when requested,
    # so the core import graph stays clean and callers get a clear error
    # instead of a NameError or a silent no-op.

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

    # -- regions (Task 16) --------------------------------------------------

    def mark_regions(self, fun_regions):
        """Partition the mesh into regions by a function ``fun_regions(pt) -> id``.

        ``fun_regions`` maps a point (a length-3 coordinate array) to a hashable
        region id; each *cell* is assigned the region of its midpoint. This builds:

        - ``self.region_ids``: an ordered ``{user_id: contiguous_int}`` map
          (matching the legacy ``mark_regions`` contiguous remap);
        - ``self.region_markers``: a per-cell :class:`dolfinx.mesh.MeshTags`;
        - a subdomain-restricted measure (see :meth:`region_measure`).

        Per-region energy and magnetisation accounting then work through
        :meth:`compute_energy` (``region=...``) and
        :meth:`m_average_in_region`. Region-restricted *field extraction* and
        submesh output (``get_field_as_dolfin_function(region=...)``,
        ``save_m_in_region``, ``get_submesh``) remain deferred by name.

        Serial only, consistent with the rest of the DOLFINx state contract.
        """
        if self.mesh.comm.size > 1:
            _deferred("mark_regions", "distributed (multi-rank) region tagging")
        tdim = self.mesh.topology.dim
        n_cells = self.mesh.topology.index_map(tdim).size_local
        cell_indices = np.arange(n_cells, dtype=np.int32)
        midpoints = dmesh.compute_midpoints(self.mesh, tdim, cell_indices)
        raw_ids = [fun_regions(pt) for pt in midpoints]

        ordered_ids = list(dict.fromkeys(raw_ids))
        self.region_ids = {region: i for i, region in enumerate(ordered_ids)}
        markers = np.array(
            [self.region_ids[r] for r in raw_ids], dtype=np.int32)
        self.region_markers = dmesh.meshtags(
            self.mesh, tdim, cell_indices, markers)
        self._region_dx = ufl.Measure(
            "dx", domain=self.mesh, subdomain_data=self.region_markers)
        log.debug("Marked {} region(s) on simulation '{}'.".format(
            len(self.region_ids), self.name))
        return self.region_ids

    def region_measure(self, region):
        """Return the UFL ``dx`` measure restricted to a marked ``region``."""
        if not hasattr(self, "_region_dx"):
            raise RuntimeError("call mark_regions(...) before region_measure")
        if region not in self.region_ids:
            raise KeyError("unknown region id {!r}; known: {}".format(
                region, sorted(self.region_ids, key=repr)))
        return self._region_dx(self.region_ids[region])

    def m_average_in_region(self, region):
        """Volume-averaged magnetisation over a marked ``region``."""
        return self.llg.m_average_fun(dx=self.region_measure(region))

    def save_m_in_region(self, *args, **kwargs):
        _deferred("save_m_in_region", "region-restricted submesh field output")

    def get_submesh(self, *args, **kwargs):
        _deferred("get_submesh", "region/material machinery")

    def probe_field(self, *args, **kwargs):
        _deferred("probe_field", "point probing")

    def probe_field_along_line(self, *args, **kwargs):
        _deferred("probe_field_along_line", "point probing")

    # -- topological charge (Task 26a) --------------------------------------
    #
    # Restored directly against DOLFINx/UFL (removed outright in Task 9,
    # accepted judgment-call deferral; see dev/dolfinx/porting_map.md). Thin
    # delegators to finmag.sim.sim_helpers, matching the style already used
    # for save_restart_data/save_ndt/etc. above (legacy bound these directly
    # as `skyrmion_number = sim_helpers.skyrmion_number`).

    def skyrmion_number(self):
        """Skyrmion number (topological charge) of the current state.

        See ``finmag.sim.sim_helpers.skyrmion_number`` for the formula and
        the 2D/3D-top-surface integration-domain rule.
        """
        return sim_helpers.skyrmion_number(self)

    def skyrmion_number_density_function(self):
        """Skyrmion-number density as a lumped nodal ``dolfinx.fem.Function``.

        See ``finmag.sim.sim_helpers.skyrmion_number_density_function``.
        """
        return sim_helpers.skyrmion_number_density_function(self)

    # -- relaxation / hysteresis (Task 15) ---------------------------------
    #
    # Bound exactly as the legacy ``Simulation`` did (``relax =
    # sim_relax.relax``, ``hysteresis = hyst``, ``hysteresis_loop =
    # hyst_loop``): these are the untouched legacy ``sim_relax.py`` /
    # ``hysteresis.py`` modules (only their ``finmag.util.helpers`` import was
    # replaced by a local dolfin-free reimplementation; see those modules'
    # docstrings), driven through the ported scheduler/integrator/
    # EffectiveField stack. [Claude Sonnet 5]
    relax = sim_relax.relax
    hysteresis = _hysteresis
    hysteresis_loop = _hysteresis_loop

    def run_normal_modes_computation(self, *args, **kwargs):
        _deferred("run_normal_modes_computation", "normal-mode analysis")

    def set_stt(self, current_density, polarisation, thickness, direction,
                Lambda=2, epsilonprime=0.0, with_time_update=None):
        """Activate the Slonczewski spin-transfer torque in the LLG (Task 22).

        Pass-through to :meth:`finmag.physics.llg.LLG.use_slonczewski`, faithful
        to the legacy signature: current density in A/m^2 (number, callable,
        Field or Function), polarisation in [0, 1], free-layer thickness in m,
        polarisation direction (normalised), the Slonczewski/Xiao ``Lambda`` and
        secondary-torque ``epsilonprime``, and an optional ``J(t)`` returning a
        spatially uniform current density.
        """
        self.llg.use_slonczewski(
            current_density, polarisation, thickness, direction,
            Lambda=Lambda, epsilonprime=epsilonprime,
            with_time_update=with_time_update)

    def toggle_stt(self, new_state=None):
        """Toggle the Slonczewski spin-transfer torque (legacy semantics).

        With no argument the ``do_slonczewski`` flag is flipped; pass a boolean
        to set it explicitly.
        """
        if new_state:
            self.llg.do_slonczewski = new_state
        else:
            self.llg.do_slonczewski = not self.llg.do_slonczewski

    def set_zhangli(self, J_profile=(1e10, 0, 0), P=0.5, beta=0.01,
                    using_u0=False, with_time_update=None):
        """Activate the Zhang-Li spin-transfer torque in the LLG (Task 22).

        Pass-through to :meth:`finmag.physics.llg.LLG.use_zhangli`. ``J_profile``
        is any value accepted by the vector CG1 space (constant triple, callable,
        Field, Function); with ``using_u0`` false the ``1/(1+beta**2)`` factor is
        applied to ``u0 = P mu_B / e``.
        """
        self.llg.use_zhangli(
            J_profile=J_profile, P=P, beta=beta, using_u0=using_u0,
            with_time_update=with_time_update)


def sim_with(mesh, Ms, m_init, alpha=0.5, unit_length=1,
             integrator_backend="scipy", A=None, K1=None, K1_axis=None,
             H_ext=None, demag_solver="FK", demag_solver_type=None, nx=None,
             ny=None, spacing_x=None, spacing_y=None, demag_solver_params=None,
             D=None, name="unnamed", pbc=None, sim_class=Simulation):
    """Create a :class:`Simulation` and add the requested ported interactions.

    Exchange (``A``), uniaxial anisotropy (``K1`` + ``K1_axis``), Zeeman
    (``H_ext``), DMI (``D``, constant scalar, ``dmi_type='auto'``) and
    Fredkin-Koehler demag (``demag_solver='FK'``, default) are ported. Non-FK
    demag solvers and periodic macro-geometry demag (``nx``/``ny``/
    ``spacing_*``) raise ``NotImplementedError`` by name when requested; pass
    ``demag_solver=None`` to build a demag-free simulation.
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
        sim.add(DMI(D))
    if demag_solver is not None:
        if demag_solver != "FK":
            _deferred(
                "sim_with(demag_solver={!r})".format(demag_solver),
                "non-FK demag solvers (only the 'FK' Fredkin-Koehler solver "
                "is ported)",
            )
        if any(v is not None for v in (nx, ny, spacing_x, spacing_y)):
            _deferred(
                "sim_with(nx/ny/spacing_x/spacing_y)",
                "periodic macro-geometry demag",
            )
        # Import lazily so plain `import finmag` and demag-free simulations
        # never pull the native BEM extension or the demag module graph.
        from finmag.energies import Demag

        sim.add(Demag(solver="FK", solver_type=demag_solver_type,
                      parameters=demag_solver_params))
    log.debug("Successfully created simulation '{}'".format(sim.name))
    return sim
