"""Focused production tests for the direct DOLFINx core ``Simulation`` port.

These cover construction, state accessors, the interaction registry
pass-through, integrator creation/tolerances/time stepping, ``sim_with`` for
the ported interactions, an end-to-end physical-time relaxation using every
configured interaction field, and the explicit by-name deferrals for the
surfaces that are out of scope for this slice.
"""

import sys

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

import finmag.util.consts as consts
from finmag.field import Field
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.sim.sim import Simulation, sim_with


def _box(n=2, length=5.0):
    """A small structured 3D box mesh (mesh coordinates in ``unit_length``)."""
    return mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (length, length, length)],
        [n, n, n],
        mesh.CellType.tetrahedron,
    )


def _make_sim(**kwargs):
    kwargs.setdefault("unit_length", 1e-9)
    kwargs.setdefault("name", "test_sim")
    return Simulation(_box(), 8.6e5, **kwargs)


# --------------------------------------------------------------------------
# import boundary
# --------------------------------------------------------------------------

def test_ported_simulation_does_not_load_legacy_dolfin_or_native():
    assert Simulation.__module__ == "finmag.sim.sim"
    assert sim_with.__module__ == "finmag.sim.sim"
    # Checked in a fresh interpreter: once the opt-in FK demag interaction is
    # used (elsewhere in this suite) ``finmag.native`` stays in this process's
    # ``sys.modules``, so the core import-boundary invariant -- a *demag-free*
    # simulation pulls neither legacy ``dolfin`` nor ``finmag.native`` -- must
    # be asserted in isolation. [Claude Opus 4.8]
    import subprocess
    import sys as _sys

    script = (
        "import sys\n"
        "from finmag.sim.sim import Simulation, sim_with\n"
        "import dolfinx.mesh as dm\n"
        "from mpi4py import MPI\n"
        "box = dm.create_box(MPI.COMM_WORLD, [(0.,0.,0.),(5.,5.,5.)],"
        " [2,2,2], dm.CellType.tetrahedron)\n"
        "sim = sim_with(box, Ms=8.6e5, m_init=(1.,0.,0.), unit_length=1e-9,"
        " demag_solver=None)\n"
        "assert 'dolfin' not in sys.modules\n"
        "assert not any(n.startswith('finmag.native') for n in sys.modules)\n"
    )
    subprocess.run([_sys.executable, "-c", script], check=True)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_construction_core_state():
    """SR1 P1.3: ``Simulation.integrator_backend`` defaults to ``"sundials"``,
    matching the legacy default and the ``llg_integrator`` factory default that
    Task 20 fix round 1 restored. The temporary ``"scipy"`` default sanctioned
    by Phase 1 Task 8/9 is gone: the owner conditionally approved restoring the
    legacy native backend as the public default once the Sundials lifecycle was
    trustworthy, and SR1 P1.1 (backend-neutral ``reset_time``) and P1.2
    (truthful restart provenance) discharged that condition. SciPy is not
    removed -- it stays a fully supported explicit opt-in
    (``integrator_backend="scipy"``). [Claude Opus 4.8]
    """
    box = _box()
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name="my_sim")
    assert sim.mesh is box
    assert sim.unit_length == 1e-9
    assert sim.name == "my_sim"
    assert sim.integrator_backend == "sundials"
    # ``driver`` is initialised from the backend actually requested (P1.2), so
    # the public provenance surface must follow the flipped default too.
    assert sim.driver == "sundials"
    # scalar Ms exposed as a Field averaging to the requested value
    assert np.isclose(float(np.average(sim.Ms.as_array())), 8.6e5)
    # scalar alpha / gamma defaults come straight from the LLG core
    assert np.isclose(sim.alpha, 0.5)
    assert np.isclose(sim.gamma, consts.gamma)
    assert sim.Volume > 0.0


def test_sim_with_default_integrator_backend_is_sundials():
    """SR1 P1.3: the ``sim_with`` convenience factory carries the same public
    default as ``Simulation.__init__``. [Claude Opus 4.8]"""
    import inspect

    assert (inspect.signature(sim_with).parameters["integrator_backend"].default
            == "sundials")
    assert (inspect.signature(Simulation.__init__)
            .parameters["integrator_backend"].default == "sundials")

    sim = sim_with(_box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                   name="sim_with_default_backend", demag_solver=None)
    assert sim.integrator_backend == "sundials"
    assert sim.driver == "sundials"


def test_default_backend_instantiates_native_sundials_integrator():
    """SR1 P1.3: the flipped default must genuinely reach the *native*
    Sundials/CVODE driver, not merely report a string. Asserted through the
    public lazy-creation path plus a real physical-time step. [Claude Opus 4.8]
    """
    from finmag.drivers.sundials_integrator import SundialsIntegrator

    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    assert isinstance(sim.integrator, SundialsIntegrator)
    sim.run_until(1e-12)
    assert sim.t >= 1e-12
    assert sim.driver == "sundials"


def test_explicit_scipy_backend_remains_supported():
    """SR1 P1.3 non-goal guard: SciPy is not removed; the explicit opt-in still
    selects the ported ``ScipyIntegrator`` and steps. [Claude Opus 4.8]"""
    from finmag.drivers.scipy_integrator import ScipyIntegrator

    sim = _make_sim(integrator_backend="scipy")
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    assert isinstance(sim.integrator, ScipyIntegrator)
    sim.run_until(1e-12)
    assert sim.t >= 1e-12
    assert sim.integrator_backend == "scipy"
    assert sim.driver == "scipy"


def test_scalar_alpha_and_gamma_roundtrip():
    sim = _make_sim()
    sim.alpha = 0.1
    assert np.isclose(sim.alpha, 0.1)
    sim.gamma = 2.0e5
    assert np.isclose(sim.gamma, 2.0e5)


def test_spatially_varying_alpha_is_supported():
    """Task 16: spatially varying alpha (callable) is accepted via the sim
    property; see test_variable_params_dolfinx.py for the oracle-pinned RHS."""
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.alpha = lambda x: 0.1 + 0.02 * x[0]
    assert isinstance(sim.alpha, np.ndarray)
    assert np.all(sim.alpha > 0.0)


# --------------------------------------------------------------------------
# magnetisation accessors
# --------------------------------------------------------------------------

def test_set_m_constant_and_accessors():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    m = sim.m.reshape((3, -1))
    assert np.allclose(m[0], 1.0)
    assert np.allclose(m[1], 0.0)
    assert np.allclose(m[2], 0.0)
    # unit norm everywhere
    assert np.allclose(np.linalg.norm(m, axis=0), 1.0)
    assert isinstance(sim.m_field, Field)
    assert np.allclose(sim.m_average, [1.0, 0.0, 0.0], atol=1e-12)


def test_set_m_callable():
    sim = _make_sim()
    sim.set_m(lambda pt: (0.0, 0.0, 1.0))
    assert np.allclose(sim.m_average, [0.0, 0.0, 1.0], atol=1e-12)


def test_t_is_zero_before_integration():
    sim = _make_sim()
    assert sim.t == 0.0
    # reading t must not silently create an integrator
    assert not sim.has_integrator()


def test_dmdt_shape_matches_m():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.llg.effective_field.update(0.0)
    sim.llg.solve(0.0)
    assert sim.dmdt.shape == sim.m.shape


# --------------------------------------------------------------------------
# interaction registry pass-through
# --------------------------------------------------------------------------

def test_interaction_registry():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    ex = Exchange(13.0e-12)
    ze = Zeeman((0.0, 0.0, 1e6))
    sim.add(ex)
    sim.add(ze)

    assert sim.has_interaction("Exchange")
    assert sim.has_interaction("Zeeman")
    assert sorted(sim.interactions()) == ["Exchange", "Zeeman"]
    assert sorted(sim.get_interaction_list()) == ["Exchange", "Zeeman"]
    assert sim.get_interaction("Zeeman") is ze

    # energy accessors
    assert np.isfinite(sim.total_energy())
    assert np.isfinite(sim.compute_energy("Zeeman"))
    assert np.isclose(sim.compute_energy("total"), sim.total_energy())

    sim.remove_interaction("Exchange")
    assert not sim.has_interaction("Exchange")
    assert sim.interactions() == ["Zeeman"]


def test_effective_field_method():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    H = sim.effective_field()
    assert H.shape == sim.m.shape


def test_switch_off_H_ext_default_zeroes_but_keeps_interaction():
    """Bare ``switch_off_H_ext()`` matches legacy: the Zeeman interaction
    stays registered with its field/energy zeroed, it is not removed."""
    sim = _make_sim()
    sim.set_m((0.0, 0.0, 1.0))
    ze = Zeeman((0.0, 0.0, 1e6))
    sim.add(ze)
    assert sim.compute_energy("Zeeman") != 0.0

    sim.switch_off_H_ext()

    assert sim.has_interaction("Zeeman")
    assert sim.get_interaction("Zeeman") is ze
    assert np.count_nonzero(ze.compute_field()) == 0
    assert sim.compute_energy("Zeeman") == 0.0


def test_switch_off_H_ext_remove_interaction_removes_it():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))

    sim.switch_off_H_ext(remove_interaction=True)

    assert not sim.has_interaction("Zeeman")
    with pytest.raises(KeyError):
        sim.get_interaction("Zeeman")


# --------------------------------------------------------------------------
# integrator lifecycle
# --------------------------------------------------------------------------

def test_integrator_created_on_first_use():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    assert not sim.has_integrator()
    integrator = sim.integrator  # property triggers lazy creation
    assert sim.has_integrator()
    assert sim.integrator is integrator


def test_set_tol_updates_tolerances():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.set_tol(reltol=1e-8, abstol=1e-9)
    assert sim.reltol == 1e-8
    assert sim.abstol == 1e-9
    # applies to a live integrator too
    _ = sim.integrator
    sim.set_tol(reltol=1e-7, abstol=1e-7)
    assert sim.integrator.reltol == 1e-7
    assert sim.integrator.abstol == 1e-7


def test_advance_time_moves_clock():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.advance_time(1e-12)
    assert np.isclose(sim.t, 1e-12)


def test_reset_time():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.advance_time(1e-12)
    sim.reset_time(0.0)
    assert sim.t == 0.0


@pytest.mark.parametrize("backend", ["sundials", "scipy"])
def test_reset_time_to_nonzero_keeps_m_and_allows_further_integration(backend):
    """Reset is a clock operation, not a state operation (SR1 P1.1).

    Covers both backends through ``Simulation``. This used to cover only the
    then-default SciPy half; SR1 P1.3 flipped the public default to
    ``"sundials"``, so the backend is now parametrised explicitly rather than
    left implicit, keeping the SciPy half of the backend-neutral ``reset_time``
    contract pinned. The driver-level Sundials half lives in
    ``test_sundials_driver_dolfinx.py``. [Claude Opus 4.8]
    """
    sim = _make_sim(integrator_backend=backend)
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.advance_time(1e-12)
    m_before = sim.m.copy()

    sim.reset_time(5e-12)

    assert sim.t == 5e-12
    assert np.array_equal(sim.m, m_before)

    sim.advance_time(6e-12)
    assert np.isclose(sim.t, 6e-12)
    assert not np.allclose(sim.m, m_before)


def test_reinit_integrator_noop_without_integrator():
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    # must not raise even though there is no integrator yet
    sim.reinit_integrator()
    assert not sim.has_integrator()


# --------------------------------------------------------------------------
# end-to-end physics with all three interaction fields
# --------------------------------------------------------------------------

def test_run_until_relaxes_towards_field():
    box = _box(n=3)
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name="relax_sim")
    sim.alpha = 0.5
    sim.set_m((1.0, 0.0, 0.0))  # start perpendicular to the applied field

    sim.add(Exchange(13.0e-12))
    sim.add(Zeeman((0.0, 0.0, 1e6)))  # strong field along +z
    sim.add(UniaxialAnisotropy(1e5, (0.0, 0.0, 1.0)))  # easy axis along z

    m0 = sim.m_average.copy()
    sim.run_until(1e-11)
    m1 = sim.m_average

    # unit norm preserved at every node
    m_nodal = sim.m.reshape((3, -1))
    assert np.allclose(np.linalg.norm(m_nodal, axis=0), 1.0, atol=1e-5)

    # magnetisation tilts toward the +z field/easy axis
    assert m1[2] > m0[2] + 0.05
    assert m1[2] > 0.05

    # average magnetisation stays physical (bounded by the nodal unit norm,
    # which itself holds only to the integrator's ~1e-5 relaxation tolerance)
    assert np.linalg.norm(m1) <= 1.0 + 1e-4
    assert sim.t >= 1e-11


# --------------------------------------------------------------------------
# sim_with factory (ported interactions)
# --------------------------------------------------------------------------

def test_sim_with_builds_ported_interactions():
    box = _box()
    sim = sim_with(
        box,
        Ms=8.6e5,
        m_init=(1.0, 0.0, 0.0),
        alpha=0.2,
        unit_length=1e-9,
        A=13.0e-12,
        K1=1e5,
        K1_axis=(0.0, 0.0, 1.0),
        H_ext=(0.0, 0.0, 1e6),
        D=5e-3,
        demag_solver=None,
        name="sim_with_test",
    )
    assert np.isclose(sim.alpha, 0.2)
    assert sim.has_interaction("Exchange")
    assert sim.has_interaction("Zeeman")
    assert sim.has_interaction("Anisotropy")
    assert sim.has_interaction("DMI")
    assert np.allclose(sim.m_average, [1.0, 0.0, 0.0], atol=1e-12)


def test_sim_with_default_demag_builds_fk_demag():
    """Task 11b: the default FK demag solver is now ported and wired in, so
    ``sim_with`` (default ``demag_solver='FK'``) adds a working Demag
    interaction instead of raising by name (was
    ``test_sim_with_default_demag_is_deferred_by_name``)."""
    sim = sim_with(_box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9)
    assert sim.has_interaction("Demag")
    # compiled FK demag produces a finite, non-trivial field/energy
    H = sim.get_interaction("Demag").average_field()
    assert np.all(np.isfinite(H))
    assert np.isfinite(sim.total_energy())


def test_sim_with_non_fk_demag_is_deferred_by_name():
    with pytest.raises(NotImplementedError, match="non-FK|GCR|[Tt]reecode"):
        sim_with(_box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 demag_solver="GCR")


# --------------------------------------------------------------------------
# sim_with macro-geometry (periodic tiling) demag -- SR1 P2.2
#
# ``nx``/``ny``/``spacing_x``/``spacing_y`` are forwarded to
# ``MacroGeometry(nx=nx, ny=ny, dx=spacing_x, dy=spacing_y)``, exactly as in
# legacy ``sim.py`` (b5015c5a, lines 1443-1447): no transformation, no
# ``unit_length`` scaling.  ``spacing_*`` is the tile PITCH (centre-to-centre)
# in mesh coordinate units, not a gap.  [Claude Opus 4.8]
# --------------------------------------------------------------------------

_DEMAG_TOL = {"absolute_tolerance": 1e-10, "relative_tolerance": 1e-10,
              "maximum_iterations": int(1e5)}
_DEMAG_PARAMS = {"phi_1": _DEMAG_TOL, "phi_2": _DEMAG_TOL}


def _centred_box(lo, hi, n):
    return mesh.create_box(
        MPI.COMM_WORLD, [np.array(lo, float), np.array(hi, float)],
        list(n), mesh.CellType.tetrahedron)


def _demag_at_origin(sim, Ms):
    """Reduced demag field ``H/Ms`` at the mesh node nearest the origin."""
    H = sim.get_interaction("Demag").compute_field().reshape((3, -1)).T
    coords, _ = sim.m_field.coords_and_values()
    i = int(np.argmin(np.linalg.norm(coords, axis=1)))
    return H[i] / Ms


def _pbc_vs_bar(m_init, component):
    """(tiled 20nm cube, directly meshed 60x20x20 bar) reduced demag fields.

    Reproduces the legacy acceptance test for this feature
    (``b5015c5a:src/finmag/energies/demag/demag_pbc_test.py``): a 20nm cube
    tiled 3x along x must give the same demag field at the centre as a single
    directly meshed 60x20x20 nm bar.  This is a genuine cross-geometry check --
    a periodic image sum against a different, directly meshed body -- so a
    units or pitch-vs-gap regression in the ``spacing_x`` wiring cannot pass it.

    The pitch is 20.001 rather than the legacy default of exactly 20 (touching):
    the exactly-touching case is the coincident-node defect pinned by
    ``test_treecode_pbc_demag_dolfinx.py::
    test_pbc_coincident_tile_spacing_produces_a_non_finite_bem`` and is refused
    by name by ``sim_with``.  [Claude Opus 4.8]
    """
    Ms = 1e6
    cube = _centred_box((-10, -10, -10), (10, 10, 10), (10, 10, 10))
    bar = _centred_box((-30, -10, -10), (30, 10, 10), (30, 10, 10))
    tiled = sim_with(cube, Ms=Ms, m_init=m_init, unit_length=1e-9,
                     nx=3, spacing_x=20.001, demag_solver_params=_DEMAG_PARAMS,
                     name="pbc_tiled_cube")
    ref = sim_with(bar, Ms=Ms, m_init=m_init, unit_length=1e-9,
                   demag_solver_params=_DEMAG_PARAMS, name="pbc_ref_bar")
    return (_demag_at_origin(tiled, Ms)[component],
            _demag_at_origin(ref, Ms)[component])


def test_sim_with_macro_geometry_reproduces_directly_meshed_bar_in_plane():
    h, h_ref = _pbc_vs_bar((1.0, 0.0, 0.0), 0)
    assert abs((h - h_ref) / h_ref) < 0.01, (h, h_ref)


def test_sim_with_macro_geometry_reproduces_directly_meshed_bar_out_of_plane():
    h, h_ref = _pbc_vs_bar((0.0, 0.0, 1.0), 2)
    assert abs((h - h_ref) / h_ref) < 0.02, (h, h_ref)


def test_sim_with_single_tile_macro_geometry_matches_plain_fk_demag():
    """``nx=1, ny=1`` is a one-element image lattice, so it must reproduce the
    plain (non-periodic) FK demag exactly.  Cheap exactness guard; note it is
    insensitive to ``spacing_*`` (Ts = [[0,0,0]] regardless), so it is a
    regression guard, not the physical witness.  [Claude Opus 4.8]"""
    Ms = 8.6e5
    kw = dict(Ms=Ms, m_init=(0.1, 0.2, 1.0), unit_length=1e-9)
    plain = sim_with(_box(), name="mg_plain", **kw)
    tiled = sim_with(_box(), nx=1, ny=1, name="mg_1x1", **kw)
    h_p = plain.get_interaction("Demag").compute_field()
    h_t = tiled.get_interaction("Demag").compute_field()
    assert np.max(np.abs(h_t - h_p)) / np.max(np.abs(h_p)) < 1e-12


def test_sim_with_macro_geometry_matches_hand_assembled_demag():
    """The four arguments are forwarded to ``MacroGeometry`` unchanged."""
    from finmag.energies import Demag
    from finmag.energies.demag import MacroGeometry

    Ms = 8.6e5
    cube = _centred_box((-10, -10, -10), (10, 10, 10), (4, 4, 4))
    via_sim_with = sim_with(cube, Ms=Ms, m_init=(1.0, 0.0, 0.0),
                            unit_length=1e-9, nx=3, spacing_x=20.001,
                            name="mg_sim_with")
    manual = Simulation(cube, Ms, unit_length=1e-9, name="mg_manual")
    manual.set_m((1.0, 0.0, 0.0))
    manual.add(Demag(solver="FK",
                     macrogeometry=MacroGeometry(nx=3, dx=20.001)))
    np.testing.assert_allclose(
        via_sim_with.get_interaction("Demag").compute_field(),
        manual.get_interaction("Demag").compute_field(), rtol=1e-10, atol=0.0)


def test_sim_with_macro_geometry_rejects_even_tile_counts():
    """Historical note: this test was
    ``test_sim_with_macro_geometry_demag_is_deferred_by_name`` and asserted a
    by-name ``NotImplementedError`` while ``nx``/``ny``/``spacing_*`` were
    unported.  SR1 P2.2 wires them through, so ``nx=2`` now reaches
    ``MacroGeometry``'s legacy odd-positive-tile validation and raises
    ``ValueError`` instead (legacy raised a bare ``Exception`` with the same
    message; the port narrows it).  [Claude Opus 4.8]"""
    with pytest.raises(ValueError, match="odd"):
        sim_with(_box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 demag_solver="FK", nx=2)
    with pytest.raises(ValueError, match="odd"):
        sim_with(_box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 demag_solver="FK", ny=-3)


def test_sim_with_touching_macro_geometry_tiles_are_deferred_by_name():
    """Deliberate divergence from the legacy default (SR1 P2.2).

    Legacy ``sim_with(nx=3)`` with no ``spacing_x`` meant "the tiles touch",
    i.e. pitch == mesh extent.  In this port that is the coincident-node case,
    which returns a silently wrong field (~158% error on a cube, NaN-poisoned
    on a flat slab) -- see ``test_treecode_pbc_demag_dolfinx.py::
    test_pbc_coincident_tile_spacing_produces_a_non_finite_bem``.  ``sim_with``
    refuses it by name rather than exposing it.  [Claude Opus 4.8]"""
    box = _box(2, 5.0)
    with pytest.raises(NotImplementedError, match="touching"):
        sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9, nx=3)
    # explicit spacing equal to the mesh extent is the same configuration
    with pytest.raises(NotImplementedError, match="touching"):
        sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 nx=3, spacing_x=5.0)
    with pytest.raises(NotImplementedError, match="touching"):
        sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 ny=3, spacing_y=5.0)
    # a single tile along a given axis never tiles, so it is not affected
    sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
             nx=1, ny=1, name="mg_touching_1x1")
    # ... and a slightly larger pitch is accepted
    sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
             nx=3, spacing_x=5.0 * (1 + 1e-6), name="mg_touching_gap")


def test_sim_with_treecode_demag_is_still_deferred_by_name():
    """The Treecode factory selector is out of scope for SR1 P2.2 and stays
    deferred even now that the macro-geometry arguments are wired."""
    with pytest.raises(NotImplementedError, match="non-FK|[Tt]reecode"):
        sim_with(_box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 demag_solver="Treecode", nx=3, spacing_x=10.0)


def test_sim_with_dmi_builds_ported_interaction():
    """Task 13: DMI (``D``, constant scalar, ``dmi_type='auto'``) is now
    ported, so ``sim_with(D=...)`` adds a working DMI interaction instead of
    raising by name (was ``test_sim_with_dmi_is_deferred_by_name``)."""
    sim = sim_with(
        _box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
        D=1e-3, demag_solver=None,
    )
    assert sim.has_interaction("DMI")
    assert np.isfinite(sim.get_interaction("DMI").compute_energy())


# --------------------------------------------------------------------------
# explicit by-name deferrals
# --------------------------------------------------------------------------

def test_pbc_is_deferred():
    with pytest.raises(NotImplementedError, match="[Pp]eriodic|pbc"):
        Simulation(_box(), 8.6e5, unit_length=1e-9, pbc="2d")


def test_parallel_flag_is_deferred():
    with pytest.raises(NotImplementedError, match="parallel|multi-rank"):
        Simulation(_box(), 8.6e5, unit_length=1e-9, parallel=True)


@pytest.mark.parametrize("kernel", ["sllg", "llg_stt"])
def test_nonstandard_kernels_are_deferred(kernel):
    with pytest.raises(NotImplementedError, match=kernel):
        Simulation(_box(), 8.6e5, unit_length=1e-9, kernel=kernel)


def test_scheduler_api_is_available():
    """Task 12: the scheduler is ported. ``schedule`` returns a scheduled item
    and an unknown shortcut string fails by name with ``KeyError`` (was
    ``test_scheduler_api_is_deferred``)."""
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    item = sim.schedule("save_ndt", every=1e-12)
    assert item is not None
    with pytest.raises(KeyError, match="unknown"):
        sim.schedule("no_such_action", every=1e-12)


def test_stt_is_ported_not_deferred():
    """Task 22: ``set_stt``/``set_zhangli``/``toggle_stt`` are ported
    pass-throughs to the LLG STT surfaces (was ``test_stt_is_deferred``). Full
    behavioural + oracle coverage lives in ``test_stt_dolfinx.py``; this only
    asserts the surfaces no longer raise ``NotImplementedError`` by name."""
    sim = _make_sim()
    sim.set_m((0.6, 0.0, 0.8))
    sim.set_stt(1e12, 0.5, 2e-9, (0.0, 0.0, 1.0))
    assert sim.llg.do_slonczewski is True
    sim.toggle_stt()
    assert sim.llg.do_slonczewski is False
    sim.set_zhangli()
    assert sim.llg.do_zhangli is True


def test_restart_and_output_are_available():
    """Task 12: restart persistence and NDT/VTK output are ported (was
    ``test_restart_is_deferred``/``test_output_helpers_are_deferred``). Full
    behavioural coverage lives in ``test_restart_output_dolfinx.py``; this only
    asserts the surfaces no longer raise ``NotImplementedError`` by name."""
    import os
    import tempfile

    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    with tempfile.TemporaryDirectory() as tmp:
        restart_file = os.path.join(tmp, "state.npz")
        sim.ndtfilename = os.path.join(tmp, "state.ndt")
        sim.save_restart_data(filename=restart_file)
        sim.restart(filename=restart_file)
        sim.save_ndt()
        sim.save_vtk(filename=os.path.join(tmp, "m.pvd"))
        sim.m_field.close_pvd()
    assert sim.t == 0.0


def test_regions_are_ported_not_deferred():
    """Task 16: mark_regions + per-region energy/magnetisation accounting are
    ported (see test_variable_params_dolfinx.py). Region-restricted submesh
    field output remains deferred by name."""
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    ids = sim.mark_regions(lambda pt: 1 if pt[0] < 2.5 else 2)
    assert set(ids) == {1, 2}
    with pytest.raises(NotImplementedError, match="save_m_in_region"):
        sim.save_m_in_region(1)


def test_hysteresis_is_ported_not_deferred():
    """Task 15: relax/hysteresis/hysteresis_loop are ported; see the focused
    ``test_hysteresis_dolfinx.py`` suite for the full behavioral contract.
    ``hysteresis([])`` returns ``None`` immediately (matching legacy) rather
    than raising."""
    sim = _make_sim()
    assert sim.hysteresis([]) is None
