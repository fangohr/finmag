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
    box = _box()
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name="my_sim")
    assert sim.mesh is box
    assert sim.unit_length == 1e-9
    assert sim.name == "my_sim"
    assert sim.integrator_backend == "scipy"
    # scalar Ms exposed as a Field averaging to the requested value
    assert np.isclose(float(np.average(sim.Ms.as_array())), 8.6e5)
    # scalar alpha / gamma defaults come straight from the LLG core
    assert np.isclose(sim.alpha, 0.5)
    assert np.isclose(sim.gamma, consts.gamma)
    assert sim.Volume > 0.0


def test_scalar_alpha_and_gamma_roundtrip():
    sim = _make_sim()
    sim.alpha = 0.1
    assert np.isclose(sim.alpha, 0.1)
    sim.gamma = 2.0e5
    assert np.isclose(sim.gamma, 2.0e5)


def test_spatially_varying_alpha_is_rejected():
    sim = _make_sim()
    with pytest.raises(NotImplementedError):
        sim.alpha = np.array([0.1, 0.2, 0.3])


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


def test_sim_with_macro_geometry_demag_is_deferred_by_name():
    with pytest.raises(NotImplementedError, match="macro-geometry|periodic"):
        sim_with(_box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 demag_solver="FK", nx=2)


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


def test_stt_is_deferred():
    sim = _make_sim()
    with pytest.raises(NotImplementedError):
        sim.set_stt(1e12, 0.5, 2e-9, (1.0, 0.0, 0.0))
    with pytest.raises(NotImplementedError):
        sim.set_zhangli()


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


def test_region_and_hysteresis_are_deferred():
    sim = _make_sim()
    with pytest.raises(NotImplementedError):
        sim.mark_regions(lambda pt: 0)
    with pytest.raises(NotImplementedError):
        sim.hysteresis([])
