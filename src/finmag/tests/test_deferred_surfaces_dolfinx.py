"""Task 10 headline deferred-surfaces sweep.

This file is the single, aggregated place a reviewer can look to confirm
"unsupported features fail by name, not through incidental import errors"
(the Task 10 gate checklist). Most of the headline deferred surfaces already
have a dedicated, by-name assertion in their owning per-slice suite; this
sweep intentionally does not duplicate that coverage. Instead it:

1. lists, with a direct reference, every headline surface already pinned
   elsewhere, so the gate has one place that documents where each is tested;
2. adds fresh coverage for the ``integrator_backend="sundials"`` path through
   ``Simulation``/``sim_with`` (only the bare ``llg_integrator(...,
   backend="sundials")`` factory call was pinned before this slice); and
3. documents, without asserting it is acceptable, one known pre-existing gap
   that this slice does not fix (see the last test below).

Already pinned by name elsewhere (not duplicated here):

- non-FK / macro-geometry demag request via ``sim_with(demag_solver=...)`` --
  ``test_simulation_dolfinx.py::test_sim_with_non_fk_demag_is_deferred_by_name``
  and ``::test_sim_with_macro_geometry_demag_is_deferred_by_name`` (the default
  ``"FK"`` path is now ported: ``::test_sim_with_default_demag_builds_fk_demag``)
- DMI via ``sim_with(D=...)`` --
  ``test_simulation_dolfinx.py::test_sim_with_dmi_is_deferred_by_name``
- scheduler (``schedule``/``unschedule``/``clear_schedule``) --
  ``test_simulation_dolfinx.py::test_scheduler_api_is_deferred``
- restart (``save_restart_data``/``restart``) --
  ``test_simulation_dolfinx.py::test_restart_is_deferred``
- STT (``set_stt``/``set_zhangli``) --
  ``test_simulation_dolfinx.py::test_stt_is_deferred``
- ``kernel="sllg"``/``kernel="llg_stt"`` and ``parallel=True`` --
  ``test_simulation_dolfinx.py::test_nonstandard_kernels_are_deferred`` and
  ``::test_parallel_flag_is_deferred``
- multi-rank/native Sundials/STT/thermal ``LLG`` state paths --
  ``test_llg_dolfinx.py::test_deferred_surfaces_raise_by_name`` and
  ``::test_multi_rank_state_paths_raise_serial_guard``
- ``backend="sundials"`` at the ``llg_integrator`` factory --
  ``test_scipy_driver_dolfinx.py::test_llg_integrator_sundials_backend_raises_by_name``
- ``TimeZeeman``/``DiscreteTimeZeeman``/``OscillatingZeeman``/``TimeZeemanPython``
  instantiation -- ``test_energies_dolfinx.py`` (``TimeZeeman(...)`` case)
- ``Field.from_expression`` --
  ``test_field_dolfinx.py::test_legacy_only_features_fail_precisely``
"""

import pytest
from dolfinx import mesh
from mpi4py import MPI

from finmag.drivers.llg_integrator import llg_integrator, SundialsIntegrator
from finmag.sim.sim import Simulation, sim_with


def _box():
    return mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)],
        [2, 2, 2],
        mesh.CellType.tetrahedron,
    )


def _make_sim(**kwargs):
    kwargs.setdefault("unit_length", 1e-9)
    kwargs.setdefault("name", "deferred_sweep_sim")
    return Simulation(_box(), 8.6e5, **kwargs)


# --------------------------------------------------------------------------
# new coverage: integrator_backend="sundials" reached through Simulation and
# sim_with (not just the bare llg_integrator() factory call)
# --------------------------------------------------------------------------

def test_simulation_sundials_backend_raises_by_name_on_first_integrator_use():
    if SundialsIntegrator is not None:
        pytest.skip("native sundials extension is available in this environment")
    sim = _make_sim(integrator_backend="sundials")
    sim.set_m((1.0, 0.0, 0.0))
    # Construction must not eagerly build an integrator (lazy creation is a
    # Task 9 invariant); the by-name failure only appears once one is needed.
    assert not sim.has_integrator()
    with pytest.raises(ImportError, match="sundials"):
        sim.integrator


def test_simulation_create_integrator_sundials_backend_raises_by_name():
    if SundialsIntegrator is not None:
        pytest.skip("native sundials extension is available in this environment")
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    with pytest.raises(ImportError, match="sundials"):
        sim.create_integrator(backend="sundials")


def test_sim_with_sundials_backend_raises_by_name_on_first_integrator_use():
    if SundialsIntegrator is not None:
        pytest.skip("native sundials extension is available in this environment")
    sim = sim_with(
        _box(), Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
        integrator_backend="sundials", demag_solver=None,
    )
    with pytest.raises(ImportError, match="sundials"):
        sim.integrator


def test_bare_llg_integrator_sundials_backend_raises_by_name_reference():
    """Cross-check only: the bare factory case is the pre-existing pin in
    ``test_scipy_driver_dolfinx.py::test_llg_integrator_sundials_backend_raises_by_name``.
    Kept here as one line so the sweep file alone demonstrates every
    ``integrator_backend="sundials"`` entry point without requiring a reader
    to open a second file."""
    if SundialsIntegrator is not None:
        pytest.skip("native sundials extension is available in this environment")
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    with pytest.raises(ImportError, match="sundials"):
        llg_integrator(sim.llg, sim.llg._m_field, backend="sundials")


# --------------------------------------------------------------------------
# known, pre-existing (Task 3) gap: this slice documents it rather than
# silently leaving it unrecorded, but does not fix it here (see the Task 10
# report and dev/dolfinx/porting_map.md "Near-Term Gaps").
# --------------------------------------------------------------------------

def test_demag_is_ported_and_the_others_remain_a_known_gap():
    """Task 11b ports the FK demag surface, so ``finmag.energies.Demag`` now
    constructs a working DOLFINx ``FKDemag`` (it no longer surfaces the legacy
    ``ModuleNotFoundError`` for ``dolfin``). ``Demag2D``/``MacroGeometry`` are
    now curated by-name ``NotImplementedError`` too.

    The still-unported ``requires_legacy_dolfin=True`` optional energy classes
    (``DMI``/``CubicAnisotropy``/``ThinFilmDemag``/``FixedEnergyDW``) remain a
    known Task 3 boundary gap: constructed directly they still surface the raw
    ``ModuleNotFoundError`` for ``dolfin`` (their module still imports legacy
    ``dolfin`` at module scope). This is tracked as a near-term gap for their
    owning later slice, not fixed by the FK demag port."""
    from finmag.energies import Demag
    from finmag.energies.demag.fk_demag import FKDemag

    assert isinstance(Demag(), FKDemag)

    # still-unported optional energies remain the documented gap
    with pytest.raises(ModuleNotFoundError, match="dolfin"):
        from finmag.energies import DMI

        DMI(1e-3)
