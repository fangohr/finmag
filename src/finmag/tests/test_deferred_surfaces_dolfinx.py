"""Task 10 headline deferred-surfaces sweep.

NO MASTER ANCESTOR (genuinely new under DOLFINx): there is no legacy
"deferred surfaces" concern on master -- master has ``dolfin`` and never
needed to defer anything by name. This aggregated sweep exists solely to
police the DOLFINx port's own deferral contract (unsupported features must
fail with a curated, feature-naming ``NotImplementedError``, never an
incidental raw import error), so it is DOLFINx-only by construction.

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
- DMI via ``sim_with(D=...)`` is now PORTED (Task 13) and covered by
  ``test_simulation_dolfinx.py::test_sim_with_dmi_builds_ported_interaction``
  and ``test_dmi_dolfinx.py``; it is no longer deferred.
- Cubic anisotropy is now PORTED (Task 14) and covered by
  ``test_cubic_anisotropy_dolfinx.py``; it is no longer deferred. (Legacy
  ``sim_with`` never had cubic-anisotropy parameters, so there is no
  ``sim_with`` wiring to test; ``Simulation.add(CubicAnisotropy(...))``
  coverage is the full integration surface.)
- scheduler, restart and NDT/VTK output are now PORTED (Task 12) and covered by
  ``test_restart_output_dolfinx.py`` and
  ``test_simulation_dolfinx.py::test_scheduler_api_is_available`` /
  ``::test_restart_and_output_are_available``; they are no longer deferred.
- STT (``set_stt``/``set_zhangli``/``toggle_stt``) are now PORTED (Task 22) as
  NumPy transcriptions of the native Slonczewski/Zhang-Li kernels; they are no
  longer deferred and are covered by ``test_stt_dolfinx.py`` and
  ``test_simulation_dolfinx.py::test_stt_is_ported_not_deferred`` /
  ``test_llg_dolfinx.py::test_stt_surfaces_are_ported_not_deferred``. The
  separate nonlocal-STT ``kernel="llg_stt"`` class (``LLG_STT``) stays deferred
  by name (distinct capability, Task 29-registered).
- ``kernel="sllg"``/``kernel="llg_stt"`` and ``parallel=True`` --
  ``test_simulation_dolfinx.py::test_nonstandard_kernels_are_deferred`` and
  ``::test_parallel_flag_is_deferred``
- multi-rank/thermal ``LLG`` state paths --
  ``test_llg_dolfinx.py::test_multi_rank_state_paths_raise_serial_guard``
- ``backend="sundials"`` at the ``llg_integrator`` factory --
  ``test_scipy_driver_dolfinx.py::test_llg_integrator_sundials_backend_raises_by_name``
- ``TimeZeeman``/``DiscreteTimeZeeman``/``OscillatingZeeman``/
  ``TimeZeemanPython``/``DipolarField`` are now PORTED (Task 15) and covered
  by ``test_timezeeman_dolfinx.py``; they are no longer deferred.
  ``test_energies_dolfinx.py``'s ``TimeZeeman((1.0, 0.0, 0.0))`` case now
  pins the ported constant-array-without-``t_off`` ``ValueError`` instead of
  a by-name deferral. ``sim.relax``/``hysteresis``/``hysteresis_loop`` are
  also now PORTED (Task 15) and covered by ``test_hysteresis_dolfinx.py``.
- ``Field.from_expression`` --
  ``test_field_dolfinx.py::test_legacy_only_features_fail_precisely``
- ``ThinFilmDemag`` is now PORTED (Task 19) and covered by
  ``test_thin_film_demag_dolfinx.py``; it is no longer deferred.
  ``FixedEnergyDW`` is now a curated by-name ``NotImplementedError``
  deferral (Task 19, Task 29 review item) -- covered by
  ``test_dw_fixed_energy_dolfinx.py``; it no longer surfaces the raw
  ``ModuleNotFoundError`` for ``dolfin`` either.
- ``Field.probe``/``Field.__call__`` (point evaluation), ``Field.get_spherical``
  and ``Simulation.skyrmion_number``/``skyrmion_number_density_function`` are
  now PORTED (Task 26a) and covered by ``test_io_utils_dolfinx.py``. Point
  probing/``get_spherical`` used to be curated by-name ``NotImplementedError``
  deferrals on ``Field`` (pinned by ``test_field_dolfinx.py::
  test_legacy_only_features_fail_precisely``, now removed from that test);
  ``skyrmion_number``/``skyrmion_number_density_function`` had instead been
  dropped outright in Task 9 (a plain ``AttributeError``, not a by-name
  deferral -- see ``dev/dolfinx/porting_map.md``). None of the three were
  ever asserted as deferred *in this sweep file*, so nothing here is removed;
  this bullet exists only to keep this file's aggregated reviewer reference
  current. ``Simulation.probe_field``/``probe_field_along_line`` remain
  deferred by name (they depend on region-restricted
  ``get_field_as_dolfin_function``, out of scope for Task 26a -- see its
  ``save_m_in_region``/``get_submesh`` siblings above).
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
# historical Task 3 boundary gap, now fully resolved as of Task 19: every
# finmag.energies public name is either ported directly or a curated by-name
# NotImplementedError deferral -- none surfaces the raw legacy
# ModuleNotFoundError('dolfin') any more (see dev/dolfinx/porting_map.md).
# --------------------------------------------------------------------------

def test_demag_dmi_cubic_anisotropy_and_optional_energies_are_ported_or_curated_deferrals():
    """State at HEAD (kept accurate by the whole-branch review -- this file is
    the designated aggregated reviewer reference; see item 2 of the Tier 1
    review). Task 11b ports the FK demag surface, so ``finmag.energies.Demag``
    now constructs a working DOLFINx ``FKDemag`` (it no longer surfaces the
    legacy ``ModuleNotFoundError`` for ``dolfin``). Task 23 ports the
    treecode-accelerated solver (``Demag(solver='Treecode')``) and the periodic
    ``MacroGeometry`` demag (``Demag(macrogeometry=...)``) on top of the native
    ``treecode_bem`` Cython kernels (see ``test_treecode_pbc_demag_dolfinx.py``);
    ``MacroGeometry`` now constructs a working tiling object. ``Demag2D`` stays a
    curated by-name ``NotImplementedError`` (heavy MeshEditor/Expression
    coupling, no treecode dependency; Task 29 review item), and the ``GCR``
    solver stays deferred by name.

    Task 13 ports ``DMI`` directly (constant scalar ``D``, ``dmi_type``
    dispatch across ``'auto'``/``'1d'``/``'2d'``/``'3d'``/``'interfacial'``);
    constructing it directly now works instead of surfacing the legacy
    ``ModuleNotFoundError`` for ``dolfin`` (see ``test_dmi_dolfinx.py`` for
    the focused DMI suite). Task 16 update: spatially varying ``D``
    (callable/``Field``/``dolfinx.fem.Function``, placed in DG0) is now
    SUPPORTED -- it is no longer a by-name deferral. Only the undocumented
    ``dmi_type='D2D'`` variant and legacy string Expressions remain curated
    by-name ``NotImplementedError``.

    Task 14 ports ``CubicAnisotropy`` directly (constant scalar
    ``K1``/``K2``/``K3`` and constant ``u1``/``u2`` axes, with
    ``u3 = u1 x u2``); constructing it directly now works too (see
    ``test_cubic_anisotropy_dolfinx.py`` for the focused suite). Task 14 fix
    round 1 (60ac165b) ports the legacy-default ``assemble=False``
    native/direct field path as a NumPy transcription of the closed-form
    analytic field, so ``compute_field()`` under the default now works too --
    it is no longer a by-name deferral (energy was always box-assembled and
    always worked, matching the legacy class exactly). Task 16 update:
    spatially varying ``K1``/``K2``/``K3`` (CG1-placed) are now SUPPORTED,
    including under the ``assemble=False`` native analytic path (see
    ``test_variable_params_dolfinx.py`` for the K2-native-typo divergence pin
    this makes LIVE). Only spatially varying ``u1``/``u2`` axes remain
    curated by-name ``NotImplementedError``.

    Task 19 update: the previously-still-unported ``requires_legacy_dolfin=
    True`` optional energy classes are now resolved -- no boundary gap
    remains here. ``ThinFilmDemag`` (the last legacy-tested energy that
    still imported raw ``dolfin`` at module scope) is now PORTED directly
    (see ``test_thin_film_demag_dolfinx.py``); constructing it directly now
    works instead of surfacing the legacy ``ModuleNotFoundError``.
    ``FixedEnergyDW`` -- untested even on legacy master, its own legacy todo
    notes calling it broken -- is converted to a curated by-name
    ``NotImplementedError`` deferral (Task 29 review item; see
    ``test_dw_fixed_energy_dolfinx.py`` and the module docstring in
    ``finmag/energies/dw_fixed_energy.py``) rather than a raw import error.
    No ``finmag.energies`` public name is ``requires_legacy_dolfin=True``
    any more (``test_import_boundary.py::
    test_no_unported_optional_energies_remain`` pins this directly)."""
    from finmag.energies import CubicAnisotropy, DMI, Demag, FixedEnergyDW, ThinFilmDemag
    from finmag.energies.demag.fk_demag import FKDemag

    assert isinstance(Demag(), FKDemag)
    assert DMI(1e-3).name == "DMI"
    assert CubicAnisotropy((1, 0, 0), (0, 1, 0), 1.0).name == "CubicAnisotropy"

    # Task 19: ThinFilmDemag is ported directly; no more ModuleNotFoundError.
    assert ThinFilmDemag().name == "ThinFilmDemag"

    # Task 19: FixedEnergyDW is a curated by-name NotImplementedError
    # deferral, not a raw ModuleNotFoundError (Task 29 review item).
    with pytest.raises(NotImplementedError, match="FixedEnergyDW"):
        FixedEnergyDW()
