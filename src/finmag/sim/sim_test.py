"""DOLFINx core ``Simulation`` tests -- minimal-diff transcription + new suite.

This file now lives at its master path
``src/finmag/sim/sim_test.py`` (formerly
``src/finmag/tests/test_simulation_dolfinx.py``), so
``git diff b5015c5a..HEAD -- src/finmag/sim/sim_test.py``
shows the port diff directly. This is a PARTIAL port: the 34 master
``sim_test.py`` functions with no named covering port function anywhere are
carried verbatim under the ``NOT PORTED`` banner at the bottom of the file
(marked ``@pytest.mark.not_ported`` and, per-test, a strict
``@pytest.mark.xfail`` naming the register row; the focused gate runs them
unfiltered and reports them as xfailed, the non-gating inventory lane
reports them as failures). The three sibling master
files whose dropped coverage this port also restores
(``tests/test_sim_ode.py``, ``drivers/tests/test_relaxation.py``,
``drivers/tests/test_relax_two_times.py``) were each single-function files
fully accounted for by a named port function here (``test_sim_ode``,
``test_easy_relaxation``, ``test_relax_two_times``, all above the
``NOT PORTED`` banner) and were REMOVED by the Task 4 review follow-up.

This file has two clearly separated parts (SR1 P5.2 restructure for
diffability + dropped-coverage restoration):

1. A MINIMAL-DIFF transcription of the core master ``Simulation`` unit tests
   from ``src/finmag/sim/sim_test.py`` (git ``b5015c5a``, the ``TestSimulation``
   class methods + module-level ``test_sim_with``), plus three dropped-coverage
   regression tests restored from sibling master files
   (``test_sim_ode`` <- ``src/finmag/tests/test_sim_ode.py``,
   ``test_easy_relaxation`` <- ``src/finmag/drivers/tests/test_relaxation.py``,
   ``test_relax_two_times`` <- ``src/finmag/drivers/tests/test_relax_two_times.py``).
   Master function NAMES, ORDER, assertion STRUCTURE and TOLERANCES are kept
   verbatim; the only differences are (a) dolfin->dolfinx API changes (each
   annotated inline), (b) py2->py3 (``print``/``xrange``/``np.NaN``), and
   (c) comments. Measured DOLFINx values are recorded next to each tolerance.
   The ``MASTER LEDGER`` comment below records the disposition
   (transcribed / covered-elsewhere / genuine-gap) of EVERY master function so
   the 1:1 mapping stays diffable even where a surface is deferred or covered
   by a dedicated dolfinx file.

2. The NEW-under-DOLFINx suite (construction/backend defaults, integrator
   lifecycle, callable pin masks, by-name deferrals, macro-geometry PBC, ...),
   which has no master ancestor. It lives unchanged below the
   ``NEW under DOLFINx`` banner.

[Claude Opus 4.8]
"""

import os
import sys
from math import cos, sin, pi, sqrt

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

import finmag.util.consts as consts
from finmag.field import Field
from finmag.energies import Demag, Exchange, UniaxialAnisotropy, Zeeman
from finmag.example import barmini
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


# ==========================================================================
# PART 1 -- MINIMAL-DIFF transcription of master sim_test.py (git b5015c5a)
# and three restored dropped-coverage tests from sibling master files.
# dolfin->dolfinx changes are annotated inline; tolerances are master's, each
# with the measured DOLFINx value recorded beside it.
# --------------------------------------------------------------------------
#
# MASTER LEDGER -- disposition of every master ``sim_test.py`` function:
#
#   TRANSCRIBED below (core Simulation surface, green under DOLFINx):
#     test_get_interaction, test_compute_energy, test_remove_interaction1,
#     test_remove_interaction2, test_switch_off_H_ext, test_set_H_ext,
#     test_set_m, test_setting_m_also_sets_the_field,
#     test_run_until_0_does_not_change_m,
#     test_can_call_save_restart_data_on_a_fresh_simulation_object,
#     test_probe_constant_m_at_individual_points,
#     test_probe_nonconstant_m_at_individual_points,
#     test_probe_m_on_regular_grid, test_sim_with  (module-level)
#
#   COVERED-ELSEWHERE (a dedicated dolfinx file already ports the surface with
#   dolfinx-native assertions; re-transcribing the heavy I/O verbatim would
#   duplicate it). Every row below names the COVERING FUNCTION, not just the
#   covering file -- each mapping was re-verified against the b5015c5a blob and
#   the named port function's body (review finding I1, 2026-07-27):
#
#     test_schedule (callable + args, ``every=``)
#         -> tests/test_restart_output.py::test_schedule_callable_and_clear
#            (schedule(callable, every=) fires at the expected instants)
#            + ::test_unschedule_removes_item
#            + ::test_schedule_unknown_shortcut_raises_by_name.
#            NARROWING (disclosed): master also passed positional/keyword
#            arguments THROUGH to the scheduled callable ('tag1',
#            optional='foo'); the port's callables take only ``sim``, so the
#            argument-forwarding path itself is not re-asserted anywhere.
#     test_save_ndt (schedule('save_ndt', every=), 6 rows via Tablereader)
#         -> tests/test_restart_output.py::test_schedule_save_ndt_every
#            (same shortcut, same 6-row count assertion)
#            + ::test_sim_save_averages_appends_rows (save_ndt/save_averages
#            method form) + ::test_ndt_format_and_roundtrip (column contract).
#            NARROWING (disclosed, D-row): master additionally asserted the
#            automatic ``E_Exchange``/``H_Exchange_*``/``E_Demag``/``H_Demag_*``
#            columns; the ported Simulation.add() does not register them (the
#            same gap pinned strict-xfail in util/plot_helpers_test.py::
#            test_plot_ndt_columns_and_plot_dynamics), so that half is NOT
#            re-asserted here.
#     test_save_restart_data (schedule('save_restart_data', at_end=True),
#     canonical filename, sim_helpers.load_restart_data)
#         -> tests/test_restart_output.py::
#            test_load_restart_data_by_simulation_uses_canonical_name
#            + ::test_restart_roundtrip_same_simulation.
#     test_restart (save_restart_data(fname); restart(fname, t0=...))
#         -> tests/test_restart_output.py::test_restart_roundtrip_same_simulation
#            + ::test_restart_t0_override
#            + ::test_restart_cross_instance_same_mesh_recipe.
#     test_reset_time
#         -> LEDGER CORRECTION (I1): this is NOT covered by
#            test_restart_output.py at all. It is covered IN THIS FILE, below
#            the NEW banner, by ``test_reset_time`` and
#            ``test_reset_time_to_nonzero_keeps_m_and_allows_further_integration``
#            (the latter parametrized over both integrator backends).
#     test_save_vtk (schedule('save_vtk', overwrite=...))
#         -> tests/test_restart_output.py::test_save_vtk_writes_pvd.
#            NARROWING (disclosed): master's ``overwrite=False`` IOError branch
#            on a pre-existing .pvd is not re-asserted.
#     test_sim_schedule_clear (clear_schedule stops further saves)
#         -> tests/test_restart_output.py::test_schedule_callable_and_clear
#            (asserts no further callbacks fire after clear_schedule()).
#     test_set_stt
#         -> tests/test_stt.py::test_simulation_set_stt_activates_slonczewski.
#     test_get_field_as_dolfin_function
#         -> the ``region=None`` path is exercised through
#            ``Simulation.probe_field`` (sim.py calls
#            get_field_as_dolfin_function internally) by the probe_field block
#            below the NEW banner: test_probe_field_single_point_returns_
#            magnetisation, test_probe_field_array_of_points_returns_per_point_
#            values, test_probe_field_along_line_matches_analytic_profile,
#            test_probe_field_of_exchange_is_zero_for_uniform_magnetisation,
#            test_probe_field_coordinates_are_mesh_units_not_metres. The
#            ``region=`` path stays deferred and is pinned BY NAME by
#            tests/test_variable_params.py::
#            test_region_restricted_field_output_still_deferred_by_name.
#     test_probe_demag_field (probe "Demag" at every vertex vs
#     get_interaction("Demag").compute_field())
#         -> the same probe_field block above for the probing mechanics
#            (test_probe_field_array_of_points_returns_per_point_values proves
#            the per-vertex stack; test_probe_field_of_exchange_is_zero_for_
#            uniform_magnetisation proves it on a computed effective-field
#            interaction rather than on 'm'), plus
#            energies/demag/fk_demag_test.py::
#            test_demag_field_for_uniformly_magnetised_sphere for the demag
#            field values themselves.
#
#   NOT COVERED after all -- ledger correction from review finding I1
#   (2026-07-27). These three were previously listed as
#   "-> test_restart_output.py", but that file's only save-related function is
#   ``test_save_field_to_vtk_xdmf``, which exercises
#   ``Simulation.save_field_to_vtk`` (XDMF/VTK output) -- a DIFFERENT method
#   from the ``.npy`` ``save_field``/``save_m`` surface master exercises here.
#   A tree-wide grep found NO named port function covering the ``.npy`` path,
#   even though ``Simulation.save_field``/``save_m`` and the ``'save_field'``
#   scheduler shortcut are all implemented in the port (sim/sim_savers.py;
#   only ``region=`` is deferred). They are therefore CARRIED VERBATIM under
#   the ``NOT PORTED`` banner rather than left claimed-but-uncovered:
#     test_save_field           -- .npy save, incremental=, overwrite=, 'Demag'
#     test_save_m               -- the save_m convenience shortcut
#     test_save_field_scheduled -- schedule('save_field', 'm', every=)
#
#   GENUINE-GAP (surface not provided by the ported Simulation; reported for an
#   owner decision, NOT fabricated). CANONICAL-PATHS MOVE 2026-07-27: all six
#   are now CARRIED VERBATIM under the ``NOT PORTED`` banner at the bottom of
#   this file and marked ``@pytest.mark.not_ported``, so nothing vanished when
#   the port took master's path:
#     test_sim_sllg, test_sim_sllg_time  -- SLLG stochastic kernel is deferred
#         by name (kernel='sllg' raises NotImplementedError).
#     test_pbc2d_m_init                  -- periodic boundaries deferred (pbc
#         raises NotImplementedError); master itself skipif(dolfin<1.2.0).
#     test_mark_regions                  -- region-restricted field->vtk export
#         deferred; master itself xfail on dolfin>=1.5.
#     test_length_scales                 -- Simulation.length_scales() not ported.
#     test_clean_up                      -- GAP CLOSED (CI T3, 2026-07-28): the
#         teardown surface (``shutdown``/``instances_delete_all_others``/
#         ``instances_list_all``/``instances_delete_all``/
#         ``instances_alive_count``/``close_logfile``) IS now ported, so this
#         test is PROMOTED: it runs LIVE (both markers removed). It stays at the
#         bottom of the file -- master's own position -- because it shuts down
#         every other live ``Simulation`` instance and so must run last.
#
# All NormalModeSimulation / eigenmode / plotting / X-display / gmsh / csg
# module-level functions in sim_test.py belong to the (separate)
# NormalModeSimulation port, not this core Simulation port, and are out of
# scope here. CANONICAL-PATHS MOVE 2026-07-27: "out of scope here" is NOT
# "covered", so all 26 of them (every master module-level function other than
# the transcribed ``test_sim_with``) are ALSO carried verbatim under the
# ``NOT PORTED`` banner at the bottom of this file. 34 master functions are
# carried in total: 8 ``TestSimulation`` methods (the 5 GENUINE-GAP ones above
# plus the 3 NOT-COVERED save_field/save_m ones added by review finding I1) +
# 26 module-level ones.
# ==========================================================================


def _boxmesh(p0, p1, nx, ny, nz):
    # dolfin df.BoxMesh(df.Point(*p0), df.Point(*p1), nx, ny, nz)
    #   -> dolfinx mesh.create_box(...) (tetrahedra, MPI.COMM_WORLD).
    return mesh.create_box(
        MPI.COMM_WORLD, [np.asarray(p0, float), np.asarray(p1, float)],
        [nx, ny, nz], mesh.CellType.tetrahedron)


def _fnormalise(arr):
    # Byte-identical to finmag.util.helpers.fnormalise (default branch);
    # reimplemented locally because helpers imports dolfin at module scope and
    # cannot load under dolfinx (Wave-1 accepted workaround).
    a = arr.astype(np.float64)
    a = a.reshape((3, -1))
    a_norm = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    a = a / a_norm
    a.shape = (-1,)
    return a


def num_interactions(sim):
    """Helper: number of interactions present in the Simulation."""
    return len(sim.interactions())


def test_get_interaction():
    # master shared fixture mesh: df.BoxMesh(Point(0,0,0), Point(1,1,1), 5,5,5)
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 5, 5, 5)
    sim = sim_with(mesh_, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                   unit_length=1e-9, A=13.0e-12, demag_solver='FK')

    # These should just work
    sim.get_interaction('Exchange')
    sim.get_interaction('Demag')

    with pytest.raises(KeyError):
        sim.get_interaction('foobar')

    exch = Exchange(A=13e-12, name='foobar')
    sim.add(exch)
    assert exch == sim.get_interaction('foobar')


def test_compute_energy():
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 5, 5, 5)
    sim = sim_with(mesh_, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                   unit_length=1e-9, A=13.0e-12, demag_solver='FK')

    # These should just work
    sim.compute_energy('Exchange')
    sim.compute_energy('Demag')
    sim.compute_energy('Total')
    sim.compute_energy('total')

    # A non-existing interaction should throw an error
    with pytest.raises(KeyError):
        sim.compute_energy('foobar')

    new_exch = Exchange(A=13e-12, name='foo')
    sim.add(new_exch)
    assert new_exch.compute_energy() == sim.compute_energy('foo')


def test_remove_interaction1():
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 1, 1, 1)
    sim = Simulation(mesh_, Ms=1, unit_length=1e-9)
    sim.add(Zeeman((0, 0, 1)))
    sim.add(Exchange(13e-12))
    assert num_interactions(sim) == 2

    sim.remove_interaction("Exchange")
    assert num_interactions(sim) == 1

    sim.remove_interaction("Zeeman")
    assert num_interactions(sim) == 0

    # No Zeeman interaction present any more
    with pytest.raises(KeyError):
        sim.remove_interaction("Zeeman")


def test_remove_interaction2():
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 1, 1, 1)
    sim = Simulation(mesh_, Ms=1, unit_length=1e-9)

    # Two different Zeeman interactions present
    sim.add(Zeeman((0, 0, 1)))
    sim.add(Zeeman((0, 0, 2), name="Zeeman2"))
    sim.remove_interaction("Zeeman")
    sim.remove_interaction("Zeeman2")

    # master asserted a re-add here raises AssertionError (legacy EffectiveField
    # refused to re-register a name that had been removed).
    # *** BEHAVIOURAL CHANGE under DOLFINx: the ported EffectiveField RELAXES
    # this restriction -- re-adding a previously-removed interaction now
    # SUCCEEDS. *** We assert the actual ported behaviour instead of master's
    # ``pytest.raises(AssertionError)``. [Claude Opus 4.8]
    sim.add(Zeeman((0, 0, 1)))
    assert num_interactions(sim) == 1
    assert sim.has_interaction("Zeeman")


def test_switch_off_H_ext():
    """Simply test that we can call sim.switch_off_H_ext()."""
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 1, 1, 1)
    sim = Simulation(mesh_, Ms=1, unit_length=1e-9)
    sim.add(Zeeman((1, 2, 3)))

    sim.switch_off_H_ext(remove_interaction=False)
    H = sim.get_interaction("Zeeman").compute_field()
    assert np.allclose(H, np.zeros_like(H))

    sim.switch_off_H_ext(remove_interaction=True)
    assert num_interactions(sim) == 0


def test_set_H_ext():
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 1, 1, 1)
    sim = Simulation(mesh_, Ms=1, unit_length=1e-9)
    sim.add(Zeeman((1, 2, 3)))

    # master also read get_field_as_dolfin_function('Zeeman').vector().array()
    # here (a dead line, immediately overwritten): dolfinx Functions expose
    # .x.array (blocked ordering), not dolfin's .vector().array(), so it is
    # dropped. probe_field coordinates are MESH units in the port; master
    # passed 0.5e-9 on a [0,1] mesh (effectively the origin corner) -- for a
    # uniform Zeeman field every interior point returns the same value, so we
    # probe the mesh centre.
    H = sim.probe_field('Zeeman', [0.5, 0.5, 0.5])
    assert np.allclose(H, [1, 2, 3])

    sim.set_H_ext([-4, -5, -6])
    H = sim.probe_field('Zeeman', [0.5, 0.5, 0.5])
    assert np.allclose(H, [-4, -5, -6])

    # Set H_ext in a simulation that doesn't have a Zeeman interaction yet
    sim = Simulation(mesh_, Ms=1, unit_length=1e-9)
    sim.set_H_ext((1, 2, 3))  # this should not raise an error!
    H = sim.probe_field('Zeeman', [0.5, 0.5, 0.5])
    assert np.allclose(H, [1, 2, 3])


@pytest.mark.xfail(strict=True, reason=(
    "GENUINE GAP: the ported Simulation.set_m()/LLG.set_m() does NOT reproduce "
    "the legacy NaN-guard -- a NaN-valued m_init is accepted silently (sim.m "
    "ends up containing NaN) instead of raising ValueError. Kept as an "
    "xfail(strict) so the missing validation stays visible and the test flips "
    "to XPASS the moment the guard is restored. See report. [Claude Opus 4.8]"))
def test_set_m():
    """Test to ensure m is not set with illegal values (such as NaNs)."""
    def m_init_nan(pos):
        return [np.nan, 1, 1]  # py2 np.NaN -> np.nan

    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 1, 1, 1)
    sim = Simulation(mesh_, Ms=1e5, unit_length=1e-9)
    with pytest.raises(ValueError):
        sim.set_m(m_init_nan)


def test_setting_m_also_sets_the_field():
    """Setting 'sim.m' also sets the value of the underlying 'sim.m_field'."""
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 5, 5, 5)
    sim = sim_with(mesh_, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                   unit_length=1e-9, A=13.0e-12, demag_solver='FK')

    m_random = _fnormalise(np.random.random_sample(sim.m.shape))
    sim.m = m_random

    assert np.allclose(sim.m, m_random)
    # master: sim.m_field.f.vector().array() (dolfin blocked vector). dolfinx
    # exposes the same nodal data in the component-blocked ``xxx`` ordering that
    # ``sim.m`` uses via Field.get_ordered_numpy_array_xxx().
    assert np.allclose(sim.m_field.get_ordered_numpy_array_xxx(), m_random)


def test_run_until_0_does_not_change_m():
    """Calling "sim.run_until(0)" does not affect the value of m."""
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 5, 5, 5)
    sim = sim_with(mesh_, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                   unit_length=1e-9, A=13.0e-12, demag_solver='FK')

    m_random = np.random.random_sample(sim.m.shape)
    sim.set_m(m_random, normalise=False)

    assert (sim.m == m_random).all()

    # running until a non-zero time does change m.
    sim.run_until(1e-14)
    assert not np.allclose(sim.m, m_random)


def test_can_call_save_restart_data_on_a_fresh_simulation_object(tmpdir):
    """Regression: save_restart_data() on a newly created simulation object."""
    os.chdir(str(tmpdir))

    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 5, 5, 5)
    sim = sim_with(mesh_, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                   unit_length=1e-9, A=13.0e-12, demag_solver='FK')
    sim.save_restart_data('my_restart_data.npz')


def test_probe_constant_m_at_individual_points():
    mesh_ = _boxmesh((-2, -2, -2), (2, 2, 2), 5, 5, 5)
    m_init = np.array([0.2, 0.7, -0.4])
    # normalize the vector for later comparison
    m_init /= np.linalg.norm(m_init)
    # master passed the (3,) ndarray directly as a constant m_init; the port's
    # set_m treats an ndarray as a FULL field array (expects 3*N entries), so a
    # constant 3-vector must be a tuple/list. Pass tuple(m_init); the comparison
    # array below is unchanged.
    sim = sim_with(
        mesh_, Ms=8.6e5, m_init=tuple(m_init), unit_length=1e-9,
        demag_solver=None)

    probing_pts = [
        [0, 0, 0],
        [0.0, 0.0, 0.0],
        [1, 1, -0.5],
        [-1.3, 0.02, 0.3]]

    m_probed_vals = [sim.probe_field("m", pt) for pt in probing_pts]
    for v in m_probed_vals:
        assert np.allclose(v, m_init)

    # Probe outside the mesh -> the resulting vector is masked.
    m_probed_outside = sim.probe_field("m", [5, -6, 1])
    assert (np.ma.getmask(m_probed_outside) == True).all()


def test_probe_nonconstant_m_at_individual_points():
    TOL = 1e-5  # master 1e-5; measured max deviation ~2.7e-7 (passes verbatim)

    unit_length = 1e-9
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 1000, 2, 2)
    # master m_init = df.Expression(("cos(x[0]*pi)","sin(x[0]*pi)","0.0")).
    # dolfinx has no df.Expression; the equivalent Python callable receives
    # coordinates in mesh units (matching the port's set_m contract).
    def m_init(pt):
        return (cos(pt[0] * pi), sin(pt[0] * pi), 0.0)
    sim = sim_with(
        mesh_, Ms=8.6e5, m_init=m_init, unit_length=unit_length,
        demag_solver=None)

    xmin = 0.01
    xmax = 0.99
    y0 = 0.2
    z0 = 0.4
    pts1 = [[x, 0, 0] for x in np.linspace(xmin, xmax, 20)]
    pts2 = [[x, y0, z0] for x in np.linspace(xmin, xmax, 20)]
    probing_pts = np.concatenate([pts1, pts2])

    m_probed_vals = [sim.probe_field("m", pt) for pt in probing_pts]
    _, vals1 = sim.probe_field_along_line(
        "m", [xmin, 0, 0], [xmax, 0, 0], N=20)
    _, vals2 = sim.probe_field_along_line(
        "m", [xmin, y0, z0], [xmax, y0, z0], N=20)
    m_probed_vals2 = np.concatenate([vals1, vals2])

    for i in range(len(probing_pts)):  # py2 xrange -> py3 range
        m = m_probed_vals[i]
        m2 = m_probed_vals2[i]
        x = probing_pts[i][0]
        m_expected = np.array([cos(x * pi), sin(x * pi), 0.0])
        assert np.linalg.norm(m - m_expected) < TOL
        assert np.linalg.norm(m2 - m_expected) < TOL


def test_probe_m_on_regular_grid(tmpdir):
    """Probe m on a regular 2D grid using the barmini example."""
    os.chdir(str(tmpdir))

    sim = barmini()
    nx = 5
    ny = 10
    z = 5.0  # cutting plane in the middle of the cuboid
    X, Y = np.mgrid[0:3:nx * 1j, 0:3:ny * 1j]
    pts = np.array([[(X[i, j], Y[i, j], z) for j in range(ny)]  # xrange->range
                    for i in range(nx)])

    res = sim.probe_field('m', pts)

    assert res.shape == (nx, ny, 3)
    assert np.allclose(res[..., 0], 1.0 / sqrt(2))
    assert np.allclose(res[..., 1], 0.0)
    assert np.allclose(res[..., 2], 1.0 / sqrt(2))


def test_sim_with(tmpdir):
    """Call sim_with with a broad spread of parameters (master line 885)."""
    os.chdir(str(tmpdir))
    mesh_ = _boxmesh((0, 0, 0), (1, 1, 1), 3, 3, 3)  # df.UnitCubeMesh(3,3,3)
    # master passed demag_solver_params with dolfin cg/ilu solver names; the
    # port's FK demag takes the phi_1/phi_2 tolerance-dict schema instead, so
    # the dolfin-specific solver-name dict is dropped (FK defaults are used).
    sim = sim_with(mesh_, Ms=8e5, m_init=[1, 0, 0], alpha=1.0, unit_length=1e-9,
                   integrator_backend='sundials', A=13e-12, K1=520e3,
                   K1_axis=[0, 1, 1], H_ext=[0, 0, 1e6], D=6.98e-3,
                   demag_solver='FK', name='test_simulation')


# --------------------------------------------------------------------------
# RESTORED dropped-coverage regression tests (sibling master files, git
# b5015c5a). These were dropped in the port's rewrite; the audit flagged them
# as key regressions to bring back. Names/structure/tolerances are master's.
# --------------------------------------------------------------------------

alpha = 0.1  # module-level, as in master test_sim_ode.py


def test_sim_ode(do_plot=False):
    # <- src/finmag/tests/test_sim_ode.py (macrospin tanh analytic oracle).
    # master built Sim(mesh, 8.6e5, unit_length=1e-9, pbc='2d'); pbc is DEFERRED
    # by name in the port (raises NotImplementedError) and is PHYSICALLY
    # IRRELEVANT for this single-cell macrospin, so it is dropped here. The
    # deterministic-dynamics oracle -- the actual point of the test -- is
    # restored VERBATIM at master's 1e-9 tolerance.
    mesh_ = _boxmesh((0, 0, 0), (2, 2, 2), 1, 1, 1)
    sim = Simulation(mesh_, 8.6e5, unit_length=1e-9)  # master: pbc='2d' (dropped)
    sim.alpha = alpha
    sim.set_m((1, 0, 0))

    sim.set_tol(1e-12, 1e-14)

    H0 = 1e5
    sim.add(Zeeman((0, 0, H0)))

    dt = 1e-12
    ts = np.linspace(0, 500 * dt, 100)

    precession_coeff = sim.gamma / (1 + alpha ** 2)
    mz_ref = np.tanh(precession_coeff * alpha * H0 * ts)

    mzs = []
    length_error = []
    for t in ts:
        sim.advance_time(t)
        mm = sim.m.copy()

        mm.shape = (3, -1)
        mx, my, mz = mm[:, 0]  # same as m_average for this macrospin problem
        mzs.append(mz)
        length = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        length_error.append(abs(length - 1.0))

    mzs = np.array(mzs)
    print("Deviation = {}, total value={}".format(  # py2 print stmt -> print()
        np.max(np.abs(mzs - mz_ref)), mz_ref))

    # master 1e-9; measured DOLFINx deviation ~1.85e-11, length_error ~4.4e-12
    # (both pass verbatim).
    assert np.max(np.abs(mzs - mz_ref)) < 1e-9
    assert np.max(length_error) < 1e-9


def test_easy_relaxation(do_plot=False):
    # <- src/finmag/drivers/tests/test_relaxation.py
    """A simulation we expect to relax well; catches obvious relaxation bugs."""
    mesh_ = _boxmesh((0, 0, 0), (50, 10, 10), 10, 2, 2)
    Ms = 0.86e6
    A = 13.0e-12

    sim = Simulation(mesh_, Ms, name="test_relaxation")  # master: default unit_length=1
    sim.set_m((1, 0, 0))
    sim.add(Zeeman((0, Ms, 0)))
    sim.add(Exchange(A))
    sim.add(Demag())
    sim.schedule(Simulation.save_averages, every=1e-12, at_end=True)
    sim.relax()

    # master 3e-10; measured DOLFINx relaxation time ~2.24e-10 (passes verbatim).
    assert sim.t < 3e-10


def test_relax_two_times():
    # <- src/finmag/drivers/tests/test_relax_two_times.py
    """Test whether we can call relax() on a Simulation two times in a row."""
    mesh_ = _boxmesh((0, 0, 0), (10, 10, 10), 2, 2, 2)
    Ms = 0.86e6

    sim = Simulation(mesh_, Ms)  # master: default unit_length=1
    sim.set_m((1, 0, 0))

    external_field = Zeeman((0, Ms, 0))
    sim.add(external_field)
    sim.relax()
    t0 = sim.t  # time needed for first relaxation

    external_field.set_value((0, 0, Ms))
    sim.relax()
    t1 = sim.t - t0  # time needed for second relaxation

    # master tolerance 1e-10; measured |t1 - t0| ~1e-13 (passes verbatim).
    assert sim.t > t0
    assert abs(t1 - t0) < 1e-10


# ==========================================================================
# ===== NEW under DOLFINx (no master ancestor) =====
#
# The tests below are the port's original DOLFINx-specific suite: backend
# defaults, integrator lifecycle, callable pin masks, by-name deferrals,
# macro-geometry PBC demag, and the point-probing contract. They have no
# ancestor in master sim_test.py and are preserved unchanged.
# ==========================================================================


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
    property; see test_variable_params.py for the oracle-pinned RHS."""
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
    ``test_sundials_driver.py``. [Claude Opus 4.8]
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
    ``energies/demag/demag_pbc_test.py::test_pbc_coincident_tile_spacing_produces_a_non_finite_bem``
    and is refused by name by ``sim_with``.  [Claude Opus 4.8]
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
    on a flat slab) -- see
    ``energies/demag/demag_pbc_test.py::test_pbc_coincident_tile_spacing_produces_a_non_finite_bem``.
    ``sim_with`` refuses it by name rather than exposing it.
    [Claude Opus 4.8]"""
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


def test_sim_with_overlapping_macro_geometry_tiles_are_deferred_by_name():
    """Overlapping tiles (pitch < mesh extent) are as broken as touching ones.

    The periodic BEM is correct only for ``pitch > extent`` (any positive gap).
    A pitch strictly *below* the extent makes neighbouring image tiles
    interpenetrate and the BEM row sums diverge further from -1 (row-sum maxdev
    1.0 at ``0.9*extent`` -- row sums -2 -- and 2.0 at ``0.5*extent`` -- row
    sums -3), giving a silently wrong demag field.  Before this commit the guard
    rejected only the exactly-touching pitch (``pitch == extent``) and let these
    overlapping pitches slip through; it now rejects ``pitch <= extent`` on every
    active axis.  [Claude Opus 4.8]"""
    box = _box(2, 5.0)  # 5nm extent along each axis
    with pytest.raises(NotImplementedError, match="touching|overlap"):
        sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 nx=3, spacing_x=0.9 * 5.0)
    with pytest.raises(NotImplementedError, match="touching|overlap"):
        sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 nx=3, spacing_x=0.5 * 5.0)
    with pytest.raises(NotImplementedError, match="touching|overlap"):
        sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
                 ny=3, spacing_y=0.9 * 5.0)
    # a genuinely larger gap (pitch > extent) must still be accepted
    sim_with(box, Ms=8.6e5, m_init=(1.0, 0.0, 0.0), unit_length=1e-9,
             nx=3, spacing_x=5.0 * (1 + 1e-6), name="mg_overlap_ok")


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
    behavioural + oracle coverage lives in ``test_stt.py``; this only
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
    behavioural coverage lives in ``test_restart_output.py``; this only
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
    """Task 16 + SR1 P4-region: mark_regions + per-region energy/magnetisation
    accounting are ported (see test_variable_params.py), and
    ``save_m_in_region`` now registers a per-region ``<m>`` column in the .ndt
    table (faithful legacy behaviour -- NOT a field-to-file write). The
    region-restricted submesh field extraction path stays deferred by name."""
    sim = _make_sim()
    sim.set_m((1.0, 0.0, 0.0))
    ids = sim.mark_regions(lambda pt: 1 if pt[0] < 2.5 else 2)
    assert set(ids) == {1, 2}
    sim.save_m_in_region(1)  # registers an ndt column; no longer raises
    assert any(
        name.startswith("region_") for name in sim.tablewriter._entities)
    with pytest.raises(NotImplementedError, match="get_submesh"):
        sim.get_submesh(1)


def test_hysteresis_is_ported_not_deferred():
    """Task 15: relax/hysteresis/hysteresis_loop are ported; see the focused
    ``sim/hysteresis_test.py`` suite for the full behavioral contract.
    ``hysteresis([])`` returns ``None`` immediately (matching legacy) rather
    than raising."""
    sim = _make_sim()
    assert sim.hysteresis([]) is None


# --------------------------------------------------------------------------
# SR1 P2.3: callable pin masks (coordinate -> dof selection)
#
# Legacy resolved a callable pin mask in ``Simulation.__set_pins`` (sim layer):
# the callable receives ONE raw-mesh-unit coordinate triple at a time (NOT
# scaled by ``unit_length``), a truthy return marks that node pinned, and the
# resulting index is the position in the owned-node ``xxx`` coordinate ordering
# that ``LLG._pins`` consumes. ``LLG.set_pins`` stays index-only.
# --------------------------------------------------------------------------


def _pin_box_sim():
    """A [0,30]x[0,10]x[0,10] (mesh-unit) box sim matching the P2.3 probe.

    16 owned vertices; the z==0 face is the site the callable tests select.
    """
    box = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([30.0, 10.0, 10.0])],
        [3, 1, 1],
        mesh.CellType.tetrahedron,
    )
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name="pin_sim")
    sim.alpha = 0.5
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Exchange(13.0e-12))
    sim.add(Zeeman((0.0, 0.0, 1e5)))
    return sim


def test_callable_pins_intended_sites_and_holds_them():
    """A callable pins exactly the physical sites it selects, and those nodes'
    magnetisation is held exactly constant while unpinned nodes evolve."""
    sim = _pin_box_sim()
    coords, _ = sim.llg._m_field.coords_and_values()

    zmin = coords[:, 2].min()
    eps = 1e-6

    # Independently recompute the index set the callable selects, from the same
    # owned-node coordinate array ``_pins`` is indexed against.
    hand_indices = np.where(coords[:, 2] <= zmin + eps)[0]
    assert hand_indices.size > 0  # the z-min face is non-empty

    sim.pins = lambda c: c[2] <= zmin + eps

    assert set(sim.llg.pins.tolist()) == set(hand_indices.tolist())
    # Every pinned node really is on the z==zmin face.
    assert np.allclose(coords[sim.llg.pins, 2], zmin)

    m0 = sim.llg._m_field.get_ordered_numpy_array_xxx().reshape((3, -1)).copy()
    sim.run_until(2e-12)
    m1 = sim.llg._m_field.get_ordered_numpy_array_xxx().reshape((3, -1)).copy()

    pinned = sim.llg.pins.tolist()
    unpinned = [i for i in range(m0.shape[1]) if i not in set(pinned)]

    # Pinned nodes are held exactly constant (measured delta 0.0).
    for i in pinned:
        assert np.linalg.norm(m1[:, i] - m0[:, i]) == 0.0
    # At least some unpinned nodes moved (probe measured max delta ~0.0384).
    unpinned_deltas = [np.linalg.norm(m1[:, i] - m0[:, i]) for i in unpinned]
    assert max(unpinned_deltas) > 1e-3


def test_callable_pin_coordinates_are_mesh_units_not_metres():
    """The callable receives raw mesh-unit coordinates (0..10 in z), NOT metres
    (0..1e-8). A threshold of ``z <= 0 + tol`` in mesh units selects the z==0
    face; the same numeric threshold in metres would select every node."""
    sim = _pin_box_sim()
    coords, _ = sim.llg._m_field.coords_and_values()
    tol = 1e-6

    sim.pins = lambda c: c[2] <= 0.0 + tol

    selected = set(sim.llg.pins.tolist())
    expected = set(np.where(coords[:, 2] <= 0.0 + tol)[0].tolist())
    assert selected == expected

    # Guard against a future ``* unit_length`` regression: if the coordinates
    # were scaled to metres (~1e-8), the threshold 0+tol would select ALL nodes.
    assert 0 < len(selected) < coords.shape[0]


def test_callable_selecting_nothing_yields_no_pins():
    sim = _pin_box_sim()
    sim.pins = lambda c: False
    assert sim.llg.pins.size == 0


def test_callable_and_index_list_are_equivalent():
    """A callable and the explicit index list it resolves to produce identical
    ``sim.llg.pins``."""
    coords, _ = _pin_box_sim().llg._m_field.coords_and_values()
    zmin = coords[:, 2].min()
    eps = 1e-6
    hand_indices = np.where(coords[:, 2] <= zmin + eps)[0]

    sim_callable = _pin_box_sim()
    sim_callable.pins = lambda c: c[2] <= zmin + eps

    sim_indexed = _pin_box_sim()
    sim_indexed.pins = list(hand_indices)

    assert np.array_equal(
        np.sort(sim_callable.llg.pins), np.sort(sim_indexed.llg.pins)
    )


def test_indexed_pins_unchanged_by_callable_support():
    """Regression guard: setting an explicit index list still works exactly as
    before the callable path was added."""
    sim = _pin_box_sim()
    sim.pins = [0, 2]
    assert sim.llg.pins.tolist() == [0, 2]


# --------------------------------------------------------------------------
# point probing -- probe_field / probe_field_along_line (SR1 P4-probe)
# --------------------------------------------------------------------------
#
# The ``region=None`` path is fully ported: ``get_field_as_dolfin_function``
# returns the field as a DOLFINx ``Function`` and ``field.evaluate_at_point``
# restores dolfin's point-in-cell evaluation. Coordinates are MESH units (the
# legacy docstring: "point coordinates must be specified in mesh coordinates"),
# NOT metres -- ``unit_length`` is never applied, matching ``set_m(callable)``.

_PROBE_LEN = 4.0  # box edge in mesh units; vertices land on integer z in 0..4


def _probe_box():
    return mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (_PROBE_LEN, _PROBE_LEN, _PROBE_LEN)],
        [4, 4, 4],
        mesh.CellType.tetrahedron,
    )


def _m_analytic(z):
    """A spatially varying, unit-norm magnetisation rotating in the xy plane
    with height ``z``: (1,0,0) at z==0, (0,1,0) at z==_PROBE_LEN. Unit norm
    everywhere so ``set_m`` normalisation is a no-op and the CG1 nodal value at
    each mesh vertex equals this profile exactly (allowing exact per-point
    assertions). Nonlinear in z, so a wrong-point/units bug cannot pass."""
    angle = (z / _PROBE_LEN) * (np.pi / 2.0)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def _probe_sim():
    sim = Simulation(_probe_box(), 8.6e5, unit_length=1e-9, name="probe_sim")
    sim.set_m(lambda pt: tuple(_m_analytic(pt[2])))
    return sim


def test_probe_field_single_point_returns_magnetisation():
    """probe_field('m', pt) at a mesh vertex returns the analytic m there."""
    sim = _probe_sim()
    val = sim.probe_field("m", [2.0, 2.0, 3.0])
    assert np.asarray(val).shape == (3,)
    assert np.allclose(val, _m_analytic(3.0), atol=1e-12)


def test_probe_field_array_of_points_returns_per_point_values():
    """probe_field('m', [p0, p1, p2]) returns the (3, 3) stack of per-point
    values, each matching the analytic profile at that point's z."""
    sim = _probe_sim()
    pts = [[2.0, 2.0, 0.0], [2.0, 2.0, 2.0], [2.0, 2.0, 4.0]]
    vals = sim.probe_field("m", pts)
    assert np.asarray(vals).shape == (3, 3)
    for pt, val in zip(pts, vals):
        assert np.allclose(val, _m_analytic(pt[2]), atol=1e-12)


def test_probe_field_along_line_matches_analytic_profile():
    """probe_field_along_line returns (pts, vals): N vertex samples along z,
    each matching the analytic ramp, endpoints exactly (1,0,0) and (0,1,0)."""
    sim = _probe_sim()
    a = [2.0, 2.0, 0.0]
    b = [2.0, 2.0, 4.0]
    pts, vals = sim.probe_field_along_line("m", a, b, N=5)
    assert np.asarray(vals).shape == (5, 3)
    assert np.allclose(pts[0], a)
    assert np.allclose(pts[-1], b)
    for pt, val in zip(pts, vals):
        assert np.allclose(val, _m_analytic(pt[2]), atol=1e-12)
    # explicit endpoint checks against the closed-form values
    assert np.allclose(vals[0], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(vals[-1], [0.0, 1.0, 0.0], atol=1e-12)


def test_probe_field_of_exchange_is_zero_for_uniform_magnetisation():
    """Probing a computed effective-field interaction ('Exchange') at an
    interior point is finite, vector-shaped, and analytically zero for a
    uniform magnetisation."""
    sim = Simulation(_probe_box(), 8.6e5, unit_length=1e-9, name="probe_ex")
    sim.set_m((0.0, 0.0, 1.0))
    sim.add(Exchange(13.0e-12))
    val = sim.probe_field("Exchange", [2.0, 2.0, 2.0])
    assert np.asarray(val).shape == (3,)
    assert np.all(np.isfinite(val))
    assert np.allclose(val, [0.0, 0.0, 0.0], atol=1e-6)


def test_probe_field_region_is_deferred_by_name_while_none_works():
    """A non-None ``region`` raises a clear by-name NotImplementedError
    (region-restricted probing deferred); ``region=None`` works."""
    sim = _probe_sim()
    with pytest.raises(NotImplementedError, match="probe_field"):
        sim.probe_field("m", [2.0, 2.0, 2.0], region="core")
    with pytest.raises(NotImplementedError, match="probe_field_along_line"):
        sim.probe_field_along_line("m", [2, 2, 0], [2, 2, 4], region="core")
    # region=None still returns a value
    val = sim.probe_field("m", [2.0, 2.0, 2.0], region=None)
    assert np.allclose(val, _m_analytic(2.0), atol=1e-12)


def test_probe_field_coordinates_are_mesh_units_not_metres():
    """Guard against a ``* unit_length`` regression. The probe point [2,2,3] is
    in MESH units and must return the analytic profile at z==3. A spurious
    ``* unit_length`` (1e-9) would collapse the point onto the origin corner
    (z~3e-9, still inside the [0,4] box) and return the z==0 profile ~(1,0,0)
    instead -- a value clearly distinguishable from _m_analytic(3.0)."""
    sim = _probe_sim()
    val = sim.probe_field("m", [2.0, 2.0, 3.0])
    assert not np.ma.is_masked(val)
    # correct mesh-unit interpretation
    assert np.allclose(val, _m_analytic(3.0), atol=1e-12)
    # the two profiles are far apart, so a scaled (metres) interpretation cannot
    # masquerade as the mesh-unit one
    assert not np.allclose(_m_analytic(3.0), _m_analytic(0.0), atol=0.1)
    assert not np.allclose(val, _m_analytic(0.0), atol=0.1)


# ============================================================================
# ===== NOT PORTED (carried verbatim from master b5015c5a; expected to fail) =====
# ============================================================================
# Every master ``sim_test.py`` function that the MASTER LEDGER above accounts
# for as a GENUINE-GAP or as out-of-scope-here (the NormalModeSimulation /
# eigenmode / plotting / X-display / gmsh / csg module-level block) -- i.e.
# every master function NOT covered by a named port function anywhere in the
# tree -- is carried here verbatim from
# ``git show b5015c5a:src/finmag/sim/sim_test.py`` and marked
# ``@pytest.mark.not_ported``. SR1 S0 (owner decision 2026-07-27) added a
# strict ``@pytest.mark.xfail(reason="not ported: <feature> (register
# <row>)", strict=True)`` above the label for each of them, naming the
# relevant acceptance-register row; the focused ``dolfinx-src-simulation-pytest``
# gate now runs unfiltered and reports them as xfailed (strict, so porting the
# feature forces marker removal), and the non-gating inventory lane runs --
# and reports -- them as failures, which is the point. Master's own
# ``skipif``/``xfail``/``slow``/``requires_X_display`` markers are preserved
# verbatim and, where they already govern the outcome, are left as the ONLY
# outcome marker (their verdicts are not re-judged here). [owner 2026-07-27]
#
# The ONLY edits to the carried code are 2to3-level ones that would otherwise
# be a SyntaxError and break collection of this whole file (three py2 ``print``
# statements, each annotated inline). Runtime-level py2-isms that master left
# behind (``xrange``, ``dict.iteritems``, ``np.NaN``) are deliberately NOT
# modernised: they fail at call time, which is exactly the visible failure the
# inventory lane is meant to report.
#
# ``test_removing_logger_handlers_allows_to_create_many_simulation_objects``
# mutates the finmag logging level and ``resource.RLIMIT_NOFILE`` before it
# fails and, on failure, does not restore either (its own cleanup code is
# unreached). Now that SR1 S0 runs this gate unfiltered rather than
# deselecting the test, that mutation happens on every run; it has been
# empirically benign for the rest of the gate, but is called out here in
# case a future flake in a neighbouring test traces back to it.
#
# CI T3 (2026-07-28) closed the register-D24 gap this test's xfail reason
# used to name (``close_logfile``/instance teardown IS now ported --
# ``test_clean_up`` below is proof, promoted to live). This test still
# xfails, but for a DIFFERENT, unrelated reason, same class as D31's
# ``df.BoxMesh`` setup blocker: its own body constructs
# ``mesh = df.UnitIntervalMesh(1)`` with legacy ``dolfin`` absent (stubbed
# by ``_DolfinAbsent`` above), so it fails at that line with
# ``AttributeError`` before the logfile-teardown logic it actually tests is
# ever reached. Verified empirically (``pytest --runxfail``): promoting it
# (removing both markers) does NOT make it pass. Per the carry convention
# (runtime-level dolfin/py2-isms are deliberately not modernised), it stays
# under ``@pytest.mark.not_ported`` with a corrected xfail reason instead of
# the stale D24 one.
#
# Master's module-level imports are reproduced below; the ones that cannot be
# executed under DOLFINx (they raise at import time and would break collection)
# are guarded, per the carry convention.

try:  # master: ``import dolfin as df``
    import dolfin as df
except ImportError:  # pragma: no cover - the DOLFINx environment has no dolfin
    # ``df`` is dereferenced by two carried master MARKER expressions that are
    # evaluated at import/collection time rather than at call time
    # (``@pytest.mark.xfail(LooseVersion(df.__version__) >= ...)`` and
    # ``@pytest.mark.skipif("not LooseVersion(df.__version__) < ...")``).  The
    # fallback therefore reports the version the legacy lane actually runs
    # (pixi ``fenics = "2019.1.0.*"``), so master's markers keep evaluating to
    # exactly the verdicts they have on master instead of being silently
    # changed by the move. Everything else about ``df`` is absent, so the
    # carried tests fail at call time as intended.
    class _DolfinAbsent(object):
        __version__ = "2019.1.0"
    df = _DolfinAbsent()

try:  # master: ``import sh``
    import sh
except ImportError:  # pragma: no cover - ``sh`` is not in the DOLFINx env
    sh = None

try:  # master: ``from finmag import ... normal_mode_simulation``
    from finmag import normal_mode_simulation
except NotImplementedError:  # pragma: no cover - deferred by name under DOLFINx
    normal_mode_simulation = None

try:  # master: ``from finmag.normal_modes.eigenmodes import eigensolvers``
    from finmag.normal_modes.eigenmodes import eigensolvers
except ImportError:  # pragma: no cover - imports legacy dolfin
    eigensolvers = None

try:  # master: ``from finmag.util.helpers import ...``
    from finmag.util.helpers import (  # noqa: F401
        assert_number_of_files, vector_valued_function, logging_status_str,
        fnormalise)
except ImportError:  # pragma: no cover - finmag.util.helpers imports dolfin
    assert_number_of_files = vector_valued_function = None
    logging_status_str = fnormalise = None

try:  # master: ``from finmag.example.macrospin import macrospin``
    from finmag.example.macrospin import macrospin
except ImportError:  # pragma: no cover - imports legacy dolfin
    macrospin = None

# Unguarded: master had these module-level too and each imports cleanly under
# DOLFINx (the carried tests using them fail later, at call time).
import itertools  # noqa: E402
import logging  # noqa: E402
import textwrap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from glob import glob  # noqa: E402
from distutils.version import LooseVersion  # noqa: E402
from finmag import set_logging_level  # noqa: E402
from finmag.util.meshes import (  # noqa: E402,F401
    nanodisk, plot_mesh_with_paraview, mesh_volume, from_csg)
from finmag.util.mesh_templates import EllipticalNanodisk, Sphere  # noqa: E402
from finmag.sim import sim_helpers  # noqa: E402
from finmag.energies import TimeZeeman, DMI  # noqa: E402
from finmag.util.fileio import Tablereader  # noqa: E402
from finmag.util.ansistrm import ColorizingStreamHandler  # noqa: E402
from finmag.util.macrospin import make_analytic_solution  # noqa: E402
from finmag.util.consts import gamma  # noqa: E402

logger = logging.getLogger("finmag")
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSimulation(object):
    """Carried master class: only the methods NOT covered by a named port
    function are kept here (the covered ones are transcribed as module-level
    functions above). ``setup_class`` is master's, verbatim."""

    @classmethod
    def setup_class(cls):
        # N.B.: The mesh and simulation are only created once for the
        # entire test class and are re-used in each test method for
        # efficiency. Thus they should be regarded as read-only and
        # not be changed in any test method, otherwise there may be
        # unpredicted bugs or errors in unrelated test methods!
        cls.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 5, 5, 5)
        cls.sim = sim_with(cls.mesh, Ms=8.6e5, m_init=(1, 0, 0), alpha=1.0,
                           unit_length=1e-9, A=13.0e-12, demag_solver='FK')
        cls.sim.relax()

    @pytest.mark.skipif("not LooseVersion(df.__version__) < LooseVersion('1.2.0')")
    @pytest.mark.not_ported
    def test_pbc2d_m_init(self):

        def m_init_fun(pos):
            if pos[0] == 0 or pos[1] == 0:
                return [0, 0, 1]
            else:
                return [0, 0, -1]

        mesh = df.UnitSquareMesh(3, 3)

        m_init = vector_valued_function(m_init_fun, mesh)
        sim = Simulation(mesh, Ms=1, pbc2d=True)
        sim.set_m(m_init)
        expect_m = np.zeros((3, 16))
        expect_m[2, :] = np.array(
            [1, 1, 1, 1, 1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1,  1])
        expect_m.shape = (48,)

        assert np.array_equal(sim.m, expect_m)

    @pytest.mark.xfail(reason="not ported: .npy save_field surface (incremental=/overwrite=/'Demag') (register D31)", strict=True)
    @pytest.mark.not_ported
    def test_save_field(self, tmpdir):
        os.chdir(str(tmpdir))
        sim = barmini()

        # Save the magnetisation using the default filename
        sim.save_field('m')
        sim.save_field('m')
        sim.save_field('m')
        assert(len(glob('barmini_m*.npy')) == 1)
        os.remove('barmini_m.npy')

        # Save incrementally
        sim.save_field('m', incremental=True)
        sim.save_field('m', incremental=True)
        sim.save_field('m', incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 3)

        # Check that the 'overwrite' keyword works
        sim2 = barmini()
        with pytest.raises(IOError):
            sim2.save_field('m', incremental=True)
        sim2.save_field('m', incremental=True, overwrite=True)
        sim2.save_field('m', incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 2)

        sim.save_field('Demag', incremental=True)
        assert(os.path.exists('barmini_demag_000000.npy'))
        sim.save_field('Demag')
        assert(os.path.exists('barmini_demag.npy'))

        sim.save_field('Demag', filename='demag.npy', incremental=True)
        assert(os.path.exists('demag_000000.npy'))

    @pytest.mark.xfail(reason="not ported: save_m convenience shortcut (register D31)", strict=True)
    @pytest.mark.not_ported
    def test_save_m(self, tmpdir):
        """
        Similar test as 'test_save_field', but for the convenience shortcut 'save_m'.
        """
        os.chdir(str(tmpdir))
        sim = barmini()

        # Save the magnetisation using the default filename
        sim.save_m()
        sim.save_m()
        sim.save_m()
        assert(len(glob('barmini_m*.npy')) == 1)
        os.remove('barmini_m.npy')

        # Save incrementally
        sim.save_m(incremental=True)
        sim.save_m(incremental=True)
        sim.save_m(incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 3)

        # Check that the 'overwrite' keyword works
        sim2 = barmini()
        with pytest.raises(IOError):
            sim2.save_m(incremental=True)
        sim2.save_m(incremental=True, overwrite=True)
        sim2.save_m(incremental=True)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 2)

    @pytest.mark.xfail(reason="not ported: schedule('save_field', ...) shortcut (register D31)", strict=True)
    @pytest.mark.not_ported
    def test_save_field_scheduled(self, tmpdir):
        os.chdir(str(tmpdir))
        sim = barmini()
        sim.schedule('save_field', 'm', every=1e-12)
        sim.run_until(2.5e-12)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 3)
        sim.run_until(5.5e-12)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 6)

        sim.clear_schedule()
        sim.schedule('save_field', 'm', filename='mag.npy', every=1e-12)
        sim.run_until(7.5e-12)
        assert(len(glob('barmini_m_[0-9]*.npy')) == 6)
        assert(len(glob('mag_[0-9]*.npy')) == 3)

    @pytest.mark.xfail(reason="not ported: SLLG stochastic kernel (deferred: kernel='sllg' raises NotImplementedError)", strict=True)
    @pytest.mark.not_ported
    def test_sim_sllg(self, do_plot=False):
        mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(2, 2, 2), 1, 1, 1)
        sim = Simulation(mesh, 8.6e5, unit_length=1e-9, kernel='sllg')
        alpha = 0.1
        sim.alpha = alpha
        sim.set_m((1, 0, 0))
        sim.T = 0

        H0 = 1e5
        sim.add(Zeeman((0, 0, H0)))

        dt = 1e-12
        ts = np.linspace(0, 500 * dt, 100)

        precession_coeff = sim.gamma / (1 + alpha ** 2)
        mz_ref = []

        mz = []
        real_ts = []
        for t in ts:
            sim.run_until(t)
            real_ts.append(sim.t)
            mz_ref.append(np.tanh(precession_coeff * alpha * H0 * sim.t))
            # same as m_average for this macrospin problem
            mz.append(sim.m[-1])

        mz = np.array(mz)

        if do_plot:
            import matplotlib.pyplot as plt
            ts_ns = np.array(real_ts) * 1e9
            plt.plot(ts_ns, mz, "b.", label="computed")
            plt.plot(ts_ns, mz_ref, "r-", label="analytical")
            plt.xlabel("time (ns)")
            plt.ylabel("mz")
            plt.title("integrating a macrospin")
            plt.legend()
            plt.savefig(os.path.join(MODULE_DIR, "test_sllg.png"))

        print("Deviation = {}, total value={}".format(
            np.max(np.abs(mz - mz_ref)),
            mz_ref))

        assert np.max(np.abs(mz - mz_ref)) < 8e-7

    @pytest.mark.xfail(reason="not ported: SLLG stochastic kernel (deferred: kernel='sllg' raises NotImplementedError)", strict=True)
    @pytest.mark.not_ported
    def test_sim_sllg_time(self):
        mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(5, 5, 5), 1, 1, 1)
        sim = Simulation(mesh, 8.6e5, unit_length=1e-9, kernel='sllg')
        sim.alpha = 0.1
        sim.set_m((1, 0, 0))
        sim.T = 10
        assert np.max(sim.T) == 10

        ts = np.linspace(0, 1e-9, 1001)

        H0 = 1e5
        sim.add(Zeeman((0, 0, H0)))

        real_ts = []
        for t in ts:
            sim.run_until(t)
            real_ts.append(sim.t)

        print("Max Deviation = {}".format(
            np.max(np.abs(ts - real_ts))))

        assert np.max(np.abs(ts - real_ts)) < 1e-24

    @pytest.mark.xfail(reason='dolfin >=1.5')
    @pytest.mark.not_ported
    def test_mark_regions(self, tmpdir):
        os.chdir(str(tmpdir))
        sim = barmini(mark_regions=True)

        sim.save_field_to_vtk(
            'Demag', region='bottom', filename='demag_bottom.pvd')
        sim.save_field_to_vtk('Demag', region='top', filename='demag_top.pvd')
        sim.save_field_to_vtk('Demag', filename='demag_full.pvd')
        sim.save_field('Demag', region='bottom', filename='demag_bottom.npy')
        sim.save_field('Demag', region='top', filename='demag_top.npy')
        sim.save_field('Demag', filename='demag_full.npy')

        demag_bottom = np.load('demag_bottom.npy')
        demag_top = np.load('demag_top.npy')
        demag_full = sim.get_field_as_dolfin_function('Demag').vector().array()
        assert len(demag_bottom) < len(demag_full)
        assert len(demag_top) < len(demag_full)

        id_top = sim.region_ids['top']
        id_bottom = sim.region_ids['bottom']
        submesh_top = df.SubMesh(sim.mesh, sim.region_markers, id_top)
        submesh_bottom = df.SubMesh(sim.mesh, sim.region_markers, id_bottom)

        # Check that the retrieved restricted demag field vectors have the
        # expected sizes.
        assert len(demag_top) == 3 * submesh_top.num_vertices()
        assert len(demag_bottom) == 3 * submesh_bottom.num_vertices()
        assert len(demag_full) == 3 * sim.mesh.num_vertices()

    @pytest.mark.xfail(reason="not ported: Simulation.length_scales() (deferred: length_scales not ported)", strict=True)
    @pytest.mark.not_ported
    def test_length_scales(self):
        """
        Test that we can call sim.length_scales() without error and it returns a string.
        """
        info_string = self.sim.length_scales()
        assert isinstance(info_string, str)


@pytest.mark.xfail(reason="not ported: TimeZeeman auto-update in scheduler loop (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_timezeeman_is_updated_automatically(tmpdir):
    """
    Check that the TimeZeeman.update() method is called automatically
    through sim.run_until() so that the field has the correct value at
    each time step.

    """
    os.chdir(str(tmpdir))

    def check_field_value(val):
        assert(
            np.allclose(H_ext.compute_field().reshape(3, -1).T, val, atol=0, rtol=1e-8))

    t_off = 3e-11
    t_end = 5e-11

    for method_name in ['run_until', 'advance_time']:
        sim = barmini()
        f = getattr(sim, method_name)

        field_expr = df.Expression(("0", "t", "0"), t=0, degree=1)
        H_ext = TimeZeeman(field_expr, t_off=t_off)
        # this should automatically register H_ext.update(), which is what we
        # check next
        sim.add(H_ext)

        for t in np.linspace(0, t_end, 11):
            f(t)
            check_field_value([0, t, 0] if t < t_off else [0, 0, 0])


@pytest.mark.xfail(reason="not ported: .ndt writing with time-dependent field (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_ndt_writing_with_time_dependent_field(tmpdir):
    """
    Check that when we save time-dependent field values to a .ndt
    file, we actually write the values at the correct time steps (i.e.
    the ones requested by the scheduler and not the ones which are
    internally used by the time integrator).

    """
    os.chdir(str(tmpdir))
    TOL = 1e-8

    field_expr = df.Expression(("0", "t", "0"), t=0, degree=1)
    H_ext = TimeZeeman(field_expr, t_off=2e-11)
    sim = barmini()
    sim.add(H_ext)
    sim.schedule('save_ndt', every=1e-12)
    sim.run_until(3e-11)

    Hy_expected = np.linspace(0, 3e-11, 31)
    Hy_expected[20:] = 0  # should be zero after the field was switched off

    f = Tablereader('barmini.ndt')
    assert np.allclose(
        f.timesteps(), np.linspace(0, 3e-11, 31), atol=0, rtol=TOL)
    assert np.allclose(f['H_TimeZeeman_x'], 0, atol=0, rtol=TOL)
    assert np.allclose(f['H_TimeZeeman_y'], Hy_expected, atol=0, rtol=TOL)
    assert np.allclose(f['H_TimeZeeman_z'], 0, atol=0, rtol=TOL)


#@pytest.mark.skipif("True")


@pytest.mark.xfail(reason="setup blocker unrelated to register D24 (now closed): "
                    "mesh = df.UnitIntervalMesh(1) with legacy dolfin absent "
                    "(_DolfinAbsent stub) raises AttributeError before the "
                    "logfile-teardown logic under test ever runs; same class "
                    "as D31's df.BoxMesh setup blocker", strict=True)
@pytest.mark.not_ported
def test_removing_logger_handlers_allows_to_create_many_simulation_objects(tmpdir):
    """
    When many simulation objects are created in the same scripts, the
    logger will eventually complain about 'too many open files'.
    Explicitly removing logger handlers should avoid this problem.

    """
    os.chdir(str(tmpdir))
    # avoid lots of annoying info/debugging messages
    set_logging_level('WARNING')

    # Temporarily decrease the soft limit for the maximum number of
    # allowed open file descriptors (to make the test run faster and
    # ensure reproducibility across different machines).
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (42, hard_limit))

    N = 150  # maximum number of simulation objects to create

    mesh = df.UnitIntervalMesh(1)
    Ms = 8e5
    unit_length = 1e-9

    def create_loads_of_simulations(N, close_logfiles=False):
        """
        Helper function to create lots of simulation objects,
        optionally closing previously created logfiles.

        """
        for i in xrange(N):
            sim = Simulation(mesh, Ms, unit_length)
            if close_logfiles:
                sim.close_logfile()

    # The following should raise an error because lots of loggers are
    # created without being deleted again.
    with pytest.raises(IOError):
        create_loads_of_simulations(N, close_logfiles=False)

    # Remove all the file handlers created in the loop above
    hdls = list(logger.handlers)  # We need a copy of the list because we
    # are removing handlers from it below.
    for h in hdls:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.stream.close()  # this is essential, otherwise the file handler
            # will remain open
            logger.removeHandler(h)

    # The following should work since we explicitly close the logfiles
    # before each Simulation object goes out of scope.
    create_loads_of_simulations(N, close_logfiles=True)

    # Check that no file logging handler is left
    # The next line creates an error, presumably because the loop above
    # removes too many Handlers
    # py2 print statement -> py3 print() call (2to3-level mechanical fix;
    # a SyntaxError here would break collection of the whole file).
    print(logging_status_str())

    # Restore the maximum number of allowed open file descriptors. Not
    # sure this is actually necessary but can't hurt.
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


@pytest.mark.skipif("True")
@pytest.mark.not_ported
def test_schedule_render_scene(tmpdir):
    """
    Check that scheduling 'render_scene' will create incremental snapshots.

    Deactivated because it won't run on Jenkins without an X-Server.
    """
    os.chdir(str(tmpdir))
    sim = barmini()

    # Save the magnetisation using the default filename
    sim.schedule('render_scene', every=1e-11, filename='barmini_scene.png')
    sim.run_until(2.5e-11)
    assert(sorted(glob('barmini_scene_[0-9]*.png')) ==
           ['barmini_scene_000000.png',
            'barmini_scene_000001.png',
            'barmini_scene_000002.png'])


@pytest.mark.xfail(reason="not ported: Simulation.initialise_vortex (simple/feldtkeller profiles) (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_sim_initialise_vortex(tmpdir, debug=False):
    """
    Call sim.initialise_vortex() for a cylindrical sample and a cuboid.
    If debug==True, a snapshots is saved for each of them for visual
    inspection.
    """
    os.chdir(str(tmpdir))
    mesh = nanodisk(d=60, h=5, maxh=3.0)
    sim = sim_with(mesh, Ms=8e6, m_init=[1, 0, 0], unit_length=1e-9)

    def save_debugging_snapshots(sim, basename):
        if debug:
            sim.save_vtk(basename + '.pvd')
            sim.render_scene(outfile=basename + '.png')

    sim.initialise_vortex('simple', r=20)
    save_debugging_snapshots(sim, 'disk_with_simple_vortex')

    # Vortex core is actually bigger than the sample but this shouldn't matter.
    sim.initialise_vortex('simple', r=40)
    save_debugging_snapshots(sim, 'disk_with_simple_vortex2')

    # Try the Feldtkeller profile
    sim.initialise_vortex(
        'feldtkeller', beta=15, center=(10, 0, 0), right_handed=False)
    save_debugging_snapshots(sim, 'disk_with_feldtkeller_vortex')

    # Try a non-cylindrical sample, too, and optional arguments.
    sim = barmini()
    sim.initialise_vortex('simple', r=5, center=(2, 0, 0), right_handed=False)
    save_debugging_snapshots(sim, 'barmini_with_vortex')


@pytest.mark.xfail(reason="not ported: set_m discards internal relaxation state across repeated sim.relax() (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_set_m_after_relaxation(tmpdir):
    """
    Check that Simulation.set_m changes the magnetisation for relaxation
    even after some integration has taken place.

    A system with two obvious stable states for "each spin" is
    initialised; one in which the spin points in the +Z direction,
    and one in which the spin points in the -Z direction. The
    magnetisation is initialised so that the spins point in the +Z
    direction when the system is relaxed. After relaxation, the
    magnetisation is changed to encourage the reverse behaviour (all
    spins pointing in -Z). This test enforces that the an "internal
    state" of the magnetisation is not retained between the relaxations.
    """

    os.chdir(str(tmpdir))

    # Construct a 2D simulation with two obvious optima.
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(1, 1), 1, 1)
    sim = finmag.Simulation(mesh, Ms=1, unit_length=1e-9)
    sim.add(finmag.energies.UniaxialAnisotropy(1, np.array([0, 0, 1])))

    # Set the spins so that they will point in +Z upon relaxation.
    sim.set_m((0.1, 0.1, 1))
    sim.relax()
    assert sim.m_average[2] >= 0.9

    # Set the spins again so that they will point in -Z upon relaxation.
    # py2 print statements -> py3 print() calls (2to3-level mechanical fix).
    print(sim.integrator.m)
    sim.set_m((0.2, 0.2, -1))
    print(sim.integrator.m)
    sim.relax()
    assert sim.m_average[2] <= -0.9


@pytest.mark.xfail(reason="not ported: sim.relax(save_restart_data_as=/save_vtk_snapshot_as=) filename arguments (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_sim_relax_accepts_filename(tmpdir):
    """
    Check that if sim.relax() is given a filename, the relaxed state
    is saved to this file.
    """
    os.chdir(str(tmpdir))
    sim = barmini()
    sim.set_m([1, 0, 0])
    sim.set_H_ext([1e6, 0, 0])
    sim.relax(save_restart_data_as='barmini_relaxed.npz',
              save_vtk_snapshot_as='barmini_relaxed.pvd',
              stopping_dmdt=10.0)
    assert(os.path.exists('barmini_relaxed.npz'))
    assert(os.path.exists('barmini_relaxed.pvd'))

    # Check that an existing file is  overwritten.
    os.remove('barmini_relaxed.pvd')
    sim.relax(save_restart_data_as='barmini_relaxed.npz',
              stopping_dmdt=10.0)
    assert(os.path.exists('barmini_relaxed.npz'))
    assert(not os.path.exists('barmini_relaxed.pvd'))


# TODO: Separate the plotting code out so that we can run the
#       remaining tests without X display. (Max, 20.10.2014)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.requires_X_display
@pytest.mark.not_ported
def test_NormalModeSimulation(tmpdir):
    os.chdir(str(tmpdir))
    nx = ny = nz = 2
    mesh = df.UnitCubeMesh(nx, ny, nz)
    sim = normal_mode_simulation(
        mesh, Ms=8e5, A=13e-12, m_init=[1, 0, 0], alpha=1.0, unit_length=1e-9, H_ext=[1e5, 1e3, 0], name='sim')
    sim.relax(stopping_dmdt=10.0)

    t_step = 1e-13
    sim.run_ringdown(t_end=1e-12, alpha=0.01, H_ext=[1e5, 0, 0], reset_time=True,
                     save_ndt_every=t_step, save_m_every=t_step, m_snapshots_filename='foobar/foo_m.npy')

    assert(len(glob('foobar/foo_m*.npy')) == 11)
    f = Tablereader('sim.ndt')
    assert(
        np.allclose(f.timesteps(), np.linspace(0, 1e-12, 11), atol=0, rtol=1e-8))

    sim.reset_time(1.1e-12)  # hack to avoid a duplicate timestep at t=1e-12
    sim.run_ringdown(t_end=2e-12, alpha=0.02, H_ext=[1e4, 0, 0], reset_time=False,
                     save_ndt_every=t_step, save_vtk_every=2 * t_step, vtk_snapshots_filename='baz/sim_m.pvd')
    f.reload()
    assert(os.path.exists('baz/sim_m.pvd'))
    assert(len(glob('baz/sim_m*.vtu')) == 5)
    assert(
        np.allclose(f.timesteps(), np.linspace(0, 2e-12, 21), atol=0, rtol=1e-8))

    sim.plot_spectrum(use_averaged_m=True)
    sim.plot_spectrum(use_averaged_m=True, log=True, t_step=1.5e-12,
                      subtract_values='first', figsize=(16, 6), outfilename='fft_m.png')
    # sim.plot_spectrum(use_averaged_m=False)
    sim.plot_spectrum(use_averaged_m=False, t_ini=0.0, t_end=1e-12,
                      subtract_values='average', figsize=(16, 6), outfilename='fft_m_spatially_resolved.png')
    assert(os.path.exists('fft_m.png'))
    assert(os.path.exists('fft_m_spatially_resolved.png'))

    sim.plot_spectrum(
        t_step=t_step, use_averaged_m=True, outfilename='fft_m.png')

    sim.find_peak_near_frequency(10e9, component='y', use_averaged_m=True)

    sim.export_normal_mode_animation_from_ringdown('foobar/foo_m*.npy', peak_idx=2,
                                                   outfilename='animations/foo_peak_idx_2.pvd',
                                                   num_cycles=1, num_frames_per_cycle=4,
                                                   use_averaged_m=True)
    sim.export_normal_mode_animation_from_ringdown('foobar/foo_m*.npy', f_approx=0.0, component='y',
                                                   outfilename=None, directory='animations',
                                                   num_cycles=1, num_frames_per_cycle=4,
                                                   use_averaged_m=False)
    assert(os.path.exists('animations/foo_peak_idx_2.pvd'))
    assert(len(glob('animations/foo_peak_idx_2*.vtu')) == 4)

    # Either 'peak_idx' or both 'f_approx' and 'component' must be given
    with pytest.raises(ValueError):
        sim.export_normal_mode_animation_from_ringdown(
            'foobar/foo_m*.npy', f_approx=0)
    with pytest.raises(ValueError):
        sim.export_normal_mode_animation_from_ringdown(
            'foobar/foo_m*.npy', component='x')

    # Check that by default snapshots are not overwritten
    sim = normal_mode_simulation(
        mesh, Ms=8e5, A=13e-12, m_init=[1, 0, 0], alpha=1.0, unit_length=1e-9, H_ext=[1e5, 1e3, 0], name='sim')
    with pytest.raises(IOError):
        sim.run_ringdown(t_end=1e-12, alpha=0.02, H_ext=[
                         1e4, 0, 0], save_vtk_every=2e-13, vtk_snapshots_filename='baz/sim_m.pvd')
    with pytest.raises(IOError):
        sim.run_ringdown(t_end=1e-12, alpha=0.02, H_ext=[
                         1e4, 0, 0], save_m_every=2e-13, m_snapshots_filename='foobar/foo_m.npy')


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.slow
@pytest.mark.not_ported
def test_normal_mode_simulation_with_periodic_boundary_conditions_1x1(tmpdir):
    os.chdir(str(tmpdir))
    csg_string = textwrap.dedent("""
        algebraic3d
        solid cube = orthobrick (0, 0, 0; 50, 50, 3) -maxh = 3.0;
        solid cyl = cylinder (25, 25, 0; 25, 25, 1; 20);
        solid crystal = cube and not cyl;
        tlo crystal;
        """)
    mesh = from_csg(csg_string)
    sim = normal_mode_simulation(
        mesh, Ms=8e5, m_init=[1, 1, 0], A=13e-12, H_ext=None, unit_length=1e-9, pbc='1d')
    sim.relax()
    sim.save_vtk('m_relaxed.pvd')
    omega, w, relerr = sim.compute_normal_modes(solver='scipy_sparse')
    sim.plot_spatially_resolved_normal_mode(0, outfilename='mode_0.png')
    sim.plot_spatially_resolved_normal_mode(1, outfilename='mode_1.png')
    sim.plot_spatially_resolved_normal_mode(2, outfilename='mode_2.png')
    sim.export_eigenmode_animations(
        [0, 1, 2], directory='animations', create_movies=False)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.slow
@pytest.mark.not_ported
def test_normal_mode_simulation_with_periodic_boundary_conditions_9x9(tmpdir):
    os.chdir(str(tmpdir))
    csg_string = textwrap.dedent("""
        algebraic3d
        solid cube = orthobrick (0, 0, 0; 150, 150, 3) -maxh = 10.0;
        solid cylA = cylinder (25,   25, 0; 25,   25, 1; 20);
        solid cylB = cylinder (75,   25, 0; 75,   25, 1; 20);
        solid cylC = cylinder (125,  25, 0; 125,  25, 1; 20);
        solid cylD = cylinder (25,   75, 0; 25,   75, 1; 20);
        solid cylE = cylinder (75,   75, 0; 75,   75, 1; 20);
        solid cylF = cylinder (125,  75, 0; 125,  75, 1; 20);
        solid cylG = cylinder (25,  125, 0; 25,  125, 1; 20);
        solid cylH = cylinder (75,  125, 0; 75,  125, 1; 20);
        solid cylI = cylinder (125, 125, 0; 125, 125, 1; 20);
        solid crystal = cube and not cylA and not cylB and not cylC and not cylD and not cylE and not cylF and not cylG and not cylH and not cylI;
        tlo crystal;
        """)
    mesh = from_csg(csg_string)
    sim = normal_mode_simulation(
        mesh, Ms=8e5, m_init=[1, 0, 0], A=13e-12, H_ext=None, unit_length=1e-9, pbc='1d')
    sim.relax()
    sim.save_vtk('m_relaxed.pvd')
    omega, w, relerr = sim.compute_normal_modes(solver='scipy_sparse')
    sim.plot_spatially_resolved_normal_mode(0, outfilename='mode_0.png')
    sim.plot_spatially_resolved_normal_mode(1, outfilename='mode_1.png')
    sim.plot_spatially_resolved_normal_mode(2, outfilename='mode_2.png')
    sim.export_eigenmode_animations(
        [0, 1, 2], directory='animations', create_movies=False)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_H_ext_is_set_correcy_in_normal_mode_simulation(tmpdir):
    os.chdir(str(tmpdir))
    nx = ny = nz = 2
    mesh = df.UnitCubeMesh(nx, ny, nz)

    def check_H_ext(sim, value):
        if value == None:
            assert not sim.has_interaction('Zeeman')
        else:
            zeeman = sim.get_interaction('Zeeman')
            assert np.allclose(zeeman.compute_field().reshape(3, -1).T, value)

    def run_check(H_ext_1, H_ext_1_check_value, H_ext_2, H_ext_2_check_value):
        sim = normal_mode_simulation(
            mesh, Ms=8e5, A=13e-12, m_init=[1, 0, 0], alpha=1.0, unit_length=1e-9, H_ext=H_ext_1, name='sim')
        check_H_ext(sim, H_ext_1)
        sim.run_ringdown(t_end=0.0, alpha=0.01, H_ext=H_ext_2)
        check_H_ext(sim, H_ext_2_check_value)

    # Check that H_ext=None switches any existing field off during ringdown
    run_check([1e5, 1e3, 0], [1e5, 1e3, 0], None, [0, 0, 0])
    run_check(None, None, None, None)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_compute_normal_modes(tmpdir):
    """
    Compute normal modes of a simple disk system and export a couple
    of those modes to vtk files.
    """
    os.chdir(str(tmpdir))

    d = 100
    h = 10
    maxh = 10.0
    alpha = 0.0
    m_init = [1, 0, 0]
    H_ext = [1e5, 0, 0]

    mesh = nanodisk(d, h, maxh)
    sim = normal_mode_simulation(
        mesh, Ms=8e6, m_init=m_init, alpha=alpha, unit_length=1e-9, A=13e-12, H_ext=H_ext, name='nanodisk')
    omega, w, rel_errors = sim.compute_normal_modes(
        n_values=10, filename_mat_A='matrix_A.npy', filename_mat_M='matrix_M.npy')
    logger.debug("Frequencies found: {}".format(omega))
    sim.export_normal_mode_animation(
        2, filename='animation/mode_2.pvd', num_cycles=1, num_snapshots_per_cycle=10, scaling=0.1)
    sim.export_normal_mode_animation(
        5, directory='animation', num_cycles=1, num_snapshots_per_cycle=10, scaling=0.1)

    assert(os.path.exists('animation/mode_2.pvd'))
    assert(len(glob('animation/mode_2*.vtu')) == 10)
    assert(len(glob('animation/normal_mode_5__*_GHz*.pvd')) == 1)
    assert(len(glob('animation/normal_mode_5__*_GHz*.vtu')) == 10)


@pytest.mark.slow
@pytest.mark.skipif("True")
@pytest.mark.not_ported
def test_compute_eigenmode_animations(tmpdir):
    """
    Compute normal modes of a simple disk system and export a couple
    of those modes to vtk files.
    """
    os.chdir(str(tmpdir))

    d = 100
    h = 10
    maxh = 10.0
    alpha = 0.0
    m_init = [1, 0, 0]
    H_ext = [1e5, 0, 0]

    mesh = nanodisk(d, h, maxh)
    sim = normal_mode_simulation(
        mesh, Ms=8e6, m_init=m_init, alpha=alpha, unit_length=1e-9, A=13e-12, H_ext=H_ext, name='nanodisk')
    omega, w, rel_errors = sim.compute_normal_modes(
        n_values=10, filename_mat_A='matrix_A.npy', filename_mat_M='matrix_M.npy')
    logger.debug("Frequencies found: {}".format(omega))

    # Export first 4 eigenmodes without movies
    sim.export_eigenmode_animations(
        4, directory='animation_01', create_movies=False, num_cycles=1, num_snapshots_per_cycle=10, scaling=0.1)
    assert(len(glob('animation_01/*')) == 4)
    assert(len(glob('animation_01/*/*.pvd')) == 4)
    assert(len(glob('animation_01/*/*.vtu')) == 40)

    # Export first 3 eigenmodes with movies
    sim.export_eigenmode_animations(3, directory='animation_02', create_movies=True,
                                    num_cycles=1, num_snapshots_per_cycle=3, scaling=0.1)
    # 3+3 (3 folders for the eigenmodes and 3 movie files)
    assert(len(glob('animation_02/*')) == 6)
    assert(len(glob('animation_02/*/*.pvd')) == 3)
    assert(len(glob('animation_02/*/*.vtu')) == 9)
    assert(len(glob('animation_02/*.avi')) == 3)

    # Export 2 specific modes with movies
    sim.export_eigenmode_animations([5, 8], directory='animation_03', create_movies=True, directory_movies='movies_03/',
                                    num_cycles=1, num_snapshots_per_cycle=3, scaling=0.1)
    assert(len(glob('animation_03/*')) == 2)
    assert(len(glob('animation_03/*/*.pvd')) == 2)
    assert(len(glob('animation_03/*/*.vtu')) == 6)
    assert(len(glob('movies_03/*.avi')) == 2)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_compute_normal_modes_with_different_solvers(tmpdir):
    """
    Compute normal modes of a simple nanodisk using various representative
    eigensolvers (this is far from exhaustive, though).
    """
    os.chdir(str(tmpdir))

    d = 100
    h = 10
    maxh = 10.0
    alpha = 0.0
    m_init = [1, 0, 0]
    H_ext = [1e5, 0, 0]

    mesh = nanodisk(d, h, maxh)
    sim = normal_mode_simulation(
        mesh, Ms=8e6, m_init=m_init, alpha=alpha, unit_length=1e-9, A=13e-12, H_ext=H_ext, name='nanodisk')

    # Monkey-patch the eigenvalue matrices because we're not
    # interested in realistic solutions.
    # sim.A = np.diag(np.arange(1, 20+1))
    # sim.M = np.eye(20)

    # Default solver is used without any extra arguments
    omega1, w1, _ = sim.compute_normal_modes(n_values=10)

    # Scipy dense non-Hermitian solver
    solver2 = eigensolvers.ScipyLinalgEig()
    omega2a, w2a, _ = sim.compute_normal_modes(n_values=10, solver=solver2)
    omega2b, w2b, _ = sim.compute_normal_modes(
        n_values=10, solver="scipy_dense")

    # Scipy sparse non-Hermitian solver
    solver3 = eigensolvers.ScipySparseLinalgEigs(sigma=0.0, which='LM')
    omega3a, w3a, _ = sim.compute_normal_modes(n_values=10, solver=solver3)
    omega3b, w3b, _ = sim.compute_normal_modes(
        n_values=10, solver="scipy_sparse")

    # SLEPc solver
    solver4 = eigensolvers.SLEPcEigensolver(
        problem_type='GNHEP', method_type='KRYLOVSCHUR', which='SMALLEST_MAGNITUDE')
    #omega4a, w4a, _ = sim.compute_normal_modes(n_values=10, solver=solver4)
    #omega4b, w4b, _ = sim.compute_normal_modes(n_values=10, solver="slepc_krylovschur")
    with pytest.raises(TypeError):
        # Cannot currently use the SLEPcEigensolver with a generalised
        # eigenvalue problem.
        sim.compute_normal_modes(
            n_values=10, solver=solver4, use_generalised=True)

    # Check that all methods compute the same eigenvalues and eigenvectors
    #
    # Note: Currently not all of these tests pass but we're not
    # really interested in the results, only in the fact that we can
    # call these solvers.
    # assert(np.allclose(omega2a, omega1))
    # assert(np.allclose(omega2b, omega1))
    # assert(np.allclose(omega3a, omega1))
    # assert(np.allclose(omega3b, omega1))
    #assert(np.allclose(omega4a, omega1))
    #assert(np.allclose(omega4b, omega1))

    def assert_define_same_eigenspaces(vs, ws):
        """
        Assert that each pair of vectors in vs and ws defines
        is linearly dependent, which means that they define
        the same eigenspace.
        """
        for v, w in itertools.izip(vs, ws):
            # Check that v is a constant multiple of w
            a = v / w
            assert np.allclose(a, a[0])

    #assert_define_same_eigenspaces(w2a, w1)
    #assert_define_same_eigenspaces(w2b, w1)
    #assert_define_same_eigenspaces(w3a, w1)
    #assert_define_same_eigenspaces(w3b, w1)
    #assert_define_same_eigenspace(w4a, w1)
    #assert_define_same_eigenspace(w4b, w1)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.requires_X_display
@pytest.mark.not_ported
def test_plot_spatially_resolved_normal_modes(tmpdir):
    """
    Test plotting of spatially resolved normal mode profiles

    """
    os.chdir(str(tmpdir))
    from finmag.example.normal_modes import disk
    sim = disk()
    sim.compute_normal_modes()
    fig = sim.plot_spatially_resolved_normal_mode(
        k=0, outfilename='mode_00.png')
    assert(isinstance(fig, plt.Figure))


@pytest.mark.skipif("True")
@pytest.mark.requires_X_display
@pytest.mark.not_ported
def test_output_formats_for_exporting_normal_mode_animations(tmpdir):
    os.chdir(str(tmpdir))

    d = 100
    h = 10
    maxh = 10.0
    alpha = 0.0
    m_init = [1, 0, 0]
    H_ext = [1e5, 0, 0]

    mesh = nanodisk(d, h, maxh)
    sim = normal_mode_simulation(
        mesh, Ms=8e6, m_init=m_init, alpha=alpha, unit_length=1e-9, A=13e-12, H_ext=H_ext, name='nanodisk')
    omega, _, _ = sim.compute_normal_modes(
        n_values=10, filename_mat_A='matrix_A.npy', filename_mat_M='matrix_M.npy')
    logger.debug("Frequencies found: {}".format(omega))
    sim.export_normal_mode_animation(
        0, filename='animation/mode_0.pvd', num_cycles=1, num_snapshots_per_cycle=5, color_by_axis='y')
    sim.export_normal_mode_animation(
        0, filename='animation/mode_0.jpg', num_cycles=1, num_snapshots_per_cycle=5, color_by_axis='y')
    sim.export_normal_mode_animation(
        0, filename='animation/mode_0.avi', num_cycles=1, num_snapshots_per_cycle=5, color_by_axis='y')

    assert(os.path.exists('animation/mode_0.pvd'))
    assert(len(glob('animation/mode_0*.vtu')) == 5)
    assert(len(glob('animation/mode_0*.jpg')) == 5)
    assert(os.path.exists('animation/mode_0.avi'))

    with pytest.raises(ValueError):
        sim.export_normal_mode_animation(0, filename='animation/mode_0.quux')


@pytest.mark.xfail
@pytest.mark.not_ported
def test_setting_different_material_parameters_in_different_regions(tmpdir):
    """
    In this test we create a simulation with two different regions where the
    material parameters and initial magnetistaion are different for each of the
    regions.

    The goal is to check whether the initialisation of sucha  simulation works.
    However, currently we construct the dolfin Functions representing the material
    parameters by hand. Ideally, there would be a simpler way, e.g. by simply
    saying:

        sim.set_field('Ms', 8.6e5, region='nanodisk')

    This will (hopefully) eventually be implemented, but in order to do this
    lot of refactoring needs to be done so that fields can be treated in a unified
    way even though some of them may be defined within the mesh cells (such as
    Ms, A, etc.) and some of them on the nodes. Once this goal has been reached,
    this test will be a proper test of that functionality, too. For now, it
    mainly documents how to set varying parameters in different regions.

    """
    os.chdir(str(tmpdir))

    # Create a mesh consisting of a nanodisk with a spherical article on top.
    d1_disk = 50
    d2_disk = 30
    h_disk = 5
    r_sphere = 5
    center_sphere = (0, 0, 20)

    nanodisk = EllipticalNanodisk(
        d1=d1_disk, d2=d2_disk, h=h_disk, name='Nanodisk')
    sphere = Sphere(r=r_sphere, center=center_sphere, name='Sphere')
    disk_with_sphere = nanodisk + sphere

    mesh = disk_with_sphere.create_mesh(maxh=10.0, filename=os.path.join(
        MODULE_DIR, 'nanodisk_with_spherical_particle.xml.gz'))
    logger.debug(mesh)

    def is_inside_nanodisk(pt):
        x, y, z = pt
        return z <= h_disk + df.DOLFIN_EPS

    m_init_nanodisk = [1, 0, 0]
    m_init_sphere = [0, 0, 1]
    alpha_nanodisk = 0.2
    alpha_sphere = 0.08
    A_nanodisk = 13e-12
    A_sphere = 23.0
    Ms_nanodisk = 8.6e5
    Ms_sphere = 42.0
    K1_nanodisk = 0.1
    K1_sphere = 7.4e5
    K1_axis_nanodisk = [1, 0, 0]
    K1_axis_sphere = [0, 1, 0]
    D_nanodisk = 1.0
    D_sphere = 2.0

    # Define the material parameters, where each of them takes a different
    # value in each of the regions
    def m_init(pt):
        return m_init_nanodisk if is_inside_nanodisk(pt) else m_init_sphere

    def alpha(pt):
        return alpha_nanodisk if is_inside_nanodisk(pt) else alpha_sphere

    def A(pt):
        return A_nanodisk if is_inside_nanodisk(pt) else A_sphere

    def Ms(pt):
        return Ms_nanodisk if is_inside_nanodisk(pt) else Ms_sphere

    def K1(pt):
        return K1_nanodisk if is_inside_nanodisk(pt) else K1_sphere

    def K1_axis(pt):
        return K1_axis_nanodisk if is_inside_nanodisk(pt) else K1_axis_sphere

    def D(pt):
        return D_nanodisk if is_inside_nanodisk(pt) else D_sphere

    # Create a simulation object with the mesh above
    sim = sim_with(mesh, Ms=Ms, m_init=m_init, alpha=alpha, unit_length=1e-9,
                   A=A, K1=K1, K1_axis=K1_axis, D=D, name='nanodisk_with_particle')

    # Construct a CellFunction representing the subdomains. Unfortunately,
    # dolfin doesn't seem to provide an easy way of getting this from
    # mesh.domains() directly, so we construct it manually.
    # 3 represents the dimension of the mesh cells
    cell_markers = mesh.domains().markers(3)
    fun_subdomains = df.CellFunction('size_t', mesh)
    for (cell_no, marker) in cell_markers.iteritems():
        fun_subdomains[cell_no] = marker

    submesh_nanodisk = df.SubMesh(mesh, fun_subdomains, 0)
    submesh_sphere = df.SubMesh(mesh, fun_subdomains, 1)
    plot_mesh_with_paraview(
        submesh_nanodisk, camera_position=[0, -200, 100], outfile='submesh_nanodisk.png')
    plot_mesh_with_paraview(
        submesh_sphere, camera_position=[0, -200, 100], outfile='submesh_sphere.png')

    f = df.File("m.pvd")
    f << sim._m

    f = df.File("alpha.pvd")
    f << sim.alpha

    f = df.File("A.pvd")
    f << sim.get_interaction('Exchange').A

    f = df.File("Ms.pvd")
    f << sim.llg.Ms

    f = df.File("K1.pvd")
    f << sim.get_interaction('Anisotropy').K1

    f = df.File("K1_axis.pvd")
    f << sim.get_interaction('Anisotropy').axis

    f = df.File("D.pvd")
    f << sim.get_interaction('DMI').D_on_mesh

    # TODO: Extract all the values of m_init, Ms, ... on each of the submeshes.
    # This should give numpy arrays of dolfin.Function vectors which are constant
    # and whose length matches the number of nodes in the submesh. Both of these
    # properties should be checked for each field.
    raise NotImplementedError


@pytest.mark.xfail(reason="not ported: compute_energy scaling with non-normalised m (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_compute_energies_with_non_normalised_m(tmpdir):
    """
    Check that we can set the magnetisation to non-normalised values, and that the
    energy terms scale either linearly or quadratically with the magnetisation.

    XXX TODO: Should this be broken up into separate unit tests for the individual energies?

    """
    # Create a simulation with non-uniform magnetisation and a bunch of energy
    # terms
    sim = finmag.example.normal_modes.disk()
    K1 = 7.4e5
    anis = UniaxialAnisotropy(K1=K1, axis=[1, 1, 1])
    sim.add(anis)

    # Compute and store the energy values for the normalised magnetisation
    m0 = sim.m
    energies = {}
    scaling_exponents = {
        'Exchange': 2, 'Anisotropy': 2, 'Demag': 2, 'Zeeman': 1}
    for name in scaling_exponents.keys():
        energies[name] = sim.compute_energy(name)

    # Scale the magnetisation and check whether the energy terms scale
    # accordingly
    vol_mesh = mesh_volume(sim.mesh) * sim.unit_length ** 3
    for a in np.linspace(0.0, 1.0, 20):
        sim.set_m(a * m0, normalise=False)

        # Check that m was indeed scaled down
        m_norms = [np.linalg.norm(x) for x in sim.m.reshape(3, -1).T]
        assert np.allclose(m_norms, a, atol=1e-12, rtol=1e-12)

        # Check that the energy terms scale correctly
        for (name, exponent) in scaling_exponents.iteritems():
            if name == 'Anisotropy':
                # We need a separate case for the anisotropy due to the constant that
                # we're adding in the definition of the anisotropy energy.
                #
                # Note that in the assert statement we need a tiny value of 'atol'
                # for the case a=0 because the test will fail otherwise due to small
                # rounding errors when subtracting the mesh volume.
                assert(np.allclose((sim.compute_energy(name) - K1 * vol_mesh), a **
                                   exponent * (energies[name] - K1 * vol_mesh), atol=1e-31, rtol=1e-12))
            else:
                assert(np.allclose(
                    sim.compute_energy(name), a ** exponent * energies[name], atol=0, rtol=1e-12))


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.requires_X_display
@pytest.mark.not_ported
def test_compute_and_plot_power_spectral_density_in_mesh_region(tmpdir):
    """
    Create a simulation with two mesh regions, where the magnetisation
    is uniform in each region and external fields of different
    strengths are applied in each one. Record the precession for a
    certain time and then compute the spectrume in the first region
    only.

    """
    os.chdir(str(tmpdir))

    sphere1 = Sphere(r=10, center=(-20, 0, 0), name='sphere1')
    sphere2 = Sphere(r=10, center=(+20, 0, 0), name='sphere2')
    mesh = (sphere1 + sphere2).create_mesh(maxh=10.0)

    alpha1 = 0.05
    alpha2 = 0.03

    H_ext_1 = [0, 0, 1e7]
    H_ext_2 = [0, 0, 3.8e7]

    omega1 = gamma * H_ext_1[2]
    omega2 = gamma * H_ext_2[2]

    def fun_alpha(pt):
        return alpha1 if (pt[0] < 0) else alpha2

    def fun_H_ext(pt):
        return H_ext_1 if (pt[0] < 0) else H_ext_2

    def fun_regions(pt):
        return 'left' if (pt[0] < 0) else 'right'

    sim = normal_mode_simulation(mesh, Ms=8.6e5, m_init=[
                                 1, 0, 0], alpha=fun_alpha, H_ext=fun_H_ext, A=13e-12, unit_length=1e-9, demag_solver=None)
    sim.mark_regions(fun_regions)

    t_ini = 0.0
    t_end = 1e-11
    t_step = 1e-13

    ts = np.arange(t_ini, t_end, t_step)

    sim.run_ringdown(t_end, alpha=fun_alpha, H_ext=fun_H_ext,
                     save_m_every=t_step, m_snapshots_filename='m_precession.npy')

    # Create an analytic solution for the precession in region #1 with
    # which we can compare.
    H = H_ext_1[2]
    m_analytic = make_analytic_solution(H, alpha1)

    m_dynamics = np.array([m_analytic(t) for t in ts])
    mx = m_dynamics[:, 0]
    my = m_dynamics[:, 1]
    mz = m_dynamics[:, 2]

    num_vertices = mesh.num_vertices()
    psd_mx_expected = num_vertices * np.absolute(np.fft.rfft(mx)) ** 2
    psd_my_expected = num_vertices * np.absolute(np.fft.rfft(my)) ** 2
    psd_mz_expected = num_vertices * np.absolute(np.fft.rfft(mz)) ** 2

    sim._compute_spectrum(
        use_averaged_m=False, mesh_region='left', t_step=t_step, t_ini=t_ini, t_end=t_end)

    # XXX TODO: The following check doesn't work yet - I'm not even
    #           sure it's correct. Double-check this, or find a better
    #           check (perhaps check the location of the peaks?!?)
    #
    # Check that the analytically determined power spectra are the same as the computed ones.
    # RTOL = 1e-10
    # assert(np.allclose(psd_mx_expected, sim.psd_mx, atol=0, rtol=RTOL))
    # assert(np.allclose(psd_my_expected, sim.psd_my, atol=0, rtol=RTOL))
    # assert(np.allclose(psd_mz_expected, sim.psd_mz, atol=0, rtol=RTOL))

    sim.plot_spectrum(t_step=t_step, t_ini=t_ini, t_end=t_end,
                      mesh_region='left', ticks=11, outfilename='spectrum_left.png')
    sim.plot_spectrum(t_step=t_step, t_ini=t_ini, t_end=t_end,
                      mesh_region='right', ticks=11, outfilename='spectrum_right.png')

    logger.debug("Precession frequency 1: {} GHz".format(omega1 / 1e9))
    logger.debug("Precession frequency 2: {} GHz".format(omega2 / 1e9))


@pytest.mark.skipif("True")
@pytest.mark.not_ported
def test_regression_schedule_switch_off_field(tmpdir):
    """
    This is a test to remind myself to attempt a bugfix for this issue.

    Due to the way the Tablewriter works at the moment, there is an
    error if an interaction (e.g. the Zeeman interaction) is removed
    from the simulation after some of its data has been written to a
    file.

    Once this works, the default value for the keyword argument
    'remove_interaction' in the function 'sim.switch_off_H_ext()'
    should perhaps be set to True again (because it is more
    efficient).

    """
    sim = macrospin()
    sim.schedule('save_ndt', every=1e-12)
    sim.schedule('switch_off_H_ext', at=3e-12, remove_interaction=True)
    sim.run_until(5e-12)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation.run_ringdown H_ext intended-behaviour documentation (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_document_intended_behaviour_for_H_ext(tmpdir, debug=False):
    """
    The purpose of this test is simply to document how the argument
    H_ext is interpreted in the run_ringdown method of the
    NormalModeSimulation class. Namely, if a field was applied during
    the relaxation stage then using 'run_ringdown(..., H_ext=None)'
    will switch that field off!

    """
    os.chdir(str(tmpdir))
    mesh = df.UnitCubeMesh(3, 3, 3)

    # Create a simulation with an external field
    sim = normal_mode_simulation(mesh, Ms=8.6e5, alpha=0.0, m_init=[1, 0, 0],
                                 unit_length=1e-9, A=None, H_ext=[0, 0, 1e6],
                                 demag_solver=None, name='precession')

    # Call 'run_ringdown' with no external field
    sim.run_ringdown(t_end=1e-11, alpha=0.0, save_ndt_every=2e-13, H_ext=None)

    # Extract the dynamics from the .ndt file
    f = Tablereader('precession.ndt')
    ts, m_x, m_y, m_z = f['time', 'm_x', 'm_y', 'm_z']

    # Check that the magnetisation didn't change over the course of
    # the simulation (since the external field was automatically
    # switched off for the ringdown).
    assert(len(ts) == 51)
    assert(np.allclose(m_x, 1))
    assert(np.allclose(m_y, 0))
    assert(np.allclose(m_z, 0))

    #
    # Repeat the test above, but this time specify an external field
    # in 'run_ringdown'. This should result in a sinusoidal
    # precession.
    #
    H = 4.2e7  # external field strength (A/m)
    sim = normal_mode_simulation(mesh, Ms=8.6e5, alpha=0.0, m_init=[1, 0, 0],
                                 unit_length=1e-9, A=None, H_ext=[0, 0, H],
                                 demag_solver=None, name='precession2')
    sim.run_ringdown(
        t_end=1e-11, alpha=0.0, save_ndt_every=1e-13, H_ext=[0, 0, H])

    freq = gamma * H / (2 * pi)  # expected frequency

    f = Tablereader('precession2.ndt')
    ts, m_x, m_y, m_z = f['time', 'm_x', 'm_y', 'm_z']

    if debug:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(ts, m_x)
        ax.plot(ts, np.cos(2. * pi * freq * ts))
        fig.savefig('dynamics2.png')

    # Check that we get sinusoidal dynamics of the correct frequency
    TOL = 1e-3
    assert(len(ts) == 101)
    assert(np.allclose(m_x, np.cos(2 * pi * freq * ts), atol=TOL))
    assert(np.allclose(m_y, np.sin(2 * pi * freq * ts), atol=TOL))


@pytest.mark.xfail(reason="not ported: m_average volume-weighting robustness across mesh discretisation (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_m_average_is_robust_with_respect_to_mesh_discretization(tmpdir, debug=False):
    """
    This test checks that the average magnetisation takes the
    different cell volumes in the mesh correctly into account, i.e. it
    gives a bigger contribution to cells with a large volume than
    those with a small volume.

    We create a mesh representing a long strip which is coarse at the
    left and and fine at the right end. Then we initialise a
    magnetisation pattern which does a full turn, so that the average
    magnetisation is zero and check that the computed average is close
    to that.

    """
    os.chdir(str(tmpdir))

    lx = 50
    ly = 5
    lz = 3

    # We create the mesh of the nanostrip using gmsh, by writing the
    # following string to a file and converting it using gmsh and
    # dolfin-convert.
    geofile_string = textwrap.dedent("""
        nz = 1;  // number of z-layers
        lx = 50;
        ly = 5;
        lz = 3;
        lc_left = 2.0;
        lc_right = 0.1;

        Point(1) = {0, 0, 0, lc_left};
        Point(2) = {lx, 0, 0, lc_right};
        Point(3) = {lx, ly, 0, lc_right};
        Point(4) = {0, ly, 0, lc_left};

        l1 = newreg; Line(l1) = {1, 2};
        l2 = newreg; Line(l2) = {2, 3};
        l3 = newreg; Line(l3) = {3, 4};
        l4 = newreg; Line(l4) = {4, 1};

        ll1 = newreg; Line Loop(ll1) = {l1, l2, l3, l4};

        s1 = newreg; Plane Surface(s1) = {ll1};

        Extrude {0, 0, lz} {
            Surface{s1}; Layers{nz};
        }
        """)

    with open('nanostrip.geo', 'w') as f:
        f.write(geofile_string)

    # Call gmsh and dolfin-convert to bring the mesh defined above
    # into a form that's readable by dolfin.
    sh.gmsh('-3', '-optimize', '-optimize_netgen', '-o', 'nanostrip.msh', 'nanostrip.geo')
    sh.dolfin_convert('nanostrip.msh', 'nanostrip.xml')

    mesh = df.Mesh('nanostrip.xml')

    # Define magnetisation that performs a full rotation from the left
    # end to the right end of the strip.
    def m_init(pt):
        x, y, z = pt
        return [0, sin(2 * pi * x / lx), cos(2 * pi * x / lx)]

    sim = sim_with(mesh, Ms=8.6e5, m_init=m_init, unit_length=1e-9)

    # Check that the average magnetisation is close to zero
    m_avg = sim.m_average
    assert(np.allclose(m_avg, [0, 0, 0], atol=1e-3))
    logger.debug("m_avg: {}".format(m_avg))

    # Check that simply adding up the magnetization values at the vertices
    m_avg_wrong = sim.m.reshape(3, -1).sum(axis=-1)
    assert(not np.allclose(m_avg_wrong, [0, 0, 0], atol=10.0))
    logger.debug("m_avg_wrong: {}".format(m_avg_wrong))

    if debug:
        plot_mesh_with_paraview(mesh, outfile='mesh.png')
        sim.render_scene(
            color_by_axis='y', glyph_scale_factor=2, outfile='nanostrip.png')


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.not_ported
def test_eigenfrequencies_scale_with_gyromagnetic_ratio(tmpdir):
    """
    The eigenfrequencies should scale linearly with the value of gamma in the simulation.
    Here we check this for a few values of gamma.

    """
    os.chdir(str(tmpdir))

    # Create a small test simulation and compute its eigenfrequencies
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(10, 20, 3), 3, 6, 1)
    sim = normal_mode_simulation(
        mesh, m_init=[1, 0, 0], Ms=8e5, A=13e-12, unit_length=1e-9)
    omega_ref, _, _ = sim.compute_normal_modes()

    # Set gamma to some different values and check that the eigenfrequencies
    # scale accordingly
    for a in [0.7, 1.3, 2.445, 12.0]:
        sim.gamma = a * gamma
        omega, _, _ = sim.compute_normal_modes(force_recompute_matrices=True)
        assert np.allclose(omega, a * omega_ref)


@pytest.mark.xfail(reason="not ported: NormalModeSimulation/eigenmode module-level surface (deferred: normal modes)", strict=True)
@pytest.mark.requires_X_display
@pytest.mark.not_ported
def test_plot_dynamics(tmpdir):
    """
    Check whether we can call the functions `sim.plot_dynamics`
    and `sim.plot_dynamics_3d` without errors.
    """
    os.chdir(str(tmpdir))
    mesh = df.UnitCubeMesh(1, 1, 1)
    sim = sim_with(
        mesh, Ms=8e5, m_init=[1, 0, 0], A=13e-12, H_ext=[0, 0, 5e4], demag_solver=None)
    sim.schedule('save_ndt', every=5e-12)
    sim.run_until(5e-11)
    sim.plot_dynamics(figsize=(16, 3), outfile='dynamics_2d.png')
    sim.plot_dynamics_3d(figsize=(5, 5), outfile='dynamics_3d.png')


# XXX TODO: Unfortunately, calling sim.profile from within a py.test
# environment doesn't work because the namespace is not exposed to
# cProfile in the usual way. Is there a way to work around this?


@pytest.mark.xfail
@pytest.mark.not_ported
def test_profile(tmpdir):
    """
    Check whether we can call sim.profile() and whether it
    saves profiling data to th expected files.

    """
    os.chdir(str(tmpdir))
    sim = barmini()
    sim.profile('run_until(1e-12)')
    os.path.exists('barmini.prof')

    sim.profile('relax()', filename='foobar.prof', sort=-1, N=5)
    os.path.exists('foobar.prof')


# PROMOTED (CI T3, 2026-07-28, register D24 gap closed): the teardown
# surface (shutdown/instances_delete_all_others/instances_list_all/
# instances_delete_all/instances_alive_count/close_logfile) is now ported
# in sim.py, so this master test runs live. Both the strict xfail and the
# not_ported marker are removed. Master's own placement (bottom of the
# file) is kept: it shuts down every other live Simulation instance and so
# must run last.
def test_clean_up():
    """Fake test to shutdown simulation objects"""
    s = barmini()
    s.instances_delete_all_others()
    del s
