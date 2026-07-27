"""DOLFINx core ``Simulation`` tests -- minimal-diff transcription + new suite.

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
#   duplicate it):
#     test_schedule, test_save_ndt, test_save_restart_data, test_restart,
#     test_reset_time, test_save_vtk, test_sim_schedule_clear, test_save_field,
#     test_save_m, test_save_field_scheduled  -> test_restart_output_dolfinx.py
#     test_set_stt                             -> test_stt_dolfinx.py
#     test_get_field_as_dolfin_function,
#     test_probe_demag_field                   -> probe_field tests below the
#                                                 banner + energies/demag/fk_demag_test.py
#
#   GENUINE-GAP (surface not provided by the ported Simulation; reported for an
#   owner decision, NOT fabricated):
#     test_sim_sllg, test_sim_sllg_time  -- SLLG stochastic kernel is deferred
#         by name (kernel='sllg' raises NotImplementedError).
#     test_pbc2d_m_init                  -- periodic boundaries deferred (pbc
#         raises NotImplementedError); master itself skipif(dolfin<1.2.0).
#     test_mark_regions                  -- region-restricted field->vtk export
#         deferred; master itself xfail on dolfin>=1.5.
#     test_length_scales                 -- Simulation.length_scales() not ported.
#     test_clean_up                      -- Simulation.instances_delete_all_others()
#         / shutdown() NOT ported: the port keeps a plain ``instances`` dict but
#         deliberately holds no cyclic references (see sim.py class comment), so
#         the master cyclic-reference cleanup surface is absent. See report.
#
# All NormalModeSimulation / eigenmode / plotting / X-display / gmsh / csg
# module-level functions in sim_test.py belong to the (separate)
# NormalModeSimulation port, not this core Simulation port, and are out of
# scope here.
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
    ``energies/demag/demag_pbc_test.py``::
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
    on a flat slab) -- see ``energies/demag/demag_pbc_test.py``::
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
    ``test_hysteresis_dolfinx.py`` suite for the full behavioral contract.
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
