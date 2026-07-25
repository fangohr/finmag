"""Direct DOLFINx ``relax``/``hysteresis``/``hysteresis_loop`` port (Task 15).

``src/finmag/sim/sim_relax.py`` and ``src/finmag/sim/hysteresis.py`` are the
*untouched* legacy modules (verified byte-for-byte identical to the frozen
oracle checkout); the only edit made anywhere in this slice is replacing
their ``from finmag.util.helpers import ...`` (legacy ``dolfin``-backed)
import with a local, dolfin-free reimplementation in
``finmag.sim.sim_helpers`` (``norm``/``compute_dmdt``), matching the
``clean_filename`` precedent from Task 12. ``Simulation.relax``/
``hysteresis``/``hysteresis_loop`` are bound exactly the way legacy did
(``relax = sim_relax.relax``, etc).

DIAGNOSIS CORRECTION (SR1 P2.5, acceptance register D4, owner decision
2026-07-23): an earlier revision of this module (and the D4 register row)
claimed that ``hysteresis()``/``hysteresis_loop()`` "do not independently
re-relax after the first stage" -- attributing the on-axis oracle stall to a
re-relaxation defect (the first scheduled tick of each stage coinciding with
``t == integrator.cur_t``, which ``Scheduler.run()`` skips). Direct
measurement under the approved-fix investigation shows that claim is **not
reproducible**, so it is corrected here rather than acted on:

* ``relax()`` resets ``sim.relaxation = {}`` at the very start of every call
  (``sim_relax.py``), and its scheduler trigger is removed at the end of each
  stage, so **no relaxation state leaks between stages**. Each stage rebuilds
  its stopping condition and integrates to ITS OWN equilibrium. The
  ``t == integrator.cur_t`` skip is real but harmless: that first tick never
  integrates in any stage (it only records the ``last_m`` baseline);
  subsequent ticks integrate normally.
* In **non-degenerate** geometry the shared ``hysteresis()`` path re-relaxes
  fully every stage and switches the magnetisation -- **bit-for-bit identical**
  to a fresh ``Simulation``/``relax()`` per stage (witnessed by
  ``test_hysteresis_switches_each_stage_tilted_axis`` and
  ``test_stoner_wohlfarth_like_loop_witness`` below).
* The on-axis oracle loop does not switch because the applied field is exactly
  antiparallel to the easy axis, placing m at the **unstable Stoner-Wohlfarth
  saddle** where the torque is genuinely ~0. A *fresh* ``relax()`` from that
  same state does not switch either, so this is **correct degenerate-geometry
  physics, not a skipped relaxation**. The divergence between "buggy" and
  "corrected" here is exactly ZERO. The legacy test suite documents the same
  triviality with a wink (see
  ``src/finmag/sim/hysteresis_test.py::test_hysteresis_loop_and_plotting``:
  "Check that the magnetisation values are as trivial as we expect them to be
  ;-)").

Accordingly ``test_oracle_hysteresis_loop_matches_legacy`` and
``test_hysteresis_loop_legacy_trivial_invariant`` below are **DEGENERACY
PINS** (unstable-saddle geometry, faithfully preserved from legacy
``ba928093``), not "preserved re-relax defect" pins; their numeric assertions
are unchanged, only their rationale is corrected. No source file was modified
by P2.5 -- the core relaxation code already re-relaxes each stage correctly.

[Claude Sonnet 5; P2.5 diagnosis correction Claude Opus 4.8]

SR1 P5.2 minimal-diff transcription (2026-07-25, Claude Sonnet 5): the module
docstring above predates the literal master transcription. This slice adds a
MINIMAL-DIFF transcription of master's actual two ``hysteresis_test.py``
functions (git ``b5015c5a``) -- ``test_hysteresis`` and
``test_hysteresis_loop_and_plotting`` -- placed directly below the
module-level constants and ABOVE the ``NEW under DOLFINx`` banner. This
restores acceptance-register audit D4's dropped coverage: master's
``test_hysteresis`` exercises the ``fun=None`` code path with a NON-EMPTY
``H_ext_list`` (``res1 = sim1.hysteresis(H_ext_list=...); assert res1 ==
None``), which the port previously covered only for the EMPTY-list case
(``test_hysteresis_empty_list_returns_none``, preserved unchanged below the
banner). Everything from ``test_relax_reduces_dmdt_below_threshold`` onward
has no master ancestor by name and now lives below the banner accordingly.
"""

import json
import os
from glob import glob

import matplotlib
# Headless Agg backend so plot_hysteresis_loop (imported below, which pulls
# in matplotlib.pyplot) never needs an X display -- same precedent as
# test_plot_helpers_dolfinx.py. Must be set before the plot_helpers import.
matplotlib.use("Agg")

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from finmag import sim_with
from finmag.example import barmini
from finmag.energies import UniaxialAnisotropy, Zeeman
from finmag.sim.sim import Simulation
from finmag.util.plot_helpers import plot_hysteresis_loop

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "hysteresis_oracle.json")
ORACLE = json.load(open(_FIXTURE))

# Master module-level constants (git b5015c5a), used by the transcribed
# test_hysteresis_loop_and_plotting below; ONE_DEGREE_PER_NS itself is
# unreferenced in master's own two functions (dead constant, preserved
# verbatim) -- not to be confused with sim_relax.py's own internal copy or
# this file's below-banner ``_ONE_DEGREE_PER_NS``.
ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

H = 0.2e6  # maximum external field strength in A/m
initial_direction = np.array([1.0, 0.01, 0.0])
N = 5


def _box(nx=1):
    # dolfin->dolfinx: df.BoxMesh(...) -> dolfinx_mesh.create_box(...).
    return dolfinx_mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [nx, nx, nx],
        dolfinx_mesh.CellType.tetrahedron)


def _make_sim(m_init=(1.0, 0.05, 0.0), alpha=1.0, K1=1.0e4,
              easy_axis=(1.0, 0.0, 0.0), Ms=8.6e5, unit_length=1e-9, nx=1,
              name="hyst_sim"):
    sim = Simulation(_box(nx), Ms, unit_length=unit_length, name=name)
    sim.set_m(m_init)
    sim.alpha = alpha
    if K1 is not None:
        sim.add(UniaxialAnisotropy(K1, easy_axis))
    return sim


# ==========================================================================
# MINIMAL-DIFF transcription of master hysteresis_test.py (git b5015c5a).
# Function names, order and assertion structure are master's; the only
# differences are dolfin->dolfinx API changes (annotated inline), the
# py2->py3 xrange->range conversion, and explanatory comments.
# ==========================================================================

def test_hysteresis(tmpdir):
    os.chdir(str(tmpdir))
    sim = barmini()
    # dolfin->dolfinx: df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 1, 1, 1)
    # -> dolfinx_mesh.create_box(...) (same substitution as
    # finmag.example.bar._box_mesh).
    mesh = dolfinx_mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [1, 1, 1],
        dolfinx_mesh.CellType.tetrahedron)
    H_ext_list = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
    N = len(H_ext_list)

    # Run a relaxation and save a vtk snapshot at the end of each stage;
    # this should result in three .vtu files (one for each stage).
    sim1 = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0,
                    unit_length=1e-9, A=None, demag_solver=None)
    sim1.schedule('save_vtk', at_end=True, filename='barmini_hysteresis.pvd')
    res1 = sim1.hysteresis(H_ext_list=H_ext_list)
    assert(len(glob('barmini_hysteresis*.vtu')) == N)
    assert(res1 == None)

    # Run a relaxation with a non-trivial `fun` argument and check
    # that we get a list of return values.
    sim2 = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0,
                    unit_length=1e-9, A=None, demag_solver=None)
    res2 = sim2.hysteresis(H_ext_list=H_ext_list,
                           fun=lambda sim: sim.m_average[0])
    assert(len(res2) == N)


@pytest.mark.requires_X_display
def test_hysteresis_loop_and_plotting(tmpdir):
    """
    Call the hysteresis loop with various combinations for saving
    snapshots and check that the correct number of vtk files have been
    produced. Also check that calling the plotting function works
    (although the output image isn't verified).

    """
    os.chdir(str(tmpdir))

    mesh = dolfinx_mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [1, 1, 1],
        dolfinx_mesh.CellType.tetrahedron)
    sim = sim_with(mesh, Ms=1e6, m_init=(0.8, 0.2, 0), alpha=1.0,
                   unit_length=1e-9, A=None, demag_solver=None)
    H_vals, m_vals = \
        sim.hysteresis_loop(H, initial_direction, N, stopping_dmdt=10)

    # Check that the magnetisation values are as trivial as we expect
    # them to be ;-)
    assert(np.allclose(m_vals, [1.0 for _ in range(2 * N)], atol=1e-4))  # py2->3: xrange->range

    # This only tests whether the plotting function works without
    # errors. It currently does *not* check that it produces
    # meaningful results (and the plot is quite boring for the system
    # above anyway).
    plot_hysteresis_loop(H_vals, m_vals, infobox=["param_A = 23", ("param_B", 42)],
                         title="Hysteresis plot test", xlabel="H_ext", ylabel="m_avg",
                         figsize=(5, 4), infobox_loc="bottom left",
                         filename='test_plot.pdf')

    # Test multiple filenames, too
    plot_hysteresis_loop(
        H_vals, m_vals, filename=['test_plot.pdf', 'test_plot.png'])


# ===== NEW under DOLFINx (no master ancestor) =====

# --------------------------------------------------------------------------
# relax()
# --------------------------------------------------------------------------

def test_relax_reduces_dmdt_below_threshold():
    sim = _make_sim()
    sim.add(Zeeman((0.0, 0.0, 1.0e5)))
    sim.relax(stopping_dmdt=1.0)
    assert sim.t > 0.0
    # aligned nearly with the strong applied field
    assert sim.m_average[2] > 0.99



# --------------------------------------------------------------------------
# hysteresis() / hysteresis_loop(): API contract (NEW coverage inspired by,
# but not a literal transcription of, legacy hysteresis_test.py -- the
# literal master transcription lives above the banner as test_hysteresis /
# test_hysteresis_loop_and_plotting)
# --------------------------------------------------------------------------

def test_hysteresis_empty_list_returns_none():
    sim = _make_sim()
    assert sim.hysteresis([]) is None


def test_hysteresis_removes_temporary_zeeman_interaction():
    sim = _make_sim()
    assert not sim.has_interaction("Zeeman")
    sim.hysteresis([(1.0e4, 0.0, 0.0), (-1.0e4, 0.0, 0.0)],
                   stopping_dmdt=10.0)
    assert not sim.has_interaction("Zeeman")


def test_hysteresis_fun_return_value_accumulates_one_per_stage():
    sim = _make_sim()
    H_ext_list = [(1.0e4, 0.0, 0.0), (2.0e4, 0.0, 0.0), (3.0e4, 0.0, 0.0)]
    res = sim.hysteresis(
        H_ext_list, fun=lambda sim: sim.m_average[0], stopping_dmdt=10.0)
    assert len(res) == len(H_ext_list)


def test_hysteresis_loop_return_shapes():
    sim = _make_sim(K1=None)
    N = 3
    H_vals, m_vals = sim.hysteresis_loop(
        H_max=2.0e5, direction=(1, 0, 0), N=N, stopping_dmdt=10.0)
    assert len(H_vals) == 2 * N
    assert len(m_vals) == 2 * N


def test_hysteresis_loop_legacy_trivial_invariant():
    """DEGENERACY PIN (acceptance register D4, diagnosis corrected 2026-07-23).
    Transcribed from legacy ``hysteresis_test.py::
    test_hysteresis_loop_and_plotting``: a Zeeman-only (no anisotropy), no-
    exchange, no-demag single-cell system with initial direction close to
    the x-axis stays "trivially" aligned near +1 across a whole H-reversing
    loop (the legacy test's own docstring: "Check that the magnetisation
    values are as trivial as we expect them to be ;-)").

    The magnetisation stays near +1 NOT because ``relax()`` is skipped after
    the first stage but because this on-axis geometry is degenerate: with the
    field very nearly (anti)parallel to m and no anisotropy to define a
    transverse easy plane, m sits at an unstable Stoner-Wohlfarth saddle where
    the LLG torque ``m x (m x H) ~ 0``, so NO relaxation method -- fresh or
    shared -- switches it. This matches legacy because the physics is
    identical (faithfully preserved from ``ba928093``), not because a re-relax
    defect was preserved; ``test_hysteresis_switches_each_stage_tilted_axis``
    breaks the degeneracy with a 10 deg tilt and shows the same shared path
    switching. The numeric assertion is unchanged."""
    sim = _make_sim(m_init=(0.8, 0.2, 0.0), K1=None, Ms=1.0e6, nx=1)
    H = 0.2e6
    N = 5
    H_vals, m_vals = sim.hysteresis_loop(
        H_max=H, direction=(1.0, 0.01, 0.0), N=N, stopping_dmdt=10.0)
    assert np.allclose(m_vals, [1.0 for _ in range(2 * N)], atol=1e-4)


# --------------------------------------------------------------------------
# oracle fixture: the exact tiny 4-stage loop pinned above (K1 present, field
# reversal past the coercive field, native Sundials legacy backend)
# --------------------------------------------------------------------------

def test_oracle_hysteresis_loop_matches_legacy():
    """DEGENERACY PIN (acceptance register D4, diagnosis corrected 2026-07-23).
    Pins the exact tiny on-axis 4-stage loop (uniaxial K1, easy axis exactly
    along the x fields, a reversal from +1e5 to -1e5 A/m past the on-axis
    coercive field ``2 K1 / (mu0 Ms) ~ 1.85e4 A/m``) against the frozen
    legacy oracle (native Sundials backend).

    Stages 1-3 each advance exactly ``1.0e-14 s``, take a single integration
    step, and leave m_x ~ 0.99999998 -- the sample never switches even though
    the field reverses well past coercivity. This is CORRECT physics for a
    DEGENERATE geometry, NOT a re-relaxation defect: the field is exactly
    antiparallel to the easy axis, so m sits at the unstable Stoner-Wohlfarth
    saddle where the torque ``m x (m x H) ~ 0``; a fresh ``relax()`` from the
    same state also stays at +x, and ``relax()`` provably resets its state
    every stage (``sim_relax.py``). The result matches legacy ``ba928093``
    because the saddle physics is identical, not because a defect was
    preserved -- the corrected-vs-legacy divergence here is exactly zero.
    Non-degenerate switching (which the shared ``hysteresis()`` path performs
    correctly) is witnessed by
    ``test_hysteresis_switches_each_stage_tilted_axis`` and
    ``test_stoner_wohlfarth_like_loop_witness``. The stage-time and m_average
    assertions below are unchanged (bit-exact ``t``, m_average to oracle
    tolerance); only this rationale is corrected."""
    Ms = ORACLE["physical_parameters"]["Ms"]["value"]
    unit_length = ORACLE["physical_parameters"]["unit_length"]["value"]
    K1 = ORACLE["physical_parameters"]["K1"]["value"]
    alpha = ORACLE["physical_parameters"]["alpha"]["value"]
    m_init = tuple(ORACLE["physical_parameters"]["m_init"]["value"])
    stopping_dmdt = ORACLE["physical_parameters"]["stopping_dmdt"]["value"]
    H_vals = [tuple(h) for h in ORACLE["H_vals"]]

    sim = _make_sim(m_init=m_init, alpha=alpha, K1=K1, Ms=Ms,
                    unit_length=unit_length, name="oracle_hyst")

    stages = []

    def fun(sim):
        stages.append((sim.t, np.array(sim.m_average)))
        return sim.m_average

    sim.hysteresis(H_vals, fun=fun, stopping_dmdt=stopping_dmdt)

    assert len(stages) == len(ORACLE["stages"])
    for (t, m), ref in zip(stages, ORACLE["stages"]):
        tol = ref["tolerances"]
        # The stage timeline is governed by pure-Python scheduler logic
        # independent of the ODE backend, so it is expected bit-for-bit.
        assert t == ref["t"]
        np.testing.assert_allclose(
            m, ref["m_average"], atol=tol["absolute"], rtol=tol["relative"])


# --------------------------------------------------------------------------
# Stoner-Wohlfarth-like qualitative switching + loop-closure witness, using
# fresh Simulation/relax() calls per field step. NOTE: this fresh-per-stage
# construction is not needed to work around any re-relax defect (there is
# none -- see the module docstring); the shared hysteresis() path gives the
# identical loop, asserted directly in
# test_stoner_wohlfarth_loop_shared_hysteresis_path below.
# --------------------------------------------------------------------------

def test_stoner_wohlfarth_like_loop_witness():
    theta = np.deg2rad(20.0)
    H_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    H_max = 3.0e4
    N = 6
    H_norms = (list(np.linspace(H_max, -H_max, N))
               + list(np.linspace(-H_max, H_max, N)))

    def relax_fresh(m_start, H):
        sim = _make_sim(m_init=tuple(m_start), name="sw_stage")
        sim.add(Zeeman(tuple(H)))
        sim.relax(stopping_dmdt=1.0)
        return sim.m_average

    m = np.array([1.0, 0.05, 0.0])
    m_x_vals = []
    for h in H_norms:
        m = relax_fresh(m, h * H_dir)
        m_x_vals.append(float(m[0]))

    m_x_vals = np.array(m_x_vals)
    signs = np.sign(m_x_vals)

    # Genuine switching: both signs of m_x occur across the loop.
    assert (signs > 0).any() and (signs < 0).any()
    # Sign tracks the sign of the (dominant, x-projected) applied field once
    # deep enough past the coercive field (large |H|).
    assert m_x_vals[0] > 0.9  # started at +H_max, aligned with +x
    assert m_x_vals[N - 1] < -0.9  # swept down through -H_max, flipped
    assert m_x_vals[-1] > 0.9  # swept back up to +H_max: loop closes
    # Loop closure: first and last point (both at +H_max) agree closely.
    assert abs(m_x_vals[0] - m_x_vals[-1]) < 0.05


# --------------------------------------------------------------------------
# P2.5 switching-physics witnesses through the SHARED hysteresis() scheduler
# path (register D4 diagnosis correction, 2026-07-23). These assert PHYSICAL
# switching invariants, not stage counts, and demonstrate that the shared
# path already re-relaxes each stage to its own equilibrium.
# --------------------------------------------------------------------------

def _inst_dmdt_max(sim):
    """Instantaneous max nodal |dm/dt| (rad/s) recomputed at the settled m."""
    sim.llg.solve(sim.t)
    d = np.asarray(sim.dmdt).reshape((3, -1))
    return float(np.max(np.sqrt(np.sum(d ** 2, axis=0))))


# relax()'s stopping threshold: stopping_dmdt * ONE_DEGREE_PER_NS.
_ONE_DEGREE_PER_NS = 17453292.5


def test_hysteresis_switches_each_stage_tilted_axis():
    """Each applied-field stage independently re-relaxes to ITS OWN
    equilibrium, and the sample SWITCHES -- driven entirely through the shared
    ``sim.hysteresis()`` scheduler/clock path (NOT fresh per-stage sims).

    Geometry breaks the on-axis Stoner-Wohlfarth saddle degeneracy: the
    uniaxial easy axis is tilted 10 deg off the x field axis, so each stage
    has a well-defined, distinct equilibrium. Fields step down through the
    coercive field and reverse: [+1e5, +33333, -33333, -1e5] A/m on x. This is
    the direct evidence that ``hysteresis()`` re-relaxes every stage -- the
    behaviour the D4 register row wrongly claimed was defective. On the
    shipping (unmodified) code this test PASSES; it is a switching-physics
    witness, not a fixed defect."""
    phi = np.deg2rad(10.0)
    easy_tilt = (np.cos(phi), np.sin(phi), 0.0)
    sim = _make_sim(easy_axis=easy_tilt, name="tilt_switch")

    stages = []

    def fun(sim):
        stages.append((float(sim.m_average[0]), _inst_dmdt_max(sim)))
        return sim.m_average

    sim.hysteresis(
        [(1.0e5, 0.0, 0.0), (33333.0, 0.0, 0.0),
         (-33333.0, 0.0, 0.0), (-1.0e5, 0.0, 0.0)],
        fun=fun, stopping_dmdt=1.0)

    m_x = np.array([s[0] for s in stages])
    dmdt = np.array([s[1] for s in stages])

    # (i) every stage really is at its own equilibrium (re-relaxed): the
    # instantaneous torque at the settled state is at/below relax()'s
    # stopping threshold. If a stage had been skipped, a stage whose field
    # reversed would sit far above threshold.
    assert np.all(dmdt <= _ONE_DEGREE_PER_NS), dmdt

    # (ii) switching PHYSICS, not stage count: m_x follows the field sign,
    # flipping from +x to -x when the field reverses past coercivity.
    assert m_x[0] > 0.99 and m_x[1] > 0.99   # aligned with +x field
    assert m_x[2] < -0.99 and m_x[3] < -0.99  # switched to -x on reversal
    # measured (shared path): [0.9996, 0.9980, -0.9980, -0.9996]
    np.testing.assert_allclose(m_x, [0.9996, 0.9980, -0.9980, -0.9996],
                               atol=2e-3)


def test_stoner_wohlfarth_loop_shared_hysteresis_path():
    """Full Stoner-Wohlfarth loop swept along a 20 deg off-axis direction,
    driven through the SHARED ``sim.hysteresis()`` path, asserting the
    switching SEQUENCE and loop closure. This is the shared-path twin of
    ``test_stoner_wohlfarth_like_loop_witness`` (which uses fresh sims): the
    two produce the same loop, proving the shared scheduler path is not
    degraded relative to fresh per-stage relaxation."""
    theta = np.deg2rad(20.0)
    H_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    H_max = 3.0e4
    N = 6
    H_norms = (list(np.linspace(H_max, -H_max, N))
               + list(np.linspace(-H_max, H_max, N)))
    H_list = [tuple(h * H_dir) for h in H_norms]

    sim = _make_sim(name="sw_shared")
    m_x = np.array(sim.hysteresis(
        H_list, fun=lambda s: float(s.m_average[0]), stopping_dmdt=1.0))

    signs = np.sign(m_x)
    # switching SEQUENCE (measured): the four large-|H| points in each sweep
    # direction carry a definite sign; the loop is hysteretic (m_x lags H).
    assert signs.tolist() == [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]
    assert m_x[0] > 0.9        # +H_max: aligned +x
    assert m_x[N - 1] < -0.9   # swept to -H_max: flipped
    assert m_x[-1] > 0.9       # swept back to +H_max
    # loop closure: first and last (both at +H_max) agree closely.
    assert abs(m_x[0] - m_x[-1]) < 0.05
