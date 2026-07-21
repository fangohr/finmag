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

Important finding (see ``transition-notes.org`` Task 15 and
``gen_hysteresis_oracle.py``'s module docstring): running the SAME scenario
against the frozen legacy oracle (native Sundials backend, the *true*
legacy default) reveals that ``hysteresis()``/``hysteresis_loop()`` do *not*
actually achieve an independent re-relaxation at every stage after the
first one -- the magnetisation barely moves in stage 2 onward even for a
field reversal well past the Stoner-Wohlfarth coercive field, because each
subsequent ``relax()`` call's very first scheduled tick coincides with the
integrator's already-advanced clock (``t == integrator.cur_t``), and
``Scheduler.run()`` explicitly skips integrating in that case. The legacy
test suite itself documents this exact symptom with a wink (see
``src/finmag/sim/hysteresis_test.py::test_hysteresis_loop_and_plotting``:
"Check that the magnetisation values are as trivial as we expect them to be
;-)"). This DOLFINx port reproduces that (surprising, but genuinely legacy,
oracle-confirmed) behavior verbatim rather than "fixing" it -- see
``test_oracle_hysteresis_loop_matches_legacy`` and
``test_hysteresis_loop_legacy_trivial_invariant`` below.

The Stoner-Wohlfarth-like switching *capability* itself -- the reason
``relax()``/``hysteresis()`` exist at all -- is verified separately, using
fresh ``Simulation``/``relax()`` calls per field step (avoiding the above
limitation), in ``test_stoner_wohlfarth_like_loop_witness`` below.

[Claude Sonnet 5]
"""

import json
import os

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from finmag.energies import UniaxialAnisotropy, Zeeman
from finmag.sim.sim import Simulation

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "hysteresis_oracle.json")
ORACLE = json.load(open(_FIXTURE))


def _box(nx=1):
    return mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [nx, nx, nx],
        mesh.CellType.tetrahedron)


def _make_sim(m_init=(1.0, 0.05, 0.0), alpha=1.0, K1=1.0e4,
              easy_axis=(1.0, 0.0, 0.0), Ms=8.6e5, unit_length=1e-9, nx=1,
              name="hyst_sim"):
    sim = Simulation(_box(nx), Ms, unit_length=unit_length, name=name)
    sim.set_m(m_init)
    sim.alpha = alpha
    if K1 is not None:
        sim.add(UniaxialAnisotropy(K1, easy_axis))
    return sim


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
# hysteresis() / hysteresis_loop(): API contract (transcribed from legacy
# hysteresis_test.py)
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
    """Transcribed from legacy ``hysteresis_test.py::
    test_hysteresis_loop_and_plotting``: a Zeeman-only (no anisotropy), no-
    exchange, no-demag single-cell system with initial direction close to
    the x-axis stays "trivially" aligned near +1 across a whole H-reversing
    loop (the legacy test's own docstring: "Check that the magnetisation
    values are as trivial as we expect them to be ;-)"), because of the
    stage-2-onward relax() limitation documented in this module's
    docstring."""
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
# fresh Simulation/relax() calls per field step (sidestepping the
# hysteresis()/relax() stage-2-onward limitation documented above)
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
