"""Nmag 1D EXCHANGE dynamics comparison under DOLFINx (invariant witnesses).

Restores the legacy ``test_exchange_1d.py`` relaxation regression -- a 1D chain
relaxing under exchange with both endpoints pinned -- rebuilt for DOLFINx.

The mesh is a structured ``dolfinx.mesh.create_interval(10, [0, 20nm])`` whose 11
vertices are emitted in DETERMINISTIC ascending-x order, so ``pins=[0, 10]`` pins
the two physical endpoints and the reference files (sampled node-for-node by
Nmag) line up row-for-row. No live Nmag is run (register M1): only the checked-in
``*_ref.txt`` files are read.

All assertions are physical, mesh-order-robust invariants:
  * ``test_angles``    -- the relaxed state has a UNIFORM inter-node angle equal
    to pi/10 (10 equal steps between the two pinned endpoints anti-aligned along
    x). A wrong exchange stiffness, a broken pin, or a non-relaxed state would
    make the angles non-uniform or shift the mean away from pi/10.
  * ``test_averages``  -- the volume-average magnetisation trajectory ``<m>(t)``
    matches Nmag over the whole 0..5e-10 s run.
  * ``test_third_node``-- the third node's ``m(t)`` trajectory matches Nmag.
  * ``test_m_cross_H`` -- the physical invariant ``m x H_exchange`` at t0 matches
    the Nmag reference (immune to a global rotation / component reordering).

Every tolerance is the empirically MEASURED value under DOLFINx plus modest
headroom (documented inline), not the legacy constant. Field ordering is
component-blocked ``xxx`` (Task-31); ``_vectors``/``_components`` reshape it.
[Claude Opus 4.8]
"""

import os

import numpy as np
import pytest

import dolfinx.mesh as dm
from mpi4py import MPI

from finmag import Simulation as Sim
from finmag.energies import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

X_MAX = 20e-9
SIMPLEXES = 10
MS = 0.86e6
A = 1.3e-11
ALPHA = 0.2

# Dynamics sampling schedule (legacy: t in [0, 5e-10] step 1e-11 -> 50 samples).
T1 = 5e-10
DT = 1e-11


# --- dolfin-free numeric helpers (finmag.util.helpers imports dolfin) ---------
def _vectors(vs):
    n = len(vs) // 3
    return vs.view().reshape((n, -1), order="F")


def _components(vs):
    return vs.view().reshape((3, -1))


def _angle(a, b):
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _m_gen(coords):
    x = coords[0]
    mx = min(1.0, 2.0 * x / X_MAX - 1.0)
    my = np.sqrt(1.0 - mx ** 2)
    mz = 0.0
    return np.array([mx, my, mz])


_STATE = {}


def _run():
    if _STATE:
        return _STATE

    mesh = dm.create_interval(MPI.COMM_WORLD, SIMPLEXES, [0.0, X_MAX])
    assert np.all(np.diff(mesh.geometry.x[:, 0]) > 0), "vertices not ascending"

    sim = Sim(mesh, MS, unit_length=1)
    sim.alpha = ALPHA
    sim.set_m(_m_gen)
    sim.pins = [0, 10]
    ex = Exchange(A)
    sim.add(ex)

    # endpoints really are the pinned sites on this structured mesh
    assert sim.llg.pins.tolist() == [0, 10]

    H_exc_t0 = ex.compute_field().copy()
    m_t0 = sim.m.copy()

    t = 0.0
    averages = []
    third_node = []
    while t <= T1:
        mx, my, mz = sim.m_average
        averages.append([t, mx, my, mz])
        cx, cy, cz = _components(sim.m)
        third_node.append([t, cx[2], cy[2], cz[2]])
        t += DT
        sim.run_until(t)

    _STATE.update(
        m_t0=m_t0,
        H_exc_t0=H_exc_t0,
        m_final=sim.m.copy(),
        averages=np.array(averages),
        third_node=np.array(third_node),
    )
    return _STATE


def test_angles():
    # Measured under DOLFINx: max(angle)-min(angle) = 4.19e-8,
    # |mean - pi/10| = 1.89e-14. Pinned with headroom.
    MAXDIFF_TOL = 1e-7
    MEAN_TOL = 1e-12

    m = _vectors(_run()["m_final"])
    angles = np.array([_angle(m[i], m[i + 1]) for i in range(len(m) - 1)])
    max_diff = abs(angles.max() - angles.min())
    mean_angle = float(np.mean(angles))
    print("test_angles: max_diff =", max_diff, " mean =", mean_angle,
          " |mean - pi/10| =", abs(mean_angle - np.pi / 10))

    assert max_diff < MAXDIFF_TOL
    assert abs(mean_angle - np.pi / 10) < MEAN_TOL


def test_averages():
    # Measured max abs diff vs Nmag: 7.14e-5. Legacy pinned 2e-3 (absolute).
    TOLERANCE = 5e-4

    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = _run()["averages"]
    assert ref.shape == computed.shape

    assert np.max(np.abs(ref[:, 0] - computed[:, 0])) < 1e-15, "timesteps"
    ref_v, comp_v = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    max_diff = float(np.nanmax(np.abs(ref_v - comp_v)))
    print("test_averages: max abs diff per axis:",
          np.nanmax(np.abs(ref_v - comp_v), axis=0))
    assert max_diff < TOLERANCE


def test_third_node():
    # Measured max rel diff (x, y axes) vs Nmag: 9.99e-5, 2.31e-4. The z axis is
    # excluded exactly as legacy: the Nmag reference z is ~0 there, making a
    # relative diff meaningless. Legacy pinned 6e-3.
    REL_TOLERANCE = 1e-3

    ref = np.loadtxt(os.path.join(MODULE_DIR, "third_node_ref.txt"))
    computed = _run()["third_node"]
    assert ref.shape == computed.shape

    assert np.max(np.abs(ref[:, 0] - computed[:, 0])) < 1e-15, "timesteps"
    ref_v, comp_v = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = np.abs(ref_v - comp_v)
    rel_diff = np.abs(np.divide(
        diff, ref_v, out=np.full_like(diff, np.nan), where=(ref_v != 0)))
    max_diffs = np.nanmax(rel_diff, axis=0)
    print("test_third_node: max rel diff per axis:", max_diffs)
    assert max_diffs[0] < REL_TOLERANCE and max_diffs[1] < REL_TOLERANCE


def test_m_cross_H():
    """``m x H_exchange`` at t0 vs Nmag reference (physical invariant)."""
    # Measured max rel diff under DOLFINx: 2.81e-8. Legacy pinned 8e-8.
    REL_TOLERANCE = 1e-7

    st = _run()
    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m_t0_ref.txt"))
    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "exc_t0_ref.txt"))
    m_comp = _vectors(st["m_t0"])
    H_comp = _vectors(st["H_exc_t0"])
    assert m_ref.shape == m_comp.shape == (11, 3)
    assert H_ref.shape == H_comp.shape == (11, 3)

    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_comp = np.cross(m_comp, H_comp)
    diff = np.abs(m_cross_H_ref - m_cross_H_comp)
    scale = max(np.linalg.norm(v) for v in m_cross_H_ref)
    rel_diff = diff / scale
    max_rel = float(np.max(rel_diff))
    print("test_m_cross_H: max rel diff:", max_rel)

    # Non-trivial witness: computed exchange field is genuinely order MA/m.
    assert np.max(np.linalg.norm(H_comp, axis=1)) > 1e6
    assert max_rel < REL_TOLERANCE


if __name__ == "__main__":
    test_angles()
    test_averages()
    test_third_node()
    test_m_cross_H()
