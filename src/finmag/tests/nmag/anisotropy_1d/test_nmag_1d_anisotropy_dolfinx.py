"""Nmag 1D ANISOTROPY dynamics comparison under DOLFINx (invariant witnesses).

Restores the legacy ``test_nmag_1d_anisotropy.py`` relaxation regression -- a 1D
chain relaxing under uniaxial anisotropy (easy axis +z) -- rebuilt for DOLFINx.

The mesh is a structured ``dolfinx.mesh.create_interval(50, [0, 100nm])`` whose 51
vertices are emitted in DETERMINISTIC ascending-x order, so the reference files
(sampled node-for-node by Nmag) line up row-for-row and the third-node trajectory
tracks the same physical site. No live Nmag is run (register M1): only the
checked-in ``*_ref.txt`` files are read.

All assertions are physical, mesh-order-robust invariants:
  * ``test_m_cross_H`` -- the physical invariant ``m x H_anis`` at t0 matches the
    Nmag reference. This is the sharpest witness: it directly probes the
    anisotropy field ``H = (2 K1 / (mu0 Ms^2)) (m.u) u`` against Nmag, so a wrong
    K1, a wrong easy axis, or a component-ordering bug blows it up.
  * ``test_averages``  -- the volume-average magnetisation trajectory ``<m>(t)``
    matches Nmag over the whole 0..3e-10 s run.
  * ``test_third_node``-- the third node's ``m(t)`` trajectory matches Nmag.

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
from finmag.energies import UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

X_MAX = 100e-9
SIMPLEXES = 50
K1 = 520e3  # J/m^3
MS = 0.86e6
ALPHA = 0.2

T_MAX = 3e-10
DT = 5e-12


def _vectors(vs):
    n = len(vs) // 3
    return vs.view().reshape((n, -1), order="F")


def _components(vs):
    return vs.view().reshape((3, -1))


def _m_gen(coords):
    x = coords[0]
    mx = min(1.0, x / X_MAX)
    mz = 0.1
    my = np.sqrt(1.0 - (0.99 * mx ** 2 + mz ** 2))
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
    anis = UniaxialAnisotropy(K1, (0, 0, 1))
    sim.add(anis)

    H_anis_t0 = anis.compute_field().copy()
    m_t0 = sim.m.copy()

    t = 0.0
    averages = []
    third_node = []
    while t <= T_MAX:
        mx, my, mz = sim.m_average
        averages.append([t, mx, my, mz])
        cx, cy, cz = _components(sim.m)
        third_node.append([t, cx[2], cy[2], cz[2]])
        t += DT
        sim.run_until(t)

    _STATE.update(
        m_t0=m_t0,
        H_anis_t0=H_anis_t0,
        averages=np.array(averages),
        third_node=np.array(third_node),
    )
    return _STATE


def test_m_cross_H():
    """``m x H_anis`` at t0 vs Nmag reference (physical invariant)."""
    # Measured max rel diff under DOLFINx: 6.53e-5. Legacy pinned 7e-5. This is
    # the inherent finmag-vs-Nmag anisotropy-field discretisation difference
    # (node order matches, so it is the same quantity legacy compared). Pinned
    # at the measured value with modest headroom.
    REL_TOLERANCE = 1e-4

    st = _run()
    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m_t0_ref.txt"))
    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "anis_t0_ref.txt"))
    m_comp = _vectors(st["m_t0"])
    H_comp = _vectors(st["H_anis_t0"])
    assert m_ref.shape == m_comp.shape == (51, 3)
    assert H_ref.shape == H_comp.shape == (51, 3)

    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_comp = np.cross(m_comp, H_comp)
    diff = np.abs(m_cross_H_ref - m_cross_H_comp)
    scale = max(np.linalg.norm(v) for v in m_cross_H_ref)
    max_rel = float(np.max(diff / scale))
    print("test_m_cross_H: max rel diff:", max_rel)

    # Non-trivial witness: computed anisotropy field is genuinely order 1e5 A/m.
    assert np.max(np.linalg.norm(H_comp, axis=1)) > 1e4
    assert max_rel < REL_TOLERANCE


def test_averages():
    # Measured max rel diff vs Nmag: 1.83e-3. Legacy pinned 9e-2.
    REL_TOLERANCE = 1e-2

    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = _run()["averages"]
    assert ref.shape == computed.shape

    assert np.max(np.abs(ref[:, 0] - computed[:, 0])) < 1e-15, "timesteps"
    ref_v, comp_v = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref_v - comp_v
    rel_diff = np.abs(diff / np.sqrt(ref_v[0] ** 2 + ref_v[1] ** 2
                                     + ref_v[2] ** 2))
    max_rel = float(np.nanmax(rel_diff))
    print("test_averages: max rel diff per axis:", np.nanmax(rel_diff, axis=0))
    assert max_rel < REL_TOLERANCE


def test_third_node():
    # Measured max rel diff vs Nmag: 6.03e-3. Legacy pinned 3e-1.
    REL_TOLERANCE = 3e-2

    ref = np.loadtxt(os.path.join(MODULE_DIR, "third_node_ref.txt"))
    computed = _run()["third_node"]
    assert ref.shape == computed.shape

    assert np.max(np.abs(ref[:, 0] - computed[:, 0])) < 1e-15, "timesteps"
    ref_v, comp_v = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref_v - comp_v
    rel_diff = np.abs(diff / np.sqrt(ref_v[0] ** 2 + ref_v[1] ** 2
                                     + ref_v[2] ** 2))
    max_rel = float(np.nanmax(rel_diff))
    print("test_third_node: max rel diff per axis:", np.nanmax(rel_diff, axis=0))
    assert max_rel < REL_TOLERANCE


if __name__ == "__main__":
    test_m_cross_H()
    test_averages()
    test_third_node()
