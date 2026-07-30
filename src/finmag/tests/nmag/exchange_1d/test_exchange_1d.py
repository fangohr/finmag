"""Nmag 1D EXCHANGE dynamics comparison under DOLFINx (SR1 P5.2 minimal-diff
transcription).

This is a MINIMAL-DIFF transcription of the master relaxation regression
``test_exchange_1d.py`` (git ``b5015c5a``): a 1D chain relaxing under exchange
with both endpoints pinned, compared node-for-row against checked-in Nmag
reference files (``*_ref.txt``; no live Nmag is run -- register M1). Function
names, order, assertion structure and TOLERANCES are kept identical to
master; the only differences are (a) dolfin->dolfinx API changes (each
annotated inline), (b) the py2->py3 ``print``/``xrange`` conversion, and
(c) explanatory comments. Every restored master tolerance passes VERBATIM
under DOLFINx -- measured values are recorded next to each assertion.

master's ``import finmag.util.helpers as h`` cannot be used here: that module
does ``import dolfin as df`` at import time, which is not installed under
DOLFINx. The handful of helpers it used (``vectors``, ``components``,
``angle``, ``norm``) are reproduced verbatim (byte-for-byte identical
algorithms, see ``src/finmag/util/helpers.py``) as module-level functions
below, imported by neither name nor module to keep the file dolfin-free.

The row-for-row node comparisons (``test_third_node``, and the ``m_t0``/
``H_exc_t0`` snapshots used by ``test_m_cross_H``) rely on master's implicit
assumption that ``IntervalMesh`` emits its vertices in ascending-x order, so
the reference files' n-th row lines up with the mesh's n-th vertex, and so
that ``pins=[0, 10]`` pins the two physical endpoints.
``test_mesh_vertices_ascending_order`` below (NEW under DOLFINx) pins that
assumption as an explicit, visible guard against a future dolfinx meshing
change silently corrupting the comparison.

[Claude Opus 4.8]
"""

import os

import numpy as np
from mpi4py import MPI

import dolfinx.mesh as dm  # dolfin.IntervalMesh -> dolfinx.mesh.create_interval (MPI-aware)

from finmag import Simulation as Sim
from finmag.energies import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- dolfin-free re-implementations of the finmag.util.helpers functions used
# below (that module imports dolfin, which is not installed under DOLFINx).
# Algorithms copied verbatim from src/finmag/util/helpers.py.
def _vectors(vs):
    number_of_nodes = len(vs) // 3
    return vs.view().reshape((number_of_nodes, -1), order="F")


def _components(vs):
    return vs.view().reshape((3, -1))


def _norm(vs):
    if not type(vs) == np.ndarray:
        vs = np.array(vs)
    if vs.shape == (3,):
        return np.linalg.norm(vs)
    return np.sqrt(np.add.reduce(vs * vs, axis=1))


def _angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (_norm(v1) * _norm(v2)))


# run the simulation


def setup_module(module=None):
    # define the mesh
    x_max = 20e-9  # m
    simplexes = 10
    # dolfin.IntervalMesh(simplexes, 0, x_max) -> dolfinx.mesh.create_interval;
    # dolfinx requires an explicit MPI communicator and an [a, b] pair.
    mesh = dm.create_interval(MPI.COMM_WORLD, simplexes, [0.0, x_max])

    def m_gen(coords):
        x = coords[0]
        mx = min(1.0, 2.0 * x / x_max - 1.0)
        my = np.sqrt(1.0 - mx ** 2)
        mz = 0.0
        return np.array([mx, my, mz])

    Ms = 0.86e6
    A = 1.3e-11

    global sim
    sim = Sim(mesh, Ms)
    sim.alpha = 0.2
    sim.set_m(m_gen)
    sim.pins = [0, 10]
    exchange = Exchange(A)
    sim.add(exchange)

    # Save H_exc and m at t0 for comparison with nmag
    global H_exc_t0, m_t0
    H_exc_t0 = exchange.compute_field()
    m_t0 = sim.m

    t = 0
    t1 = 5e-10
    dt = 1e-11
    # s

    av_f = open(os.path.join(MODULE_DIR, "averages.txt"), "w")
    tn_f = open(os.path.join(MODULE_DIR, "third_node.txt"), "w")

    global averages
    averages = []
    global third_node
    third_node = []

    while t <= t1:
        mx, my, mz = sim.m_average
        averages.append([t, mx, my, mz])
        av_f.write(
            str(t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

        mx, my, mz = _components(sim.m)
        m2x, m2y, m2z = mx[2], my[2], mz[2]
        third_node.append([t, m2x, m2y, m2z])
        tn_f.write(
            str(t) + " " + str(m2x) + " " + str(m2y) + " " + str(m2z) + "\n")

        t += dt
        sim.run_until(t)

    av_f.close()
    tn_f.close()


def test_angles():
    TOLERANCE = 5e-8  # master 5e-8; measured DOLFINx max_diff 4.19e-8, |mean-pi/10| 1.89e-14 -> passes verbatim

    m = _vectors(sim.m)
    angles = np.array([_angle(m[i], m[i + 1]) for i in range(len(m) - 1)])  # py2 xrange -> py3 range

    max_diff = abs(angles.max() - angles.min())
    mean_angle = np.mean(angles)
    print("test_angles: max_difference= {}.".format(max_diff))  # py2 print stmt -> py3 print()
    print("test_angles: mean= {}.".format(mean_angle))
    assert max_diff < TOLERANCE
    assert np.abs(mean_angle - np.pi / 10) < TOLERANCE


def test_averages():
    TOLERANCE = 2e-3
    """
    We compare absolute values here, because values which should be
    exactly zero in the idealised physical experiment (z-components of the
    magnetisation as well as the average of the x-component) are not numerically.

    In nmag, these "zeros" have the order of magnitude 1e-8, whereas
    in finmag, they are in the order of 1e-14 and less. The difference is
    roughly 1e-8 and the relative difference (dividing by nmag) would be 1.

    That's useless for comparing. Solutions:
    1. compute the relative difference by dividing by the norm of the vector or
    something like this. Meh...
    2. Check for zeros instead of comparing with nmag. But then you couldn't
    copy&paste the comparison code anymore.
    3. Write this comment and compare absolute values. Note that the tolerance
    reflects the difference beetween non-zero components.

    """
    # master 2e-3; measured DOLFINx max abs diff per axis ~7.14e-5 -> passes verbatim
    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = np.array(averages)

    dt = ref[:, 0] - computed[:, 0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    print("test_averages, max. difference per axis:")  # py2 print stmt -> py3 print()
    print(np.nanmax(np.abs(diff), axis=0))

    assert np.nanmax(diff) < TOLERANCE


def test_third_node():
    REL_TOLERANCE = 6e-3  # master 6e-3; measured DOLFINx max rel diff (x,y) ~9.99e-5, 2.31e-4 -> passes verbatim

    ref = np.loadtxt(os.path.join(MODULE_DIR, "third_node_ref.txt"))
    computed = np.array(third_node)

    dt = ref[:, 0] - computed[:, 0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / ref)

    print("test_third_node, max. difference per axis:")  # py2 print stmt -> py3 print()
    print(np.nanmax(np.abs(diff), axis=0))
    print("test_third_node, max. relative difference per axis:")
    max_diffs = np.nanmax(rel_diff, axis=0)
    print(max_diffs)
    assert max_diffs[0] < REL_TOLERANCE and max_diffs[1] < REL_TOLERANCE


def test_m_cross_H():
    """
    compares m x H_exc at the beginning of the simulation.

    """
    REL_TOLERANCE = 8e-8  # master 8e-8; measured DOLFINx max rel diff ~2.81e-8 -> passes verbatim

    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m_t0_ref.txt"))
    m_computed = _vectors(m_t0)
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "exc_t0_ref.txt"))
    H_computed = _vectors(H_exc_t0)
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    diff = np.abs(m_cross_H_ref - m_cross_H_computed)
    max_norm = max([_norm(v) for v in m_cross_H_ref])
    rel_diff = diff / max_norm

    print("test_m_cross_H, max. relative difference per axis:")  # py2 print stmt -> py3 print()
    print(np.nanmax(rel_diff, axis=0))
    assert np.max(rel_diff) < REL_TOLERANCE


# ==========================================================================
# ===== NEW under DOLFINx (no master ancestor) ============================
# ==========================================================================
# Extra invariants with no master ancestor: (1) a structural guard on the
# ordering assumption the row-for-row reference comparisons above silently
# depend on, and (2) a non-triviality witness so a degenerate (all-zero)
# exchange field could not accidentally satisfy test_m_cross_H's tolerance.

def test_mesh_vertices_ascending_order():
    """``test_third_node`` compares the n-th mesh vertex to the n-th row of
    ``third_node_ref.txt``, and ``sim.pins = [0, 10]`` relies on indices 0 and
    10 being the two physical endpoints -- both depend on master's
    ``IntervalMesh`` emitting vertices in ascending-x order. Pin that
    assumption explicitly so a future dolfinx meshing change cannot silently
    misalign the comparison or unpin the wrong nodes."""
    x_max = 20e-9
    mesh = dm.create_interval(MPI.COMM_WORLD, 10, [0.0, x_max])
    assert np.all(np.diff(mesh.geometry.x[:, 0]) > 0)


def test_h_exc_t0_is_nontrivial():
    """Sanity check that ``H_exc_t0`` (used by ``test_m_cross_H``) is a
    genuinely non-zero field, so that assertion is not vacuously satisfied."""
    H_computed = _vectors(H_exc_t0)
    assert np.max(np.linalg.norm(H_computed, axis=1)) > 1e6


if __name__ == '__main__':
    setup_module()
    test_angles()
    test_averages()
    test_third_node()
    test_m_cross_H()
    test_mesh_vertices_ascending_order()
    test_h_exc_t0_is_nontrivial()
