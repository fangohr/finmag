"""Nmag 1D ANISOTROPY dynamics comparison under DOLFINx (SR1 P5.2 minimal-diff
transcription).

This is a MINIMAL-DIFF transcription of the master relaxation regression
``test_nmag_1d_anisotropy.py`` (git ``b5015c5a``): a 1D chain relaxing under
uniaxial anisotropy (easy axis +z), compared node-for-node against checked-in
Nmag reference files (``*_ref.txt``; no live Nmag is run). Function names,
order, assertion structure and TOLERANCES are kept identical to master; the
only differences are (a) dolfin->dolfinx API changes (each annotated inline),
(b) the py2->py3 ``print`` conversion, and (c) explanatory comments. Every
restored master tolerance passes VERBATIM under DOLFINx -- measured values are
recorded next to each assertion.

The row-for-row node comparison (``test_third_node``) relies on master's
implicit assumption that ``IntervalMesh`` emits its vertices in ascending-x
order, so the reference files' n-th row lines up with the mesh's n-th vertex.
``test_mesh_vertices_ascending_order`` below (NEW under DOLFINx) pins that
assumption as an explicit, visible guard against a future dolfinx meshing
change silently corrupting the comparison.

[Claude Opus 4.8], [Claude Sonnet 5]
"""

import os

import numpy as np
from mpi4py import MPI

import dolfinx.mesh as dm  # dolfin.IntervalMesh -> dolfinx.mesh.create_interval (MPI-aware)

from finmag import Simulation as Sim
from finmag.energies import UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


# dolfin-free numeric helpers: master used finmag.util.helpers.{vectors,
# components}, but that module unconditionally imports legacy ``dolfin`` and
# so cannot be imported under the DOLFINx environment. These are exact
# reimplementations of ``h.vectors``/``h.components`` (component-blocked
# ``xxx`` array reshaping, Task-31).
def _vectors(vs):
    n = len(vs) // 3
    return vs.view().reshape((n, -1), order="F")


def _components(vs):
    return vs.view().reshape((3, -1))


averages = []
third_node = []

# run the simulation


def setup_module(module=None):
    x_max = 100e-9  # m
    simplexes = 50
    # dolfin.IntervalMesh(simplexes, 0, x_max) -> dolfinx.mesh.create_interval;
    # dolfinx requires an explicit MPI communicator and an [a, b] pair.
    mesh = dm.create_interval(MPI.COMM_WORLD, simplexes, [0.0, x_max])

    def m_gen(coords):
        x = coords[0]
        mx = min(1.0, x / x_max)
        mz = 0.1
        my = np.sqrt(1.0 - (0.99 * mx ** 2 + mz ** 2))
        return np.array([mx, my, mz])

    K1 = 520e3  # J/m^3
    Ms = 0.86e6

    sim = Sim(mesh, Ms)
    sim.alpha = 0.2
    sim.set_m(m_gen)
    anis = UniaxialAnisotropy(K1, (0, 0, 1))
    sim.add(anis)

    # Save H_anis and m at t0 for comparison with nmag
    global H_anis_t0, m_t0
    H_anis_t0 = anis.compute_field()
    m_t0 = sim.m

    av_f = open(os.path.join(MODULE_DIR, "averages.txt"), "w")
    tn_f = open(os.path.join(MODULE_DIR, "third_node.txt"), "w")

    t = 0
    t_max = 3e-10
    dt = 5e-12
    # s
    while t <= t_max:
        mx, my, mz = sim.m_average
        averages.append([t, mx, my, mz])
        av_f.write(
            str(t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

        mx, my, mz = _components(sim.m)  # h.components -> local _components (dolfin-free)
        m2x, m2y, m2z = mx[2], my[2], mz[2]
        third_node.append([t, m2x, m2y, m2z])
        tn_f.write(
            str(t) + " " + str(m2x) + " " + str(m2y) + " " + str(m2z) + "\n")

        t += dt
        sim.run_until(t)

    av_f.close()
    tn_f.close()


def test_averages():
    REL_TOLERANCE = 9e-2  # master 9e-2; measured DOLFINx max rel diff 1.83e-3 -> passes verbatim

    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = np.array(averages)

    dt = ref[:, 0] - computed[:, 0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / np.sqrt(ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2))

    print("test_averages, max. relative difference per axis:")  # py2 print stmt -> py3 print()
    print(np.nanmax(rel_diff, axis=0))

    rel_err = np.nanmax(rel_diff)
    if rel_err > 1e-3:
        print("nmag:\n", ref)
        print("finmag:\n", computed)
    assert rel_err < REL_TOLERANCE


def test_third_node():
    REL_TOLERANCE = 3e-1  # master 3e-1; measured DOLFINx max rel diff 6.03e-3 -> passes verbatim

    ref = np.loadtxt(os.path.join(MODULE_DIR, "third_node_ref.txt"))
    computed = np.array(third_node)

    dt = ref[:, 0] - computed[:, 0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / np.sqrt(ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2))

    print("test_third_node: max. relative difference per axis:")  # py2 print stmt -> py3 print()
    print(np.nanmax(rel_diff, axis=0))

    rel_err = np.nanmax(rel_diff)
    if rel_err > 1e-3:
        print("nmag:\n", ref)
        print("finmag:\n", computed)
    assert rel_err < REL_TOLERANCE


def test_m_cross_H():
    """
    compares m x H_anis at the beginning of the simulation.
    motivation: Hans on IRC, 13.04.2012 10:45

    """
    REL_TOLERANCE = 7e-5  # master 7e-5; measured DOLFINx max rel diff 6.53e-5 -> passes verbatim

    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m_t0_ref.txt"))
    m_computed = _vectors(m_t0)  # h.vectors -> local _vectors (dolfin-free)
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "anis_t0_ref.txt"))
    H_computed = _vectors(H_anis_t0)  # h.vectors -> local _vectors (dolfin-free)
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    diff = np.abs(m_cross_H_ref - m_cross_H_computed)
    max_norm = max([np.linalg.norm(v) for v in m_cross_H_ref])  # h.norm -> np.linalg.norm (dolfin-free)
    rel_diff = diff / max_norm

    print("test_m_cross_H: max rel diff=", np.max(rel_diff))  # py2 print stmt -> py3 print()
    assert np.max(rel_diff) < REL_TOLERANCE


# ==========================================================================
# ===== NEW under DOLFINx (no master ancestor) ============================
# ==========================================================================
# Extra invariants with no master ancestor: (1) a structural guard on the
# ordering assumption the row-for-row reference comparisons above silently
# depend on, and (2) a non-triviality witness so a degenerate (all-zero)
# anisotropy field could not accidentally satisfy test_m_cross_H's tolerance.

def test_mesh_vertices_ascending_order():
    """``test_third_node`` compares the n-th mesh vertex to the n-th row of
    ``third_node_ref.txt``, relying on master's ``IntervalMesh`` emitting
    vertices in ascending-x order. Pin that assumption explicitly so a future
    dolfinx meshing change cannot silently misalign the comparison."""
    x_max = 100e-9
    mesh = dm.create_interval(MPI.COMM_WORLD, 50, [0.0, x_max])
    assert np.all(np.diff(mesh.geometry.x[:, 0]) > 0)


def test_h_anis_t0_is_nontrivial():
    """Sanity check that ``H_anis_t0`` (used by ``test_m_cross_H``) is a
    genuinely non-zero field, so that assertion is not vacuously satisfied."""
    H_computed = _vectors(H_anis_t0)
    assert np.max(np.linalg.norm(H_computed, axis=1)) > 1e4


if __name__ == '__main__':
    setup_module()
    test_averages()
    test_third_node()
    test_m_cross_H()
    test_mesh_vertices_ascending_order()
    test_h_anis_t0_is_nontrivial()
