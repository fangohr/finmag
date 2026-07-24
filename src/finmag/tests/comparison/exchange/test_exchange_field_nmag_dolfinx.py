"""Nmag EXCHANGE-field comparison under DOLFINx (physical-invariant witness).

This restores the legacy ``test_exchange_field.py::test_against_nmag`` regression
against the checked-in Nmag reference exchange field (``m0_nmag.txt`` /
``H_exc_nmag.txt``, both ``11x3``), rebuilt for DOLFINx.

Why this is safe to compare row-for-row:
  * The magnetisation is set from the SAME analytic profile the legacy test used
    (``m_gen`` below), and it is compared against ``m0_nmag.txt`` -- the two agree
    to ~7e-16, confirming the analytic m0 *is* the Nmag reference m0.
  * A structured ``dolfinx.mesh.create_interval(10, [0, 20nm])`` emits its 11
    vertices in DETERMINISTIC ascending-x order (asserted here), which matches the
    row order of the Nmag reference files. So node-order pairing is legitimate on
    this structured 1D mesh -- unlike the unstructured 3D case (M8 mesh-drift).
  * No live Nmag (register M1): ``nsim`` is never run; only the checked-in
    reference text files are read.

The discriminating quantity is the physical invariant ``m x H_exchange`` (a
per-node cross product of the 3-vectors), exactly as legacy. A units error, a
component-ordering (Task-31 xxx-blocking) bug, or a broken assembler would blow
the relative difference up by many orders of magnitude.

Field ordering is component-blocked ``xxx = [x0..xn, y0..yn, z0..zn]`` (Task-31);
``_vectors`` reshapes that (order="F") back to per-node 3-vectors.
[Claude Opus 4.8]
"""

import os

import numpy as np
import pytest

import dolfinx.mesh as dm
import dolfinx.fem as fem
from mpi4py import MPI

from finmag.field import Field
from finmag.energies import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Legacy material/geometry parameters (test_exchange_field.py).
X0 = 0.0
X1 = 20e-9
XN = 10
MS = 0.86e6
A = 1.3e-11

# Measured max relative difference of ``m x H`` under DOLFINx: 1.41e-14 (see the
# printed diagnostic). Legacy pinned 2e-14 on identical node-for-node dolfin
# assembly; DOLFINx assembly reproduces it to the same order, so we pin the
# MEASURED value with modest headroom rather than copying the legacy constant.
REL_TOLERANCE = 5e-14


def _vectors(vs):
    """Component-blocked ``[x0..xn, y0..yn, z0..zn]`` -> ``(n, 3)`` per-node."""
    n = len(vs) // 3
    return vs.view().reshape((n, -1), order="F")


def _m_gen(r):
    """Legacy analytic initial magnetisation on coordinate columns ``r`` (1, n)."""
    x = np.maximum(np.minimum(r[0] / X1, 1.0), 0.0)
    mx = (2 * x - 1) * 2 / 3
    mz = np.sin(2 * np.pi * x) / 2
    my = np.sqrt(1.0 - mx ** 2 - mz ** 2)
    return np.array([mx, my, mz])


def _compute():
    mesh = dm.create_interval(MPI.COMM_WORLD, XN, [X0, X1])

    # DETERMINISTIC node order: the 11 vertices are ascending x, matching the
    # Nmag reference row order. Asserted so a future mesh-generator change that
    # reorders vertices cannot silently invalidate the row-for-row pairing.
    coords_x = mesh.geometry.x[:, 0]
    assert np.all(np.diff(coords_x) > 0), "interval vertices not ascending"

    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))

    m = Field(S3)
    m.set_with_numpy_array_debug(_m_gen(np.array([coords_x])).flatten())

    ex = Exchange(A)
    ex.setup(m, Field(DG, MS))

    H = ex.compute_field()

    m_comp = _vectors(m.get_numpy_array_debug())
    H_comp = _vectors(H)
    return m_comp, H_comp


def test_against_nmag():
    m_comp, H_comp = _compute()

    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m0_nmag.txt"))
    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "H_exc_nmag.txt"))
    assert m_ref.shape == m_comp.shape == (11, 3)
    assert H_ref.shape == H_comp.shape == (11, 3)

    # The analytic m0 must equal the Nmag reference m0 (this is what makes the
    # row-for-row H comparison meaningful). Measured max abs diff ~7.2e-16.
    m_abs = np.max(np.abs(m_ref - m_comp))
    print("m0 vs Nmag reference, max abs diff:", m_abs)
    assert m_abs < 1e-12

    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_comp = np.cross(m_comp, H_comp)
    diff = np.abs(m_cross_H_ref - m_cross_H_comp)
    scale = max(np.linalg.norm(v) for v in m_cross_H_ref)
    rel_diff = diff / scale
    max_rel = float(np.max(rel_diff))
    print("m x H_exchange vs Nmag, max relative difference:", max_rel)

    # Non-trivial witness: the computed exchange field must be genuinely large
    # (order MA/m) and comparable to the Nmag reference magnitude, so this is not
    # a vacuous 0-vs-0 pass. Nmag reference |H| max ~3.9e6 A/m.
    finmag_scale = np.max(np.linalg.norm(H_comp, axis=1))
    ref_scale = np.max(np.linalg.norm(H_ref, axis=1))
    print("finmag |H_exchange| max:", finmag_scale, " Nmag ref:", ref_scale)
    assert finmag_scale > 0.5 * ref_scale

    assert max_rel < REL_TOLERANCE, (
        "max rel_diff {} exceeds tolerance {}".format(max_rel, REL_TOLERANCE))


if __name__ == "__main__":
    test_against_nmag()
