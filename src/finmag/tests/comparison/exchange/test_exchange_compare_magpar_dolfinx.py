"""Magpar EXCHANGE-field comparison under DOLFINx (coordinate-based witness).

This restores the legacy ``test_exchange_compare_magpar`` regression against the
checked-in Magpar reference exchange field, rebuilt for DOLFINx as a
COORDINATE-BASED comparison: instead of assuming finmag and Magpar share an
identical node *order* (the legacy path did ``mesh.coordinates()[:] *= 1e9`` and
paired arrays position-by-position via ``magpar.compare_field``), we match each
Magpar node to the finmag mesh vertex at the SAME physical coordinate and
compare the finmag exchange field value there. Each finmag/Magpar pair is
therefore matched by *location*, not by array index.

Why coordinate-based:
  * Immune to M8 (mesh-drift): the DOLFINx mesh generator no longer emits
    vertices in the same *order* legacy dolfin did, so the old node-order
    pairing is invalid. Matching by physical coordinate sidesteps node ordering
    entirely -- it depends only on the vertex coordinates coinciding, which we
    assert (coord-match distance is 0: the 40x2x2 BoxMesh produces exactly
    Magpar's 369 nodes at identical coordinates).
  * No live Magpar (M15): ``magpar.exe`` is never run here. We only read the
    checked-in reference files through the dolfin-free readers in
    ``finmag.util.magpar_io`` (``get_field`` -> saved coords + saved field).

Field value at a saved coordinate. Every Magpar node coincides exactly with a
finmag CG1 vertex, so the finmag field value there is simply that vertex's nodal
value (a CG1 nodal coefficient IS the field value at the vertex -- no
interpolation error). We therefore read the finmag nodal value at the matched
vertex. NB: the interpolating point-in-cell probe (``Field.probe`` /
``evaluate_at_point``) is exercised here as a sanity spot-check at an INTERIOR
coordinate, but is NOT used to drive the boundary comparison: its bounding-box
point-in-cell location is unreliable for points lying exactly on the outer
``x = x_max`` face (it can resolve to the wrong boundary vertex), which would
inject a spurious ~0.5 disagreement that is a probe-location artifact, not a
real field difference. The coordinate-keyed nodal lookup is exact for all nodes.

The Magpar reference nodes are in NANOMETRES and the field is a flat array in
component-blocked order ``[Hx0..Hx(N-1), Hy0.., Hz0..]`` already converted to
A/m; the finmag mesh is in METRES, so the physical coordinate of Magpar node i
is ``node_nm[i] * 1e-9``. [Claude Opus 4.8]
"""

import os

import numpy as np

import dolfinx.mesh as dm
import dolfinx.fem as fem
from mpi4py import MPI

from finmag.field import Field, evaluate_at_point
from finmag.energies import Exchange
from finmag.util import magpar_io

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Legacy material/geometry parameters (git show b5015c5a).
X_MAX, Y_MAX, Z_MAX = 10e-9, 1e-9, 1e-9  # metres
MS = 8.6e5
C = 1.3e-11
N_NODES = 369  # Magpar reference node count

# MEASURED agreement: coordinate-keyed nodal comparison gives
#   max rel_diff = 8.70e-08, mean rel_diff = 4.48e-09
# (essentially the legacy 9e-8: because Magpar nodes land exactly on finmag CG1
# vertices, this is the legacy node-for-node comparison re-expressed as a
# coordinate match, so the same high-accuracy-patch agreement holds). Tolerance
# is that measured max plus ~2.3x headroom. Do NOT raise this to paper over a
# real disagreement: a units or component-ordering bug would blow it up by many
# orders of magnitude (the bulk here agrees to ~1e-7).
REL_TOLERANCE = 2e-7


def _m0_callable(x):
    """Legacy initial magnetisation, evaluated on DOLFINx coordinate columns.

    ``x`` is shape ``(3, n_points)`` in METRES. Legacy used
    ``mx = sin(0.2*x*1e9)**2, my = 0, mz = cos(0.2*x*1e9)**2`` (the ``x*1e9``
    converts the metre x-coordinate to nm), normalised per node.
    """
    xnm = 0.2 * x[0] * 1e9
    mx = np.sin(xnm) ** 2
    my = np.zeros_like(mx)
    mz = np.cos(xnm) ** 2
    norm = np.sqrt(mx**2 + my**2 + mz**2)
    return np.vstack((mx / norm, my / norm, mz / norm))


def _compute():
    mesh = dm.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [X_MAX, Y_MAX, Z_MAX]],
        [40, 2, 2],
    )
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))

    m = Field(S3, value=_m0_callable, normalised=True)

    # NOTE: no unit_length passed => unit_length defaults to 1, mesh is in metres.
    ex = Exchange(C)
    ex.setup(m, Field(DG, MS))

    H = Field(S3)
    H.set_with_numpy_array_debug(ex.compute_field())
    # Owned-vertex rows of the finmag exchange field, aligned with the owned
    # rows of ``mesh.geometry.x`` (both in owned-vertex order).
    geo = mesh.geometry.x
    finmag_nodal = H.get_numpy_array_debug().reshape((3, -1)).T

    # Sanity spot-check that the interpolating probe agrees with the nodal value
    # at an INTERIOR coordinate (exercises evaluate_at_point; boundary probing
    # is deliberately avoided, see module docstring).
    interior = np.array([5e-9, 0.0, 0.0])
    j_int = np.argmin(np.abs(geo - interior).sum(axis=1))
    probe_int = evaluate_at_point(H.f, interior)
    assert np.allclose(probe_int, finmag_nodal[j_int], rtol=1e-9, atol=1e-3), (
        "interior probe {} disagrees with nodal value {}".format(
            probe_int, finmag_nodal[j_int]
        )
    )

    # --- Magpar reference (saved coords + saved field), read dolfin-free ---
    base = os.path.join(MODULE_DIR, "magpar_result", "test_exch")
    nodes_nm, magpar_flat = magpar_io.get_field(base, "exch")
    N = nodes_nm.shape[0]
    assert N == N_NODES, "unexpected Magpar node count {}".format(N)

    # --- UNIT CHECK: confirm nodes_nm*1e-9 falls within the finmag mesh -----
    lo = geo.min(axis=0)
    hi = geo.max(axis=0)
    nodes_m = nodes_nm * 1e-9
    print("mesh.geometry.x min:", lo, "max:", hi)
    print("nodes_nm min:", nodes_nm.min(axis=0), "max:", nodes_nm.max(axis=0))
    print("nodes_m  min:", nodes_m.min(axis=0), "max:", nodes_m.max(axis=0))
    outside = np.logical_or(nodes_m < lo - 1e-15, nodes_m > hi + 1e-15)
    n_out = int(np.any(outside, axis=1).sum())
    print("nodes outside mesh bounds (>1e-15 non-microscopic):", n_out)
    assert n_out == 0, (
        "{} Magpar nodes land non-microscopically outside the mesh -- "
        "suspect a units error".format(n_out)
    )

    # --- Match each Magpar node to the finmag vertex at the same coordinate --
    magpar_vecs = np.empty((N, 3), dtype=np.float64)
    magpar_vecs[:, 0] = magpar_flat[0:N]
    magpar_vecs[:, 1] = magpar_flat[N:2 * N]
    magpar_vecs[:, 2] = magpar_flat[2 * N:3 * N]

    finmag_vecs = np.empty((N, 3), dtype=np.float64)
    max_match_dist = 0.0
    for i in range(N):
        j = int(np.argmin(np.abs(geo - nodes_m[i]).sum(axis=1)))
        finmag_vecs[i] = finmag_nodal[j]
        max_match_dist = max(
            max_match_dist, float(np.abs(geo[j] - nodes_m[i]).max())
        )
    print("max coord-match distance (m):", max_match_dist)
    # Guard the coincidence assumption: if DOLFINx ever drifts vertex
    # *coordinates* (not just order), this catches it before comparing.
    assert max_match_dist < 1e-13, (
        "Magpar nodes no longer coincide with finmag vertices "
        "(max match distance {} m)".format(max_match_dist)
    )

    diff = np.linalg.norm(finmag_vecs - magpar_vecs, axis=1)
    scale = np.max(np.linalg.norm(magpar_vecs, axis=1))
    rel_diff = diff / scale

    return dict(
        finmag=finmag_vecs,
        magpar=magpar_vecs,
        rel_diff=rel_diff,
        scale=scale,
    )


def test_exchange_field_matches_magpar_at_saved_coordinates():
    res = _compute()
    rel = res["rel_diff"]
    max_rel = float(np.nanmax(rel))
    mean_rel = float(np.nanmean(rel))
    print("max rel_diff :", max_rel)
    print("mean rel_diff:", mean_rel)
    print("magpar |H| scale (A/m):", res["scale"])

    # Sanity: a broken setup (zero / structureless field) must fail. The finmag
    # exchange field must be genuinely nonzero and its magnitude comparable to
    # Magpar's, so this is a real witness, not a vacuous 0-vs-0 pass.
    finmag_scale = np.max(np.linalg.norm(res["finmag"], axis=1))
    assert finmag_scale > 0.5 * res["scale"], (
        "finmag exchange field magnitude {} is implausibly small vs Magpar "
        "{} -- broken setup".format(finmag_scale, res["scale"])
    )

    assert max_rel < REL_TOLERANCE, (
        "max rel_diff {} exceeds tolerance {}".format(max_rel, REL_TOLERANCE)
    )


if __name__ == "__main__":
    r = _compute()
    print("max rel_diff", np.nanmax(r["rel_diff"]))
    print("mean rel_diff", np.nanmean(r["rel_diff"]))
