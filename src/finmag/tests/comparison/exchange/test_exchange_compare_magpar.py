"""Magpar EXCHANGE-field comparison under DOLFINx (coordinate-based witness).

This restores the legacy ``test_exchange_compare_magpar`` regression against the
checked-in Magpar reference exchange field, rebuilt for DOLFINx as a
COORDINATE-BASED comparison: instead of assuming finmag and Magpar share an
identical node *order* (the legacy path did ``mesh.coordinates()[:] *= 1e9`` and
paired arrays position-by-position via ``magpar.compare_field``), we match each
Magpar node to the finmag mesh vertex at the SAME physical coordinate and
compare the finmag exchange field value there. Each finmag/Magpar pair is
therefore matched by *location*, not by array index.

MASTER -> PORT traceability:
  * Master ``test_three_dimensional_problem()`` (pytest entry point) and its
    helper ``three_dimensional_problem()`` (mesh/field setup + comparison), both
    in ``test_exchange_compare_magpar.py`` (git show b5015c5a) -> port
    ``test_exchange_field_matches_magpar_at_saved_coordinates()`` and its helper
    ``_compute()`` below. Same physics (Exchange field, C=1.3e-11, Ms=8.6e5, on
    a 10nm x 1nm x 1nm 40x2x2 box, m0 = (sin^2, 0, cos^2) pattern) compared
    against the same checked-in Magpar reference (``magpar_result/test_exch``).
  * The only structural change is the pairing strategy: master's node-order,
    position-by-position array zip (``magpar.compare_field``) is replaced by
    the coordinate-keyed nodal lookup described below, because DOLFINx's mesh
    generator does not reproduce dolfin's legacy vertex *order* (M8
    mesh-drift) -- so index-for-index pairing is no longer valid, even though
    the vertex *coordinates* still coincide exactly (see the coord-match-
    distance assertion in ``_compute``).
  * The interior ``evaluate_at_point`` probe check in ``_compute`` is a NEW
    sanity spot-check with no master ancestor; it exists only to demonstrate
    the nodal-vs-interpolated agreement in the interior, and is deliberately
    NOT used on the boundary because of a known ``evaluate_at_point``
    outer-face defect -- see the next paragraph for the workaround.

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
vertex -- the primary comparison never calls the interpolating probe at all.
NB (``evaluate_at_point`` outer-face defect + workaround): the interpolating
point-in-cell probe (``Field.probe`` / ``evaluate_at_point``) is exercised here
ONLY as a NEW sanity spot-check at an INTERIOR coordinate (see the
"MASTER -> PORT" note above); it is NOT used to drive the boundary comparison,
because its bounding-box point-in-cell location is unreliable for points lying
exactly on the outer ``x = x_max`` face (it can resolve to the wrong boundary
vertex), which would inject a spurious ~0.5 disagreement that is a
probe-location artifact, not a real field difference. The workaround is simply
to never probe on the outer face: the coordinate-keyed nodal lookup used for
every comparison point (interior and boundary alike) is exact and side-steps
the probe entirely.

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

# RED-first tolerance check (measured in this environment, 3 repeat runs,
# bit-identical each time -- this comparison is deterministic, no MPI/thread
# nondeterminism observed):
#   max rel_diff  = 8.698001683854669e-08
#   mean rel_diff = 4.480401491142829e-09
# Master's tolerance (``REL_TOLERANCE = 9e-8`` in test_exchange_compare_magpar.py,
# labelled "needs higher accuracy patch") PASSES this measured value verbatim
# (margin ~3.4%) -- because Magpar nodes land exactly on finmag CG1 vertices,
# the coordinate-keyed nodal lookup here reduces to the same node-for-node
# values master compared, so master's accuracy-patch-dependent tolerance
# carries over unchanged. Kept verbatim rather than loosened: a units or
# component-ordering bug would blow this up by many orders of magnitude (the
# bulk of nodes agree to ~1e-9), so 9e-8 remains a real, tight witness, not
# papering over a genuine disagreement.
REL_TOLERANCE = 9e-8


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
    # master: df.BoxMesh(df.Point(0,0,0), df.Point(x_max,y_max,z_max), 40, 2, 2)
    mesh = dm.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [X_MAX, Y_MAX, Z_MAX]],
        [40, 2, 2],
    )
    # master: df.VectorFunctionSpace(mesh, 'Lagrange', 1) / df.FunctionSpace(mesh, 'DG', 0)
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))

    # master: vector_valued_function((m0_x, m0_y, m0_z), V, normalise=True) with
    # m0_x = "pow(sin(0.2*x[0]*1e9), 2)", m0_y = "0", m0_z = "pow(cos(0.2*x[0]*1e9), 2)"
    # -> reimplemented as the vectorised callable ``_m0_callable`` above (no
    # dolfin Expression string-JIT under DOLFINx).
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

    # --- Magpar reference (saved coords + saved field), read dolfin-free ---
    # master: magpar.get_field(magpar_result, 'exch') (imports dolfin at module
    # scope) -> magpar_io.get_field, the dolfin-free extraction of the same
    # reader (see finmag.util.magpar_io module docstring).
    base = os.path.join(MODULE_DIR, "magpar_result", "test_exch")
    nodes_nm, magpar_flat = magpar_io.get_field(base, "exch")
    N = nodes_nm.shape[0]
    assert N == N_NODES, "unexpected Magpar node count {}".format(N)

    # --- UNIT CHECK: confirm nodes_nm*1e-9 falls within the finmag mesh -----
    # master converted its OWN mesh to nm (``mesh.coordinates()[:] = tmp_c * 1e9``)
    # to match Magpar's nm-scale nodes before pairing; the port instead leaves
    # the finmag mesh in metres and converts Magpar's saved nm nodes to metres
    # (``nodes_m = nodes_nm * 1e-9`` below) -- inverse direction, same physical
    # coordinates, no rescaling of the finmag mesh itself.
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
        H=H,
        geo=geo,
        finmag_nodal=finmag_nodal,
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


# ==========================================================================
# ===== NEW under DOLFINx (no master ancestor) ============================
# ==========================================================================

def test_interior_probe_matches_nodal_value():
    """Sanity spot-check: ``evaluate_at_point`` agrees with the CG1 nodal value
    at an INTERIOR coordinate.

    No master ancestor -- master never probed the field, it only compared
    nodal arrays. This exercises ``Field.probe`` / ``evaluate_at_point``
    (unused by the primary coordinate-keyed comparison above) but deliberately
    ONLY at an interior point: probing on the outer ``x = x_max`` face is
    known to be unreliable (see the module docstring's "outer-face defect"
    paragraph) and is not exercised here.
    """
    res = _compute()
    geo = res["geo"]
    finmag_nodal = res["finmag_nodal"]
    interior = np.array([5e-9, 0.0, 0.0])
    j_int = np.argmin(np.abs(geo - interior).sum(axis=1))
    probe_int = evaluate_at_point(res["H"].f, interior)
    assert np.allclose(probe_int, finmag_nodal[j_int], rtol=1e-9, atol=1e-3), (
        "interior probe {} disagrees with nodal value {}".format(
            probe_int, finmag_nodal[j_int]
        )
    )


if __name__ == "__main__":
    r = _compute()
    print("max rel_diff", np.nanmax(r["rel_diff"]))
    print("mean rel_diff", np.nanmean(r["rel_diff"]))
