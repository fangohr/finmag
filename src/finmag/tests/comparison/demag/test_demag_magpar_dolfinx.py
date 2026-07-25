"""Coordinate-based DOLFINx restoration of the Magpar demag comparison.

Restores ``test_demag_field.py::test_using_magpar`` (currently an M8 xfail).
The legacy test compared the finmag demag field to a checked-in Magpar
reference **node-for-node**, which broke when the sphere mesh was regenerated
under a newer Netgen (M8 node drift): the saved Magpar node array no longer
lines up index-for-index with the finmag mesh vertices, so
``compare_field_directly`` aborted on a shape/coordinate mismatch.

This restoration is a *probe-at-saved-coordinates* comparison instead: the
finmag demag field is evaluated (``Field.probe``) at each saved Magpar node
coordinate, so the two meshes no longer need to share vertex ordering (or even
the same tessellation). No live Magpar is executed -- running ``magpar.exe``
remains deferred (acceptance register M15); we only read the checked-in
reference via the dolfin-free :mod:`finmag.util.magpar_io`.

Unit / normalisation notes (verified, not assumed):

* ``sphere.geo`` yields mesh coordinates on the same nm-scale (~[-10, 10]) as
  the saved Magpar ``.femsh`` nodes, so we probe **directly** at the node
  coordinates with no ``1e-9`` rescaling (unit check asserted below).
* The saved Magpar run used polarisation ``Js = 1e-6 T`` (``test_demag.log``:
  Edem = 1.319e-7 J/m^3), i.e. ``Ms = Js/mu0 ~ 0.7958 A/m`` -- NOT the ``Ms = 1``
  the legacy finmag setup used. That factor-0.796 mismatch is exactly why the
  legacy assertion carried the meaningless ``REL_TOLERANCE = 10.0``. Here we run
  finmag at the **same** Ms so the comparison is physically meaningful (both
  fields ~ ``-Ms/3`` in x).

Boundary handling (IMPORTANT -- interpolation is only trusted in the interior):
The interpolating point evaluator (``Field.probe`` / ``evaluate_at_point``) has
a known latent defect for points lying exactly on an *outer* mesh face: it can
resolve to the wrong vertex and return a spuriously large (~0.5) error. The
Magpar sphere-surface nodes (``felog``: n_vert_bnd=552) do not coincide with the
finmag ``sphere.geo`` vertices, so they must be interpolated -- exactly where
that defect bites and where genuine demag surface artefacts are also largest.
We therefore restrict the comparison to nodes that are strictly *interior* to
the finmag mesh: a node is included only if ``probe`` succeeds at the node AND
at the node nudged slightly *outward* (proving mesh exists further out, so the
node is not on/just-outside the outer boundary). Any node exactly coincident
with a finmag vertex is read by exact nodal lookup (no interpolation). Surface
nodes -- which here all fall strictly outside the faceted finmag sphere -- are
excluded and counted. This is deliberately NOT absorbed into an inflated
tolerance.

Measured (this environment): N=1084, 0 coincident, 532 interior included, 552
surface excluded; interior max rel_diff ~ 4.7e-3, mean ~ 1.5e-3, no node above
0.1 (i.e. the boundary-probe artefact does not contaminate the retained set).

Test layout: master's ``test_using_magpar`` had exactly one assertion
(``assert np.max(rel_diff) < REL_TOLERANCE``); everything else it computed
(unit ranges, mean vectors, the magpar-vs-analytic comparison) was only
printed/tabulated into ``table.rst``, never asserted. That single assertion is
restored verbatim-in-spirit below as ``test_demag_against_magpar_at_coords``
(same comparison, coordinate-probed values instead of node-order-paired
values). The quantities master only printed are promoted to hard assertions
under the ``NEW under DOLFINx`` banner further down, because the
coordinate-probe restructuring introduces failure modes master's node-order
path could not have (e.g. all points silently excluded by the interior
filter, or a coordinate-scale mismatch between the two meshes) -- a test that
only prints these would not catch that. [Claude Opus 4.8]
"""

import os

import numpy as np
import pytest
from scipy.spatial import cKDTree
import dolfinx.fem as fem

from finmag.field import Field
from finmag.energies import Demag
from finmag.util.meshes import from_geofile
from finmag.util import magpar_io

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

mu0 = np.pi * 4e-7
# Magpar reference used Js = 1e-6 T -> Ms = Js/mu0 (see module docstring).
MS = 1e-6 / mu0

# Master (test_demag_field.py::test_using_magpar) used REL_TOLERANCE = 10.0,
# which was meaningless (it only had to absorb the Ms=1 vs Ms=0.7958
# mismatch described above -- any real regression would have passed too).
# Measured interior max rel_diff here, at matched Ms and probe-at-coords, is
# ~4.7e-3; pinned at 1.5e-2 (~3x headroom) to absorb Netgen mesh-to-mesh
# variation. NOT inflated to hide boundary-probe garbage -- boundary nodes are
# excluded, not tolerated.
REL_TOLERANCE = 1.5e-2

_COINCIDENT_TOL = 1e-6   # a Magpar node this close to a finmag vertex is exact
_OUTWARD_EPS = 1e-3      # relative outward nudge for the interior test


def _setup_sphere_demag():
    mesh = from_geofile(os.path.join(MODULE_DIR, "sphere.geo"),
                        save_result=False)
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))
    m = Field(S3, (1.0, 0.0, 0.0))
    demag = Demag()
    demag.setup(m, Field(DG, MS), unit_length=1e-9)
    H = Field(S3)
    H.set_with_numpy_array_debug(demag.compute_field())
    return mesh, H


def _sample_finmag_at_nodes(H, nodes):
    """Sample ``H`` at ``nodes`` using exact nodal lookup for coincident nodes
    and interpolation only at strictly-interior nodes.

    Returns ``(values, kind)`` with ``kind`` in {0: coincident (exact nodal),
    1: interior (interpolated), -1: excluded (surface/boundary)}.
    """
    fverts, fvals = H.coords_and_values()
    tree = cKDTree(fverts)
    centroid = fverts.mean(axis=0)
    dist, idx = tree.query(nodes)

    N = nodes.shape[0]
    values = np.full((N, 3), np.nan)
    kind = np.full(N, -1, dtype=int)
    for i in range(N):
        if dist[i] < _COINCIDENT_TOL:
            values[i] = fvals[idx[i]]     # exact nodal value, no interpolation
            kind[i] = 0
            continue
        p = nodes[i]
        try:
            v = H.probe(p)
        except RuntimeError:
            continue                      # strictly outside the finmag mesh
        outward = p + (p - centroid) * _OUTWARD_EPS
        try:
            H.probe(outward)              # mesh exists further out -> interior
        except RuntimeError:
            continue                      # on/just-inside the outer boundary
        values[i] = v
        kind[i] = 1
    return values, kind


@pytest.fixture(scope="module")
def _magpar_comparison():
    """Shared demag solve + Magpar sampling for every test in this module (one
    demag solve, reused by the restored master assertion and by the NEW
    witnesses below, rather than re-solving per test)."""
    mesh, H = _setup_sphere_demag()

    nodes, flat = magpar_io.get_field(
        os.path.join(MODULE_DIR, "magpar_result", "test_demag"), "demag")
    N = nodes.shape[0]
    magpar_vecs = np.column_stack((flat[:N], flat[N:2 * N], flat[2 * N:3 * N]))

    fin_vecs, kind = _sample_finmag_at_nodes(H, nodes)
    included = kind >= 0

    norm = np.max(np.linalg.norm(magpar_vecs[included], axis=1))
    rel_diff = np.linalg.norm(
        fin_vecs[included] - magpar_vecs[included], axis=1) / norm

    return dict(
        mesh=mesh, nodes=nodes, N=N, magpar_vecs=magpar_vecs,
        fin_vecs=fin_vecs, kind=kind, included=included,
        n_coincident=int((kind == 0).sum()),
        n_interior=int((kind == 1).sum()),
        n_excluded=int((kind == -1).sum()),
        rel_diff=rel_diff,
    )


def test_demag_against_magpar_at_coords(_magpar_comparison):
    """finmag demag sampled at saved Magpar node coordinates matches Magpar.

    Direct restoration of master's ``test_using_magpar`` assertion
    (``assert np.max(rel_diff) < REL_TOLERANCE``), evaluated at coordinate-
    probed values instead of node-order-paired values (M8, see module
    docstring).
    """
    c = _magpar_comparison
    print("N nodes =", c["N"])
    print("coincident (exact nodal):", c["n_coincident"],
          "interior (interpolated):", c["n_interior"],
          "excluded (surface/boundary):", c["n_excluded"])
    print("comparison with magpar, H, relative_difference:")
    print("interior max rel_diff:", np.max(c["rel_diff"]),
          "mean:", np.mean(c["rel_diff"]))

    # PRIMARY assertion -- master: assert np.max(rel_diff) < REL_TOLERANCE
    # (master's REL_TOLERANCE=10.0 was meaningless; see docstring/comment
    # above for the measured value and the justified replacement bound).
    assert np.max(c["rel_diff"]) < REL_TOLERANCE


# ==========================================================================
# ===== NEW under DOLFINx (no master ancestor) ============================
# ==========================================================================
# Master computed each of these quantities too (coordinate ranges, mean
# field vectors, the magpar-vs-analytic-solution comparison) but only ever
# printed or tabulated them into table.rst -- it never asserted on them,
# because on a shared node-ordered mesh they could not silently degrade
# without the primary assertion above also catching it. The coordinate-probe
# restructuring breaks that guarantee (e.g. the interior filter could
# silently exclude every node, or the two meshes could be built on
# mismatched coordinate scales, and the primary test would still "pass" by
# comparing an empty or vacuous set). These are promoted to hard assertions
# so that cannot happen unnoticed.

def test_probe_coordinate_scale_matches_magpar_nodes(_magpar_comparison):
    """Unit check: mesh coords and Magpar nodes must share scale (both nm)."""
    c = _magpar_comparison
    coords = c["mesh"].geometry.x
    nodes = c["nodes"]
    print("finmag mesh coord range:", coords.min(axis=0), coords.max(axis=0))
    print("magpar node coord range:", nodes.min(axis=0), nodes.max(axis=0))
    assert np.allclose(coords.min(axis=0), nodes.min(axis=0), atol=0.5)
    assert np.allclose(coords.max(axis=0), nodes.max(axis=0), atol=0.5)


def test_probe_interior_coverage_is_substantial(_magpar_comparison):
    """Coverage witness: the retained interior set must be substantial (a
    broken probe that dropped everything must not pass vacuously)."""
    c = _magpar_comparison
    n_included = c["n_coincident"] + c["n_interior"]
    assert n_included > c["N"] // 3


def test_finmag_mean_field_matches_analytic_sphere(_magpar_comparison):
    """Genuine witness: a broken/zero demag solve would not sit on -Ms/3 in x."""
    c = _magpar_comparison
    fin_mean = c["fin_vecs"][c["included"]].mean(axis=0)
    print("finmag mean vec:", fin_mean)
    assert np.isclose(fin_mean[0], -MS / 3.0, rtol=5e-2)
    assert abs(fin_mean[1]) < 5e-2 * MS
    assert abs(fin_mean[2]) < 5e-2 * MS


def test_magpar_reference_mean_field_matches_analytic_sphere(_magpar_comparison):
    """Secondary witness: the Magpar reference itself, normalised by its own
    Ms, sits near the analytic uniform-sphere value (-1/3, 0, 0)."""
    c = _magpar_comparison
    magpar_norm_mean = c["magpar_vecs"].mean(axis=0) / MS
    print("magpar normalised mean vec:", magpar_norm_mean)
    assert np.isclose(magpar_norm_mean[0], -1.0 / 3.0, atol=2e-2)
    assert abs(magpar_norm_mean[1]) < 2e-2
    assert abs(magpar_norm_mean[2]) < 2e-2
