"""Array-only FK BEM native surface for the DOLFINx environment (Task 11a).

NO MASTER ANCESTOR (genuinely new under DOLFINx): ``finmag.native.bem_arrays``
is a new, array-only extraction of the FK/GCR BEM kernels, separated from the
legacy SWIG-DOLFIN mesh converters so it can build without ``-ldolfin`` on
Python 3.12. Master's own BEM test (``test_bem_computation.py``) exercises
the dolfin-mesh-coupled surface and is out of scope here (restored
separately); this file has no test-file ancestor of its own, so it is
validated instead against golden arrays captured from the known-good
compiled legacy module.

These tests exercise the ``finmag.native.bem_arrays`` extension: the
array-only Fredkin-Koehler BEM/LLG native bindings, separated from the legacy
SWIG-DOLFIN mesh converters and the ``-ldolfin`` link dependency so the surface
can be compiled and imported on Python 3.12 in the ``dolfinx`` pixi environment
*without* legacy DOLFIN installed.

The golden values below were captured from the known-good compiled
``finmag.native.llg`` module in the FEniCS-2019/Python-3.11 M3 lane (the
existing ``dev/bin/native-fk-bem-smoke.py`` reference path). The unit-cube
boundary arrays are exactly ``BoundaryMesh(UnitCubeMesh(1, 1, 1), "exterior")``,
so the DOLFINx-environment build must reproduce the legacy physics bit-for-bit.
[Claude Opus 4.8]
"""

import sys

import numpy as np
import pytest

# Importing the array-only native surface must NOT require legacy dolfin. This
# import also triggers the native `make` build of `bem_arrays.so`.
from finmag.native.bem_arrays import (
    compute_bem_fk_from_arrays,
    compute_bem_gcr_from_arrays,
    compute_lindholm_L,
    compute_lindholm_K,
)


# Boundary of dolfin.UnitCubeMesh(1, 1, 1) (8 vertices, 12 triangles), captured
# verbatim from the FEniCS-2019 lane so the arrays path gets identical input.
CUBE_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
CUBE_CELLS = np.array(
    [
        [1, 0, 2], [0, 1, 3], [0, 4, 2], [4, 0, 5],
        [6, 0, 3], [0, 6, 5], [1, 2, 7], [3, 1, 7],
        [2, 4, 7], [4, 5, 7], [6, 3, 7], [5, 6, 7],
    ],
    dtype=np.int64,
)
CUBE_B2G = np.array([0, 1, 3, 5, 2, 6, 4, 7], dtype=np.int64)

# Golden FK BEM matrix from the legacy compiled module for the arrays above.
GOLDEN_BEM_FK = np.array(
    [
        [-0.875, -0.017472244000087095, -0.012914061818109767, -0.012914061818109767, -0.017472244000087095, -0.012914061818109767, -0.017472244000087095, -0.03384108254540941],
        [-0.017472244000087095, -0.875, -0.010405719984878088, -0.010405719984878098, -0.019980585833318774, -0.019708034514991413, -0.019980585833318756, -0.02704710984852776],
        [-0.02704710984852776, -0.010405719984878098, -0.875, -0.019980585833318756, -0.010405719984878088, -0.01998058583331877, -0.019708034514991417, -0.017472244000087088],
        [-0.027047109848527765, -0.010405719984878088, -0.01998058583331877, -0.875, -0.019708034514991413, -0.019980585833318756, -0.010405719984878098, -0.017472244000087088],
        [-0.017472244000087095, -0.019980585833318756, -0.010405719984878098, -0.01970803451499141, -0.875, -0.010405719984878088, -0.019980585833318774, -0.02704710984852776],
        [-0.027047109848527765, -0.019708034514991417, -0.019980585833318756, -0.01998058583331877, -0.010405719984878098, -0.875, -0.010405719984878088, -0.017472244000087088],
        [-0.017472244000087095, -0.019980585833318774, -0.019708034514991413, -0.010405719984878088, -0.019980585833318756, -0.010405719984878098, -0.875, -0.027047109848527765],
        [-0.0338410825454094, -0.012914061818109763, -0.017472244000087088, -0.017472244000087088, -0.012914061818109763, -0.017472244000087088, -0.012914061818109763, -0.875],
    ],
    dtype=np.float64,
)

# Golden compute_lindholm_L(zeros, r1, r2, r3) (mirrors test_bem_computation.test_simple).
LINDHOLM_R1 = np.array([1.0, 0.0, 0.0])
LINDHOLM_R2 = np.array([2.0, 1.0, 3.0])
LINDHOLM_R3 = np.array([5.0, 0.0, 1.0])
GOLDEN_LINDHOLM_L = np.array(
    [-0.0015465160093087022, -0.0006150048560493065, -0.0004479797339585579]
)


def test_import_does_not_require_legacy_dolfin():
    """The array-only native surface must import with no legacy dolfin."""
    assert "dolfin" not in sys.modules
    # The legacy mesh-coupled module must not be pulled in transitively.
    assert "finmag.native.llg" not in sys.modules


def test_fk_bem_matches_legacy_golden():
    """compute_bem_fk_from_arrays reproduces the known-good legacy matrix."""
    bem, b2g = compute_bem_fk_from_arrays(CUBE_COORDS, CUBE_CELLS, CUBE_B2G)
    assert bem.shape == (8, 8)
    assert np.all(np.isfinite(bem))
    np.testing.assert_allclose(bem, GOLDEN_BEM_FK, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(b2g), CUBE_B2G)


def test_fk_bem_solid_angle_row_sum_identity():
    """FK BEM row sums equal -1: the Fredkin-Koehler solid-angle identity.

    The double-layer potential of a constant density over a closed surface,
    with the solid-angle diagonal term and the -1 shift, gives B @ 1 = -1.
    """
    bem, _ = compute_bem_fk_from_arrays(CUBE_COORDS, CUBE_CELLS, CUBE_B2G)
    row_sums = bem.sum(axis=1)
    np.testing.assert_allclose(row_sums, -np.ones(8), rtol=0, atol=1e-12)


def test_fk_bem_diagonal_is_negative():
    """Surface-node diagonal entries carry (solid_angle/4pi - 1) < 0."""
    bem, _ = compute_bem_fk_from_arrays(CUBE_COORDS, CUBE_CELLS, CUBE_B2G)
    diag = np.diag(bem)
    assert np.all(diag < 0.0)
    # All cube corners are congruent, so the diagonal is uniform here.
    np.testing.assert_allclose(diag, -0.875 * np.ones(8), rtol=0, atol=1e-14)


def test_gcr_bem_arrays_finite():
    """The GCR array entry point also builds a finite matrix (no dolfin)."""
    bem, b2g = compute_bem_gcr_from_arrays(CUBE_COORDS, CUBE_CELLS, CUBE_B2G)
    assert bem.shape == (8, 8)
    assert np.all(np.isfinite(bem))
    np.testing.assert_array_equal(np.asarray(b2g), CUBE_B2G)


def test_lindholm_L_matches_legacy_golden():
    """compute_lindholm_L reproduces the legacy single-triangle golden value."""
    L = compute_lindholm_L(np.zeros(3), LINDHOLM_R1, LINDHOLM_R2, LINDHOLM_R3)
    np.testing.assert_allclose(L, GOLDEN_LINDHOLM_L, rtol=0, atol=1e-15)


def test_lindholm_L_is_rotation_invariant():
    """The Lindholm double-layer weights are invariant under rigid rotation."""
    # A fixed proper rotation (about z by 37 degrees composed with about x).
    theta, phi = np.radians(37.0), np.radians(24.0)
    Rz = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    Rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(phi), -np.sin(phi)],
         [0.0, np.sin(phi), np.cos(phi)]]
    )
    Rot = Rz @ Rx
    r = np.array([0.3, -0.2, 0.7])
    base = compute_lindholm_L(r, LINDHOLM_R1, LINDHOLM_R2, LINDHOLM_R3)
    rotated = compute_lindholm_L(
        Rot @ r, Rot @ LINDHOLM_R1, Rot @ LINDHOLM_R2, Rot @ LINDHOLM_R3
    )
    np.testing.assert_allclose(rotated, base, rtol=0, atol=1e-13)


def test_lindholm_K_is_finite():
    """The single-layer (GCR) Lindholm weights build without dolfin."""
    K = compute_lindholm_K(np.zeros(3), LINDHOLM_R1, LINDHOLM_R2, LINDHOLM_R3)
    assert K.shape == (3,)
    assert np.all(np.isfinite(K))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
