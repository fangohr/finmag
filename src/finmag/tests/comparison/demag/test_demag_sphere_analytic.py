"""DOLFINx restoration of the demag-field comparison (SR1 P5.2, slice 3a).

Restores ``test_demag_field.py::test_using_analytical_solution``: the demag
field of a uniformly magnetised sphere is the analytic ``H = (-M/3, 0, 0)``.
This is a pure physical invariant -- no external reference data and no shared
reference mesh -- so it is immune to the M8 Netgen mesh-drift that retired the
legacy node-ordered Nmag/Magpar sphere comparisons.

The legacy ``compare_field``/``compare_field_directly`` node-order Magpar path
is intentionally NOT reproduced here (register M8); the restored coordinate-based
Magpar demag comparison lives in the Magpar probe-at-coords slice. This file
carries the analytic contract, pointwise, on the same ``sphere.geo`` geometry
the legacy test used. [Claude Opus 4.8]

Sphere-field coverage reconciliation (SR1 P5.2 minimal-diff pass): the
*canonical* uniformly-magnetised-sphere field test -- the minimal-diff
transcription of master ``fk_demag_test.py::
test_demag_field_for_uniformly_magnetised_sphere`` -- now lives in
``src/finmag/energies/demag/fk_demag_test.py`` (demag gate; formerly
``src/finmag/tests/test_fk_demag_dolfinx.py``), where it asserts
master's tighter 7e-3 absolute bound on ``sphere(r=1, maxh=0.2)`` (measured
max diff ~4.4e-3). THIS file is retained as the distinct comparison-gate
restoration of legacy ``test_demag_field.py`` on the *different* r=10
``sphere.geo`` geometry, exercising the ``from_geofile`` mesh path and the
``Demag()`` factory. To avoid two tests asserting the same invariant at
differently-justified tolerances, the previously-arbitrary 2e-2 bound here is
tightened to its own measured-justified value (see below).
"""

import os

import numpy as np
import dolfinx.fem as fem

from finmag.field import Field
from finmag.energies import Demag
from finmag.util.meshes import from_geofile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Legacy test_demag_field.py:64 used 2e-2, but that was looser than this
# geometry actually achieves under DOLFINx. Measured pointwise max rel_diff on
# the r=10 sphere.geo mesh (1335 vertices) is ~1.31e-2; tightened to 1.5e-2 so
# this comparison variant asserts at its own measured-justified bound rather
# than duplicating the canonical master test's contract at an arbitrary
# tolerance. (Canonical 7e-3-absolute test: energies/demag/fk_demag_test.py,
# formerly test_fk_demag_dolfinx.py.)
REL_TOLERANCE = 1.5e-2


def _setup_sphere_demag():
    mesh = from_geofile(os.path.join(MODULE_DIR, "sphere.geo"),
                        save_result=False)
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))
    Ms = 1.0
    m = Field(S3, (1.0, 0.0, 0.0))
    Ms_field = Field(DG, Ms)
    demag = Demag()
    demag.setup(m, Ms_field, unit_length=1e-9)
    return demag, Ms


def test_using_analytical_solution():
    """Uniformly magnetised sphere: expect H = (-1/3, 0, 0) at every node."""
    demag, Ms = _setup_sphere_demag()

    H = demag.compute_field().reshape((3, -1))
    H_ref = np.zeros(H.shape)
    H_ref[0] -= Ms / 3.0

    norm = np.sqrt(np.max(H_ref[0] ** 2 + H_ref[1] ** 2 + H_ref[2] ** 2))
    rel_diff = np.abs(H - H_ref) / norm

    # Pointwise contract: every node is within tolerance of the analytic value.
    assert np.max(rel_diff) < REL_TOLERANCE

    # A genuine-witness cross-check: the volume average must sit on -Ms/3 in x
    # and vanish in y/z. (A demag that returned ~0 -- the failure mode of a
    # broken solve -- would give rel_diff ~ 1 and fail the pointwise assert
    # above; this pins the sign and magnitude of the average as well.)
    avg = demag.average_field().reshape(-1)
    assert np.isclose(avg[0], -Ms / 3.0, rtol=REL_TOLERANCE)
    assert abs(avg[1]) < REL_TOLERANCE * Ms
    assert abs(avg[2]) < REL_TOLERANCE * Ms
