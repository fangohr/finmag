"""Treecode-accelerated Fredkin-Koehler demag, ported to DOLFINx (Task 23).

``TreecodeBEM`` is the ``Demag(solver='Treecode')`` variant.  It rides the
ported :class:`~finmag.energies.demag.fk_demag.FKDemag` unchanged except for the
one expensive step -- the boundary-element matrix-vector product
``phi_2|_bnd = B @ phi_1|_bnd``.  The dense BEM (``O(n^2)`` storage and matvec)
is replaced by the native octree fast-summation approximation of the *same*
Lindholm double-layer operator (``finmag.native.treecode_bem.FastSum``), so the
result converges to the dense FK demag as the accuracy knobs are tightened.

Accuracy knobs (legacy surface preserved):

- ``mac``: multipole acceptance criterion.  In the direct-sum limit the
  treecode evaluates the boundary sum exactly (matches dense FK to machine
  precision); at the legacy default ``mac=0.3`` it approximates it to ``~1e-4``.
- ``p``: multipole expansion order; ``num_limit``: max particles per leaf;
  ``correct_factor``: near-field correction radius; ``type_I``: index scheme.

Validation is by cross-method comparison against the dense FK demag on the same
geometry (there are no treecode oracle fixtures -- the frozen oracle env never
built the module); see ``test_treecode_pbc_demag_dolfinx.py`` and
``transition-notes.org``. [Claude Opus 4.8]
"""

import logging

import numpy as np

from .fk_demag import FKDemag, boundary_solid_angles, boundary_triangle_normals

logger = logging.getLogger("finmag")


class TreecodeBEM(FKDemag):
    """Fredkin-Koehler demag with the BEM matvec done by octree fast-summation.

    Constructor preserves the legacy signature
    ``TreecodeBEM(mac, p, num_limit, correct_factor, type_I, name,
    macrogeometry, thin_film)``.
    """

    def __init__(self, mac=0.3, p=3, num_limit=100, correct_factor=10,
                 type_I=True, name="Demag", macrogeometry=None,
                 thin_film=False):
        super(TreecodeBEM, self).__init__(
            name=name, macrogeometry=macrogeometry, thin_film=thin_film)
        self.mac = mac
        self.p = p
        self.num_limit = num_limit
        self.correct_factor = correct_factor
        self.type_I = type_I
        if macrogeometry is not None:
            # The legacy treecode path ignores the macro-geometry lattice (the
            # fast-summation kernel builds only the single-sample boundary sum);
            # preserved here for surface compatibility, with a warning.
            logger.warning(
                "TreecodeBEM ignores 'macrogeometry'; use the dense-BEM "
                "Demag(macrogeometry=...) for periodic demag.")

    def _setup_bem(self, coords, cells, b2g):
        """Build the treecode fast-summation operator instead of a dense BEM."""
        from finmag.native.treecode_bem import FastSum

        self._b2g_map = np.asarray(b2g, dtype=np.int64)
        self._bem = None  # no dense matrix in the treecode path

        coords_c = np.ascontiguousarray(coords, dtype=np.float64)
        face_nodes = np.ascontiguousarray(cells, dtype=np.int32)
        t_normals = boundary_triangle_normals(coords_c, face_nodes)
        vert_bsa = boundary_solid_angles(self.domain, self.S1, self._b2g_map)

        self._fast_sum = FastSum(
            p=self.p, mac=self.mac, num_limit=self.num_limit,
            correct_factor=self.correct_factor, type_I=self.type_I)
        self._fast_sum.init_mesh(coords_c, t_normals, face_nodes,
                                 np.ascontiguousarray(vert_bsa))
        self._phi2_b = np.zeros(coords_c.shape[0])

    def _apply_bem(self, phi_1_boundary):
        """``phi_2|_bnd = B @ phi_1|_bnd`` via the treecode fast-summation."""
        self._phi2_b[:] = 0.0
        self._fast_sum.fastsum(
            self._phi2_b, np.ascontiguousarray(phi_1_boundary, dtype=np.float64))
        return self._phi2_b
