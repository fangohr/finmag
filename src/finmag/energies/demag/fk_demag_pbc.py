"""Periodic (macro-geometry) demag, ported to DOLFINx (Task 23).

The Fredkin-Koehler demag field of a finite sample can be turned into that of
an infinite periodic array of copies by replacing the single-sample boundary
element matrix with a sum of Lindholm double-layer contributions over a lattice
of translation vectors ``Ts`` (the "macro geometry"): each image copy of the
boundary contributes a translated boundary-element block, and the
solid-angle diagonal is added exactly once.

``MacroGeometry`` builds the (in-plane) translation lattice; ``build_periodic_bem``
assembles the periodic dense BEM from the ported, coordinate-driven boundary
arrays (:func:`finmag.energies.demag.fk_demag.boundary_bem_arrays`) using the
native ``finmag.native.treecode_bem`` Lindholm kernels (``build_boundary_matrix``);
``BMatrixPBC`` preserves the legacy ``BMatrixPBC(mesh, Ts)`` construction surface
on top of a DOLFINx mesh.

The single-tile case (``Ts=[(0,0,0)]``) reproduces the dense Fredkin-Koehler
BEM (``finmag.native.bem_arrays.compute_bem_fk_from_arrays``) bit-for-bit -- the
treecode Lindholm kernels use the same convention/orientation as the array BEM.

Only the ``dolfin``-free FEM-API layer changed relative to the legacy module;
the periodic-summation physics is preserved. [Claude Opus 4.8]
"""

import logging

import numpy as np

logger = logging.getLogger("finmag")


class MacroGeometry(object):
    """In-plane periodic tiling of the sample for macro-geometry demag.

    Preserves the legacy public surface: ``MacroGeometry(nx, ny, dx, dy, Ts)``.
    ``nx``/``ny`` (odd, positive) are the number of image tiles along x and y;
    ``dx``/``dy`` are the tile spacings (inferred from the mesh bounding box if
    omitted); an explicit ``Ts`` overrides everything.
    """

    def __init__(self, nx=None, ny=None, dx=None, dy=None, Ts=None):
        self.nx = nx or 1
        self.ny = ny or 1
        self.dx = dx
        self.dy = dy
        self.Ts = Ts

        if Ts is not None and (nx is not None or ny is not None
                               or (dx is not None and dy is not None)):
            logger.warning(
                "Ignoring arguments 'nx', 'ny', 'dx', 'dy' because 'Ts' is "
                "explicitly provided.")
        elif Ts is None:
            if (self.nx < 1 or self.nx % 2 == 0
                    or self.ny < 1 or self.ny % 2 == 0):
                raise ValueError(
                    "Both nx and ny should be larger than 0 and must be odd.")

    def compute_Ts(self, mesh):
        """Return the list of ``[Tx, Ty, 0]`` image-translation vectors."""
        if self.Ts is not None:
            return self.Ts

        # Only inspect the mesh bounding box for spacings that were not given
        # explicitly, so an all-explicit MacroGeometry needs no mesh.
        if self.dx is None or self.dy is None:
            dx, dy = self.find_mesh_info(mesh)
            if self.dx is None:
                self.dx = dx
            if self.dy is None:
                self.dy = dy

        Ts = []
        for i in range(-self.nx // 2 + 1, self.nx // 2 + 1):
            for j in range(-self.ny // 2 + 1, self.ny // 2 + 1):
                Ts.append([self.dx * i * 1.0, self.dy * j * 1.0, 0.0])

        logger.debug(
            "Creating macro-geometry with demag {} x {} tiles (dxdy: {} x "
            "{}).".format(self.nx, self.ny, self.dx, self.dy))
        self.Ts = Ts
        return self.Ts

    def find_mesh_info(self, mesh):
        """Bounding-box extents (dx, dy) of a DOLFINx mesh (mesh-coord units)."""
        xt = _mesh_coordinates(mesh)
        sizes = xt.max(axis=0) - xt.min(axis=0)
        return sizes[0], sizes[1]


def build_periodic_bem(coords, cells, bsa, Ts):
    """Assemble the periodic Fredkin-Koehler dense BEM.

    *Arguments*

    - ``coords``: ``(n, 3)`` boundary-node coordinates in BEM-local order;
    - ``cells``: ``(m, 3)`` outward-oriented boundary triangles (BEM-local);
    - ``bsa``: ``(n,)`` boundary solid-angle diagonal (BEM-local order);
    - ``Ts``: iterable of ``(Tx, Ty, Tz)`` image-translation vectors.

    Returns the dense ``(n, n)`` periodic boundary-element matrix.
    """
    from finmag.native.treecode_bem import build_boundary_matrix

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    face_nodes = np.ascontiguousarray(cells, dtype=np.int32)
    n = coords.shape[0]
    n_face = face_nodes.shape[0]
    bm = np.zeros((n, n), dtype=np.float64)

    for T in Ts:
        build_boundary_matrix(
            coords, face_nodes, bm,
            np.ascontiguousarray(np.asarray(T, dtype=np.float64)),
            n, n_face)

    diag = np.arange(n)
    bm[diag, diag] += np.asarray(bsa, dtype=np.float64)
    return bm


class BMatrixPBC(object):
    """Legacy ``BMatrixPBC(mesh, Ts)`` surface on a DOLFINx mesh.

    Extracts the boundary arrays from ``mesh`` (via the ported FK boundary
    extraction) and builds the periodic BEM in ``self.bm``.  ``self.coords`` and
    ``self.b2g_map`` expose the BEM-local boundary ordering.
    """

    def __init__(self, mesh, Ts=[(0.0, 0.0, 0.0)]):
        from dolfinx import fem

        from .fk_demag import boundary_bem_arrays, boundary_solid_angles

        self.mesh = mesh
        S1 = fem.functionspace(mesh, ("Lagrange", 1))
        coords, cells, b2g = boundary_bem_arrays(mesh, S1)
        self.coords = coords
        self.cells = cells
        self.b2g_map = np.asarray(b2g, dtype=np.int64)
        self.Ts = np.array(Ts, dtype=np.float64)
        self.vert_bsa = boundary_solid_angles(mesh, S1, self.b2g_map)
        self.bm = build_periodic_bem(coords, cells, self.vert_bsa, self.Ts)

    def compute_bmatrix(self):
        """Recompute ``self.bm`` from the stored boundary arrays and ``Ts``."""
        self.bm = build_periodic_bem(
            self.coords, self.cells, self.vert_bsa, self.Ts)
        return self.bm


def _mesh_coordinates(mesh):
    """Geometry node coordinates ``(n, 3)`` of a DOLFINx mesh.

    Only used for the sample bounding box (``dx``/``dy`` extents), so all local
    geometry nodes suffice -- no vertex/geometry-node correspondence assumed.
    """
    return mesh.geometry.x
