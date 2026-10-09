"""Boundary geometry for the Fredkin-Koehler demag BEM.

Pure mesh geometry, extracted from :mod:`fk_demag` so that module is about the
two-potential Poisson/Laplace solve rather than about surface extraction. None
of this touches the FEM solve: it turns a DOLFINx volume mesh into the
boundary-node coordinates, outward-oriented boundary triangles and
boundary-to-global dof map that the boundary-element kernels consume
(``finmag.native.bem_arrays`` for the dense Fredkin-Koehler matrix,
:mod:`fk_demag_pbc` for the periodic image sum, and
:mod:`treecode_bem` for the fast-summation approximation).

The BEM-local index of a boundary node is defined by its *coordinate*, which
keeps the mapping robust to DOLFINx submesh/entity reordering. That makes this
the one place a silent permutation would corrupt the demag field, so the
mapping is pinned bit-for-bit against the golden BEM matrix in
``fk_demag_test.py``.
"""

from math import pi

import numpy as np
from dolfinx import fem, mesh as dmesh


def boundary_bem_arrays(domain, S1):
    """Extract the FK BEM inputs from a DOLFINx volume mesh.

    Returns ``(coords, cells, b2g)`` where

    - ``coords`` is ``(n, 3)`` boundary-node coordinates in BEM-local order,
      one row per CG1 boundary degree of freedom of ``S1``;
    - ``cells`` is ``(m, 3)`` boundary triangles as BEM-local node indices,
      oriented so the triangle normal points *out* of the domain (the winding
      the Lindholm double-layer kernel and the legacy ``BoundaryMesh`` assume);
    - ``b2g`` is ``(n,)`` the S1 degree-of-freedom index for each BEM-local
      boundary node, so ``phi.x.array[b2g]`` gathers boundary values in
      BEM-local order and the reverse scatter writes them back.

    The BEM-local index of a boundary node is defined by its *coordinate*, so
    the mapping is robust to DOLFINx submesh/entity reordering. This is the
    exact place a silent permutation would corrupt the demag field, so it is
    pinned bit-for-bit against the Task 11a golden matrix.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0)
    domain.topology.create_connectivity(tdim, 0)
    ext_facets = dmesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(S1, fdim, ext_facets)
    boundary_dofs = np.asarray(boundary_dofs, dtype=np.int64)
    dof_coords = S1.tabulate_dof_coordinates()
    coords = np.ascontiguousarray(dof_coords[boundary_dofs], dtype=np.float64)

    n = coords.shape[0]
    keyed = {tuple(np.round(coords[i], 9)): i for i in range(n)}
    if len(keyed) != n:
        raise RuntimeError(
            "FK demag boundary extraction found coincident boundary-node "
            "coordinates; the coordinate mapping would be ambiguous"
        )

    # vertex -> coordinate, via the geometry dofmap (P1 simplex geometry).
    vcoords = domain.geometry.x
    geo_dofmap = domain.geometry.dofmap
    c2v = domain.topology.connectivity(tdim, 0)
    imap = domain.topology.index_map(0)
    nverts = imap.size_local + imap.num_ghosts
    vert_coord = np.full((nverts, 3), np.nan)
    for c in range(c2v.num_nodes):
        vs = c2v.links(c)
        gs = geo_dofmap[c]
        for lv, v in enumerate(vs):
            vert_coord[v] = vcoords[gs[lv]]

    f2v = domain.topology.connectivity(fdim, 0)
    f2c = domain.topology.connectivity(fdim, tdim)
    cells = np.empty((len(ext_facets), 3), dtype=np.int64)
    for row, f in enumerate(ext_facets):
        vs = list(f2v.links(f))
        p = [vert_coord[v] for v in vs]
        cell = f2c.links(f)[0]
        opp = [v for v in c2v.links(cell) if v not in vs][0]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(normal, p[0] - vert_coord[opp]) < 0.0:
            vs = [vs[0], vs[2], vs[1]]
        cells[row] = [keyed[tuple(np.round(vert_coord[v], 9))] for v in vs]

    return coords, cells, boundary_dofs


def boundary_solid_angles(domain, S1, b2g):
    """Boundary solid-angle correction per boundary node, in BEM-local order.

    Reproduces the legacy ``__compute_bsa`` accumulation: for every tetrahedron
    and each of its four vertices, add the solid angle subtended at that vertex
    by the opposite face; normalise by ``4*pi`` and subtract one; then restrict
    to the boundary via ``b2g``.  This is the diagonal of the Fredkin-Koehler
    boundary-element matrix (the periodic BEM adds it once, the treecode carries
    it as ``vert_bsa``).  ``b2g`` are S1 dof indices in BEM-local order.
    [Claude Opus 4.8]
    """
    from finmag.native.treecode_bem import compute_solid_angle_single

    dofmap = S1.dofmap
    dof_coords = S1.tabulate_dof_coordinates()
    ndofs = dofmap.index_map.size_local * dofmap.index_map_bs
    bsa = np.zeros(ndofs)
    tdim = domain.topology.dim
    ncells = domain.topology.index_map(tdim).size_local
    for c in range(ncells):
        d = dofmap.cell_dofs(c)
        for j in range(4):
            bsa[d[j]] += compute_solid_angle_single(
                np.ascontiguousarray(dof_coords[d[j]]),
                np.ascontiguousarray(dof_coords[d[(j + 1) % 4]]),
                np.ascontiguousarray(dof_coords[d[(j + 2) % 4]]),
                np.ascontiguousarray(dof_coords[d[(j + 3) % 4]]))
    bsa = bsa / (4.0 * pi) - 1.0
    return np.ascontiguousarray(bsa[np.asarray(b2g, dtype=np.int64)])


def boundary_triangle_normals(coords, cells):
    """Outward unit normals per boundary triangle.

    ``cells`` are already outward-oriented by :func:`boundary_bem_arrays`, so the
    normalised ``cross(p1-p0, p2-p0)`` points out of the domain -- the
    orientation the treecode ``FastSum`` (and the legacy ``face.normal()``)
    assume. [Claude Opus 4.8]
    """
    p0 = coords[cells[:, 0]]
    p1 = coords[cells[:, 1]]
    p2 = coords[cells[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    n = n / np.linalg.norm(n, axis=1)[:, None]
    return np.ascontiguousarray(n, dtype=np.float64)
