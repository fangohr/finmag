# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import numpy as np

__all__ = ["Mesh", "MeshField"]

# A scalar/vector/tensor field on a mesh
class MeshField(object):
    def __init__(self, mesh, arr, dims):
        dims = tuple(dims)
        assert mesh.is_compatible(arr, len(dims))
        assert tuple(arr.shape[:len(dims)]) == dims

        self.flat = arr.view()
        self.nonflat = arr.view()
        self.nonflat.shape = np.append(dims, mesh.mesh_size_ao)
        self.flat.shape = np.append(dims, np.prod(mesh.mesh_size_ao))
        self.dims = dims
        self.mesh = mesh

    def to_xyz_array(self):
        n = len(self.dims)
        return np.ascontiguousarray(np.transpose(self.nonflat, axes=np.append(np.arange(n), n+np.argsort(self.mesh.array_order))))

    # Only implemented for vector fields, i.e. len(dims) == 1
    def subfield(self, a, b):
        assert len(self.dims) == 1
        return MeshField(self.mesh, self.flat[a:b], (b-a,))

    def copy(self):
        return MeshField(self.mesh, self.flat.copy(), self.dims)

# A 3D rectangular mesh
class Mesh(object):
    # The tuple is the indices for transposing from XYZ to the array order
    XYZ=(0, 1, 2)
    ZXY=(2, 0, 1)
    ZYX=(2, 1, 0)

    # meshsize and cellsize use XYZ order of coordinates
    # array_order is the order used by the
    def __init__(self, meshsize, cellsize=None, origin=(0,0,0), array_order=ZYX, size=None):
        if cellsize is None and size is not None:
            cellsize = np.array(size, dtype=float)/np.array(meshsize, dtype=float)
        self.mesh_size = np.array(meshsize, dtype=int)
        self.cell_size = np.array(cellsize, dtype=float)
        self.origin = np.array(origin, dtype=float)
        self.array_order = array_order
        self.n = np.prod(self.mesh_size)
        self.mesh_size_ao = self.mesh_size[list(array_order)]
        self.cell_size_ao = self.cell_size[list(array_order)]

        # Check validity
        assert self.mesh_size.shape == (3,)
        assert self.cell_size.shape == (3,)
        assert self.origin.shape == (3,)
        assert len(array_order) == 3

        assert self.cell_size[0] > 0 and self.cell_size[1] > 0 and self.cell_size[2] > 0
        assert self.mesh_size[0] > 0 and self.mesh_size[1] > 0 and self.mesh_size[2] > 0
        assert all(np.isfinite(self.cell_size)) and all(np.isfinite(self.origin))
        assert sorted(array_order) == [0, 1, 2]

    # An array is compatible with the specified mesh if
    #    a) it is C-contiguous
    #    b) it has more than one dimension
    # and either
    #    c1) it has dim+1 dimensions and the last dimension is equal to the total number of mesh points
    # or
    #    c2) it has dim+3 dimensions and the last 3 dimensions coincide with mesh_size
    def is_compatible(self, arr, ndim=1):
        if type(arr) is not np.ndarray:
            return False
        if not arr.flags.contiguous:
            return False
        if arr.ndim == ndim+1:
            return arr.shape[-1] == self.n
        elif arr.ndim == ndim+3:
            return arr.shape[-3] == self.mesh_size_ao[0] and arr.shape[-2] == self.mesh_size_ao[1] and arr.shape[-1] == self.mesh_size_ao[2]
        else:
            return False

    # Returns an array of the form [(x_min, x_max, x_num), (y_min, y_max, y_num), (z_min, z_max, z_num)]
    def get_lattice_spec(self):
        return [(self.origin[i] + self.cell_size[i]*0.5, self.origin[i] + self.cell_size[i]*0.5 + self.cell_size[i]*self.mesh_size[i], self.mesh_size[i]) for i in xrange(3)]

    def field_from_xyz_array(self, arr):
        assert arr.ndim == 4
        assert tuple(arr.shape[1:]) == tuple(self.mesh_size)
        res = arr.view()
        res.shape = np.append(arr.shape[:1], self.mesh_size)
        res = np.ascontiguousarray(np.transpose(res, axes=np.append([0], 1+np.array(self.array_order))))
        return MeshField(self, res, arr.shape[:1])

    def new_field(self, dims):
        if type(dims) == int:
            dims = [dims]
        arr = np.zeros(np.append(dims, [self.n]))
        return MeshField(self, arr, dims)

    def field_from_array(self, arr, ndim=1):
        return MeshField(self, arr, arr.shape[:ndim])

    def field_from_flat_array(self, dims, arr):
        a = arr.view()
        a.shape = (dims, -1)
        return MeshField(self, a, [dims])

    def iter_coords_int(self):
        l, m, n = self.array_order
        for i in xrange(self.mesh_size[l]):
            for j in xrange(self.mesh_size[m]):
                for k in xrange(self.mesh_size[n]):
                    r = [-1] * 3
                    r[l] = i; r[m] = j; r[n] = k;
                    yield tuple(r)

    def iter_coords(self):
        for r in self.iter_coords_int():
            x = self.origin[0] + self.cell_size[0]*0.5 + self.cell_size[0]*r[0]
            y = self.origin[1] + self.cell_size[1]*0.5 + self.cell_size[1]*r[1]
            z = self.origin[2] + self.cell_size[2]*0.5 + self.cell_size[2]*r[2]
            yield((x, y, z))

    endpoint = property(lambda self: self.origin + self.cell_size*self.mesh_size)
    is_zyx = property(lambda self: self.array_order == Mesh.ZYX)
