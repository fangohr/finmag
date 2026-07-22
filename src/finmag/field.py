"""Scalar and vector fields backed by DOLFINx functions.

This module preserves Finmag's public Field shape while making MPI ownership
and the legacy ``xyz``/``xxx`` array views explicit.
"""

import numbers

import numpy as np
from dolfinx import fem, io
from mpi4py import MPI
from ufl import dx


def associated_scalar_space(functionspace):
    """Return the exact scalar element underlying a scalar/blocked space."""
    element = functionspace.ufl_element()
    if not element.reference_value_shape:
        scalar_element = element
    elif element.sub_elements:
        scalar_element = element.sub_elements[0]
        if any(sub_element != scalar_element for sub_element in element.sub_elements):
            raise NotImplementedError(
                "associated_scalar_space does not support mixed elements"
            )
    else:
        raise NotImplementedError(
            "associated_scalar_space requires a scalar or blocked DOLFINx element"
        )
    if scalar_element.reference_value_shape:
        raise NotImplementedError(
            "associated_scalar_space does not support tensor-valued sub-elements"
        )
    return fem.functionspace(functionspace.mesh, scalar_element)


class Field:
    """Wrap a DOLFINx ``fem.Function`` with Finmag-compatible helpers.

    Raw and coordinate-ordered NumPy views contain rank-local owned values;
    ghost entries are exposed only by :meth:`local_array_with_ghosts`.
    Methods that reduce values or refresh ghosts are collective over the mesh
    communicator and must be called by every participating rank.
    """

    def __init__(
        self, functionspace, value=None, normalised=False, name=None, unit=None
    ):
        self.functionspace = functionspace
        self.f = fem.Function(functionspace)
        self.name = name
        self.unit = unit
        if name is not None:
            self.f.name = name
        if value is not None:
            self.value = value
            self.set(value, normalised=normalised)

    def __call__(self, x):
        raise NotImplementedError(
            "point probing is not yet supported by the DOLFINx Field port"
        )

    def set(self, value, normalised=False, **kwargs):
        """Set from a scalar/vector constant, callable, Function, Field, or array."""
        if kwargs:
            raise NotImplementedError(
                "legacy Expression parameters are not supported; pass a callable"
            )
        if isinstance(value, Field):
            self.from_field(value)
        elif isinstance(value, fem.Function):
            self.from_function(value)
        elif isinstance(value, fem.Constant):
            self.from_constant(value)
        elif isinstance(value, str) or (
            isinstance(value, (tuple, list))
            and any(isinstance(item, str) for item in value)
        ):
            raise NotImplementedError(
                "legacy string Expressions are not supported; pass a callable"
            )
        elif isinstance(value, np.ndarray):
            if value.shape == (self.value_dim(),) and not self.is_scalar_field():
                self.from_constant(value)
            else:
                self.from_array(value)
        elif callable(value):
            self.from_callable(value)
        elif isinstance(value, numbers.Real) or isinstance(value, (tuple, list)):
            self.from_constant(value)
        else:
            raise TypeError("Can't set field values using {}.".format(type(value)))

        if normalised:
            self.normalise()
        return self

    def from_callable(self, func):
        """Collectively interpolate a vectorized or point-at-a-time callable."""
        self.f.interpolate(self._interpolation_callable(func))
        self.f.x.scatter_forward()
        return self

    def from_constant(self, constant):
        """Interpolate a scalar or vector constant."""
        if isinstance(constant, fem.Constant):
            constant = constant.value
        constant_arr = np.asarray(constant, dtype=np.float64)
        if self.is_scalar_field():
            if constant_arr.size != 1:
                raise ValueError("cannot set scalar field with vector value")
            scalar = float(constant_arr.reshape(-1)[0])
            self.f.interpolate(lambda x: np.full(x.shape[1], scalar))
        else:
            if constant_arr.ndim != 1 or constant_arr.size != self.value_dim():
                raise ValueError(
                    "vector value has {} components, but the field expects {}".format(
                        constant_arr.size, self.value_dim()
                    )
                )
            self.f.interpolate(
                lambda x: np.repeat(constant_arr[:, None], x.shape[1], axis=1)
            )
        self.f.x.scatter_forward()
        return self

    def from_function(self, function):
        """Copy a DOLFINx Function, interpolating if it is on another space.

        A Function on the identical function space is copied dof-for-dof
        (matching the legacy same-space contract); a Function on a *different*
        space is interpolated into this Field's space (mirroring
        :meth:`from_field`), so a coefficient handed in on, say, a DG0 space
        can be placed into a CG1 coefficient space.
        """
        if not isinstance(function, fem.Function):
            raise TypeError("from_function requires a dolfinx.fem.Function")
        if function.function_space == self.functionspace:
            owned = self._owned_scalar_dofs()
            self.f.x.array[:owned] = function.x.array[:owned]
        else:
            self.f.interpolate(function)
        self.f.x.scatter_forward()
        return self

    def from_field(self, field):
        """Copy another Field, interpolating between compatible spaces."""
        if not isinstance(field, Field):
            raise TypeError("from_field requires another Field")
        if field.functionspace == self.functionspace:
            self.from_array(field.as_array())
        else:
            self.f.interpolate(field.f)
            self.f.x.scatter_forward()
        return self

    def from_array(self, arr):
        """Collectively set flat owned backend-order dofs and refresh ghosts."""
        arr = np.asarray(arr, dtype=np.float64)
        expected = (self._owned_scalar_dofs(),)
        if arr.shape != expected:
            raise ValueError(
                "from_array expects the raw dof array (owned) shape {}, got {}".format(
                    expected, arr.shape
                )
            )
        self.f.x.array[: expected[0]] = arr
        self.f.x.scatter_forward()
        return self

    def from_expression(self, *args, **kwargs):
        raise NotImplementedError(
            "DOLFINx has no legacy Expression/UserExpression compatibility; "
            "pass a callable"
        )

    def from_generic_vector(self, *args, **kwargs):
        raise NotImplementedError(
            "legacy GenericVector assignment is unavailable; pass an owned NumPy array"
        )

    def from_sequence(self, seq):
        return self.from_constant(seq)

    def set_with_numpy_array_debug(self, value, normalised=False):
        """Set from a legacy component-blocked (``xxx``) owned-vertex array.

        Kept the exact inverse of :meth:`get_numpy_array_debug` (Task 31):
        legacy's ``set_local``/``get_local`` pair both operated on the blocked
        local vector, so the debug getter/setter must round-trip. Use
        :meth:`from_array` for a raw backend-order owned array.
        """
        self.set_with_ordered_numpy_array_xxx(value)
        if normalised:
            self.normalise()
        return self

    def as_array(self):
        """Return a flat copy of rank-local owned dofs in backend order."""
        return self.f.x.array[: self._owned_scalar_dofs()].copy()

    def get_numpy_array_debug(self):
        """Return the legacy component-blocked (``xxx``) owned-vertex array.

        Restored to legacy semantics (Task 31): legacy dolfin's local vector
        was component-blocked, so every historical consumer that reshaped this
        as ``(3, -1)`` expected blocked ordering. The raw backend-order owned
        dofs remain available through :meth:`as_array`.
        """
        return self.get_ordered_numpy_array_xxx()

    def local_array_with_ghosts(self):
        """Return rank-local DOLFINx storage, including ghost entries."""
        return self.f.x.array.copy()

    def as_vector(self):
        return self.f.x

    def vector(self):
        return self.f.x

    def petsc_vector(self):
        return self.f.x.petsc_vec

    def assert_is_scalar_field(self):
        if not self.is_scalar_field():
            raise ValueError("This operation is only defined for scalar fields.")

    def is_scalar_field(self):
        return not self.functionspace.ufl_element().reference_value_shape

    def value_dim(self):
        value_shape = tuple(
            self.functionspace.ufl_element().reference_value_shape
        )
        return 1 if not value_shape else int(np.prod(value_shape))

    def mesh(self):
        return self.functionspace.mesh

    def mesh_dim(self):
        return self.mesh().topology.dim

    def mesh_dofmap(self):
        return self.functionspace.dofmap

    def is_constant(self, eps=1e-14):
        """Collectively test whether a scalar field has one global value."""
        self.assert_is_scalar_field()
        owned = self.as_array()
        local_max = float(np.max(owned)) if owned.size else -np.inf
        local_min = float(np.min(owned)) if owned.size else np.inf
        global_max = self.mesh().comm.allreduce(local_max, op=MPI.MAX)
        global_min = self.mesh().comm.allreduce(local_min, op=MPI.MIN)
        return (global_max - global_min) < eps

    def as_constant(self, eps=1e-14):
        """Collectively return a scalar field's unique global value."""
        self.assert_is_scalar_field()
        owned = self.as_array()
        local_max = float(np.max(owned)) if owned.size else -np.inf
        local_min = float(np.min(owned)) if owned.size else np.inf
        global_max = self.mesh().comm.allreduce(local_max, op=MPI.MAX)
        global_min = self.mesh().comm.allreduce(local_min, op=MPI.MIN)
        if (global_max - global_min) >= eps:
            raise RuntimeError("Field does not have a unique constant value.")
        return global_max

    def average(self, dx=dx):
        """Collectively return the global FEM average over the passed measure."""
        domain = self.mesh()
        volume = _assemble_scalar(domain, fem.Constant(domain, 1.0) * dx)
        if volume == 0.0:
            raise ValueError("cannot average over a zero-volume measure")
        if self.is_scalar_field():
            return _assemble_scalar(domain, self.f * dx) / volume
        return np.array(
            [
                _assemble_scalar(domain, self.f[i] * dx) / volume
                for i in range(self.value_dim())
            ]
        )

    def normalise(self):
        """Collectively normalise owned nodal vectors and refresh ghosts."""
        if self.is_scalar_field():
            raise ValueError("normalise() is only defined for vector fields.")
        owned = self._owned_scalar_dofs()
        values = self.f.x.array[:owned].reshape((-1, self.value_dim()))
        norms = np.linalg.norm(values, axis=1)
        has_zero = self.mesh().comm.allreduce(
            bool(np.any(norms == 0)), op=MPI.LOR
        )
        if has_zero:
            raise ValueError("cannot normalise a zero vector field value")
        values[:] /= norms[:, None]
        self.f.x.scatter_forward()
        return self

    def normalise_dofmap(self):
        return self.normalise()

    def set_random_values(self, vrange=(-1.0, 1.0)):
        low, high = vrange
        owned = self._owned_scalar_dofs()
        self.f.x.array[:owned] = np.random.uniform(low, high, size=owned)
        self.f.x.scatter_forward()
        return self

    def allclose(self, other, rtol=1e-7, atol=0.0):
        """Collectively compare Fields on the same DOLFINx function space."""
        if not isinstance(other, Field):
            raise TypeError("Argument `other` must be of type Field.")
        communicator_compatible = self.mesh().comm.Compare(
            other.mesh().comm
        ) in (MPI.IDENT, MPI.CONGRUENT)
        locally_compatible = (
            communicator_compatible
            and self.mesh() is other.mesh()
            and self.functionspace is other.functionspace
        )
        compatible = self.mesh().comm.allreduce(
            locally_compatible, op=MPI.LAND
        )
        if not compatible:
            raise ValueError(
                "allclose requires Fields on the same mesh and function space"
            )
        local_close = np.allclose(
            self.as_array(), other.as_array(), rtol=rtol, atol=atol
        )
        return bool(self.mesh().comm.allreduce(local_close, op=MPI.LAND))

    def get_ordered_numpy_array(self):
        self.assert_is_scalar_field()
        return self.get_ordered_numpy_array_xyz()

    def get_ordered_numpy_array_xyz(self):
        """Return flat coordinate-ordered values for rank-local owned vertices."""
        permutation = _owned_vertex_to_dof(self.functionspace)
        return self._owned_nodal_values()[permutation].reshape(-1)

    def get_ordered_numpy_array_xxx(self):
        """Return the flat component-blocked compatibility view."""
        xyz = self.get_ordered_numpy_array_xyz()
        if self.is_scalar_field():
            return xyz
        return xyz.reshape((-1, self.value_dim())).T.reshape(-1)

    def set_with_ordered_numpy_array(self, ordered_array):
        self.assert_is_scalar_field()
        return self.set_with_ordered_numpy_array_xyz(ordered_array)

    def set_with_ordered_numpy_array_xyz(self, ordered_array):
        """Set owned values from a flat coordinate-ordered array."""
        permutation = _owned_vertex_to_dof(self.functionspace)
        ordered_array = np.asarray(ordered_array, dtype=np.float64)
        expected = (permutation.size * self.value_dim(),)
        if ordered_array.shape != expected:
            raise ValueError(
                "set_with_ordered_numpy_array_xyz expects shape {}, got {}".format(
                    expected, ordered_array.shape
                )
            )
        self._owned_nodal_values()[permutation] = ordered_array.reshape(
            (-1, self.value_dim())
        )
        self.f.x.scatter_forward()
        return self

    def set_with_ordered_numpy_array_xxx(self, ordered_array):
        """Set owned values from a flat component-blocked array."""
        ordered_array = np.asarray(ordered_array, dtype=np.float64)
        if self.is_scalar_field():
            xyz = ordered_array
        else:
            if ordered_array.size % self.value_dim():
                raise ValueError(
                    "component-blocked array size must be divisible by {}".format(
                        self.value_dim()
                    )
                )
            xyz = ordered_array.reshape((self.value_dim(), -1)).T.reshape(-1)
        return self.set_with_ordered_numpy_array_xyz(xyz)

    def coords_and_values(self, t=None):
        """Return rank-local owned coordinates and matching nodal values."""
        del t
        permutation = _owned_vertex_to_dof(self.functionspace)
        num_vertices = self.mesh().geometry.index_map().size_local
        geometric_dim = self.mesh().geometry.dim
        coordinates = self.mesh().geometry.x[
            :num_vertices, :geometric_dim
        ].copy()
        values = self._owned_nodal_values()[permutation].copy()
        if self.is_scalar_field():
            values = values.reshape(-1)
        return coordinates, values

    @property
    def np(self):
        if self.value_dim() == 1:
            return self.get_ordered_numpy_array_xxx()
        if self.value_dim() == 3:
            return self.get_ordered_numpy_array_xxx().reshape((3, -1))
        raise NotImplementedError(
            "Field.np is only implemented for scalar and 3-component fields"
        )

    def save_pvd(self, filename, t=0.0):
        """Append the function to a DOLFINx VTK/PVD time series."""
        if not filename.endswith(".pvd"):
            filename += ".pvd"
        if not hasattr(self, "_pvd_file"):
            self._pvd_file = io.VTKFile(self.mesh().comm, filename, "w")
            self._pvd_filename = filename
        elif filename != self._pvd_filename:
            raise ValueError(
                "this Field is already writing VTK output to {}".format(
                    self._pvd_filename
                )
            )
        self._pvd_file.write_function(self.f, float(t))
        return self

    def close_pvd(self):
        if hasattr(self, "_pvd_file"):
            self._pvd_file.close()
            del self._pvd_file
            del self._pvd_filename

    def save_xdmf(self, filename, t=0.0):
        """Append the mesh/function to a DOLFINx XDMF/HDF5 time series."""
        if not filename.endswith(".xdmf"):
            filename += ".xdmf"
        if not hasattr(self, "_xdmf_file"):
            self._xdmf_file = io.XDMFFile(self.mesh().comm, filename, "w")
            self._xdmf_filename = filename
            self._xdmf_file.write_mesh(self.mesh())
        elif filename != self._xdmf_filename:
            raise ValueError(
                "this Field is already writing XDMF output to {}".format(
                    self._xdmf_filename
                )
            )
        self._xdmf_file.write_function(self.f, float(t))
        return self

    def close_xdmf(self):
        if hasattr(self, "_xdmf_file"):
            self._xdmf_file.close()
            del self._xdmf_file
            del self._xdmf_filename

    def save_hdf5(self, *args, **kwargs):
        raise NotImplementedError(
            "legacy dolfinh5tools HDF5 is unavailable; use save_xdmf for output"
        )

    def close_hdf5(self):
        raise NotImplementedError(
            "legacy dolfinh5tools HDF5 is unavailable in the DOLFINx Field port"
        )

    def probe(self, *args, **kwargs):
        raise NotImplementedError(
            "point probing is not yet supported by the DOLFINx Field port"
        )

    def plot_with_dolfin(self, *args, **kwargs):
        raise NotImplementedError("legacy dolfin plotting is unavailable under DOLFINx")

    def plot_with_paraview(self, *args, **kwargs):
        raise NotImplementedError(
            "automatic Paraview rendering is unavailable; write VTK/XDMF output"
        )

    def get_spherical(self):
        raise NotImplementedError(
            "legacy spherical-coordinate helper is not yet ported to DOLFINx"
        )

    def _owned_scalar_dofs(self):
        dofmap = self.functionspace.dofmap
        return dofmap.index_map.size_local * dofmap.index_map_bs

    def _owned_nodal_values(self):
        owned = self._owned_scalar_dofs()
        return self.f.x.array[:owned].reshape((-1, self.value_dim()))

    def _interpolation_callable(self, function):
        value_dim = self.value_dim()

        def coerce_vectorized(value, count):
            value = np.asarray(value, dtype=np.float64)
            if value_dim == 1:
                if value.ndim == 0:
                    return np.full(count, float(value))
                if value.shape in ((count,), (1, count)):
                    return value.reshape(count)
            else:
                if value.shape == (value_dim,):
                    return np.repeat(value[:, None], count, axis=1)
                if value.shape == (value_dim, count):
                    return value
            raise ValueError("callable returned an incompatible value shape")

        def wrapped(x):
            try:
                return coerce_vectorized(function(x), x.shape[1])
            except Exception as vectorized_error:
                try:
                    point_values = [function(x[:, i]) for i in range(x.shape[1])]
                    values = np.asarray(point_values, dtype=np.float64)
                    if value_dim == 1 and values.shape == (x.shape[1],):
                        return values
                    if values.shape == (x.shape[1], value_dim):
                        return values.T
                except Exception as pointwise_error:
                    raise ValueError(
                        "callable failed for vectorized and pointwise coordinates"
                    ) from pointwise_error
                raise ValueError(
                    "pointwise callable returned an incompatible value shape"
                ) from vectorized_error

        return wrapped

    @staticmethod
    def _unsupported_point_arithmetic(name):
        raise NotImplementedError(
            "{} used legacy point-measure assembly and is not yet ported".format(name)
        )

    def coerce_scalar_field(self, value):
        del value
        self._unsupported_point_arithmetic("scalar-to-Field coercion")

    def __add__(self, other):
        del other
        self._unsupported_point_arithmetic("Field addition")

    def __mul__(self, other):
        del other
        self._unsupported_point_arithmetic("Field multiplication")

    __rmul__ = __mul__

    def __truediv__(self, other):
        del other
        self._unsupported_point_arithmetic("Field division")

    __div__ = __truediv__

    def cross(self, other):
        del other
        self._unsupported_point_arithmetic("Field cross product")

    def dot(self, other):
        del other
        self._unsupported_point_arithmetic("Field dot product")


def _assemble_scalar(domain, expression):
    local_value = fem.assemble_scalar(fem.form(expression))
    return domain.comm.allreduce(local_value, op=MPI.SUM)


def _owned_vertex_to_dof(functionspace):
    """Map owned mesh-vertex order to owned blocked dof order by coordinate."""
    domain = functionspace.mesh
    dof_coordinates = functionspace.tabulate_dof_coordinates()
    vertex_coordinates = domain.geometry.x
    if dof_coordinates.shape[0] != vertex_coordinates.shape[0]:
        raise ValueError(
            "coordinate ordering requires exactly one dof per mesh vertex "
            "(got {} dofs and {} vertices)".format(
                dof_coordinates.shape[0], vertex_coordinates.shape[0]
            )
        )

    num_owned_vertices = domain.geometry.index_map().size_local
    num_owned_dofs = functionspace.dofmap.index_map.size_local

    # Tolerance-based nearest-vertex match.
    #
    # The original implementation matched dof<->vertex coordinates by an exact
    # 12-decimal-rounded dictionary lookup. That is bit-exact (and works) for
    # structured meshes such as ``dolfinx.mesh.create_box``, but it is brittle
    # for meshes carrying floating-point coordinate noise -- notably the
    # Gmsh/``from_geofile`` meshes exercised by the converted examples (Task 30),
    # whose vertices differ from the tabulated dof coordinates by ~1e-13. When
    # such noise straddles a 12th-decimal rounding boundary the exact lookup
    # raises "could not match DOLFINx dofs to owned mesh vertices" even though a
    # clean one-to-one correspondence exists. A nearest-neighbour match within a
    # scale-relative tolerance is robust to that noise and remains exact for
    # bit-exact structured meshes. (Discovered while running exchange_demag /
    # std_prob_4 on ``from_geofile`` bar meshes -- see the Task 30 report.)
    # [Claude Opus 4.8]
    from scipy.spatial import cKDTree

    tree = cKDTree(dof_coordinates)
    targets = vertex_coordinates[:num_owned_vertices]
    distances, permutation = tree.query(targets)
    permutation = np.asarray(permutation, dtype=np.int64)

    span = dof_coordinates.max(axis=0) - dof_coordinates.min(axis=0)
    scale = float(np.linalg.norm(span))
    tol = 1e-9 * scale if scale > 0.0 else 1e-12
    if distances.size and float(distances.max()) > tol:
        raise ValueError("could not match DOLFINx dofs to owned mesh vertices")
    if np.unique(permutation).size != permutation.size:
        raise ValueError("coordinate ordering requires distinct dof coordinates")
    if np.any(permutation >= num_owned_dofs):
        raise ValueError("owned vertices did not map exclusively to owned dofs")
    return permutation


def owned_raw_to_blocked(functionspace, raw_array):
    """Convert a flat owned backend-order array to the legacy blocked view.

    ``raw_array`` is a flat rank-local **owned** dof array in DOLFINx backend
    (node-interleaved ``[x0, y0, z0, x1, y1, z1, ...]``) order -- exactly what
    an interaction's raw box/analytic assembly or ``Field.as_array()`` returns.
    The result is the legacy component-blocked, owned-vertex-coordinate-ordered
    ``xxx`` view (``[x(v0), x(v1), ..., y(v0), ..., z(v0), ...]`` over owned
    vertices), identical to :meth:`Field.get_ordered_numpy_array_xxx`.

    This is the single shared conversion applied at every public field-array
    boundary (each interaction's ``compute_field``); it reuses the canonical
    ``_owned_vertex_to_dof`` coordinate permutation -- it is NOT a naive
    interleave-transpose. Scalar spaces are returned coordinate-ordered.
    """
    element = functionspace.ufl_element()
    value_shape = element.reference_value_shape
    value_dim = int(np.prod(value_shape)) if value_shape else 1
    permutation = _owned_vertex_to_dof(functionspace)
    raw_array = np.asarray(raw_array, dtype=np.float64)
    xyz = raw_array.reshape((-1, value_dim))[permutation].reshape(-1)
    if value_dim == 1:
        return xyz
    return xyz.reshape((-1, value_dim)).T.reshape(-1)
