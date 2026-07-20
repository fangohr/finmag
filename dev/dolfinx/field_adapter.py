"""Small DOLFINx-backed field adapter for compatibility exploration.

This module probes how much of the legacy ``finmag.field.Field`` shape can be
preserved with DOLFINx. It intentionally lives in ``dev/dolfinx`` and covers a
narrow but broader subset than the first version: construction, constant/
callable/field/function assignment, scalar vs vector inspection, nodal values,
coordinate/value ordering, volume averages, per-node normalisation, constant
detection, and basic XDMF/VTK file output. It is not a production replacement
for ``finmag.field.Field`` yet, and several legacy behaviours (Paraview
plotting, the point-measure-hack arithmetic operators, `dolfinh5tools`-format
HDF5) are intentionally out of scope; see ``porting_map.md``. The direct
production implementation now lives in ``src/finmag/field.py``; this adapter
remains a frozen witness and is not an alternate package implementation.
[Codex GPT-5.6]
"""

import numpy as np
from dolfinx import fem, io
from mpi4py import MPI
from ufl import dx


class DOLFINxField:
    """Prototype DOLFINx adapter for the first legacy ``Field`` behaviours."""

    def __init__(
        self, function_space, value=None, normalised=False, name=None, unit=None
    ):
        self.functionspace = function_space
        self.f = fem.Function(function_space)
        self.name = name
        self.unit = unit
        if name is not None:
            self.f.name = name
        if value is not None:
            self.set(value, normalised=normalised)

    def value_dim(self):
        """Return scalar dimension ``1`` or vector component count."""
        value_shape = tuple(self.functionspace.element.value_shape)
        if not value_shape:
            return 1
        return int(np.prod(value_shape))

    def is_scalar_field(self):
        """Return whether the wrapped DOLFINx function is scalar-valued."""
        return self.value_dim() == 1

    def mesh(self):
        """Return the DOLFINx mesh associated with this field."""
        return self.functionspace.mesh

    def mesh_dim(self):
        """Return the topological dimension of the underlying mesh."""
        return self.mesh().topology.dim

    def set(self, value, normalised=False):
        """Set field values from a constant, callable, field, or function.

        Dispatches on ``value``'s type, mirroring legacy ``Field.set``'s
        ``from_*`` dispatch: another ``DOLFINxField`` or ``fem.Function`` is
        copied/interpolated via ``from_field``/``from_function``, a callable
        is interpolated directly (the DOLFINx equivalent of legacy's
        ``from_expression``/``from_callable``), and anything else is treated
        as a constant scalar/vector value. [GitHub Copilot / Claude Sonnet 5]
        """
        if isinstance(value, DOLFINxField):
            self.from_field(value)
        elif isinstance(value, fem.Function):
            self.from_function(value)
        elif callable(value):
            self.f.interpolate(value)
        else:
            self._set_constant(value, normalised=normalised)
        self.f.x.scatter_forward()
        return self

    def as_array(self):
        """Return a copy of the owned local scalar dofs, excluding ghosts.

        Legacy ``GenericVector.get_local()`` returned the process-owned range.
        DOLFINx ``Function.x.array`` also appends ghost blocks, so exposing the
        whole array here would silently change the driver state size on MPI
        runs. [Codex GPT-5.6]
        """
        return self.f.x.array[: self._owned_scalar_dofs()].copy()

    def from_function(self, function):
        """Set values directly from a ``dolfinx.fem.Function``.

        The function must be defined on the same function space; unlike
        legacy ``Field.from_function``, this does not attempt to interpolate
        between different spaces (use ``from_field`` for that, via another
        ``DOLFINxField``). [GitHub Copilot / Claude Sonnet 5]
        """
        if function.function_space != self.functionspace:
            raise ValueError(
                "from_function requires a function on the same function "
                "space; use from_field to interpolate between spaces"
            )
        owned = self._owned_scalar_dofs()
        self.f.x.array[:owned] = function.x.array[:owned]
        self.f.x.scatter_forward()
        return self

    def from_field(self, field):
        """Set values from another ``DOLFINxField``, interpolating if needed.

        Mirrors legacy ``Field.from_field``: a direct dof-array copy when the
        function spaces match, otherwise DOLFINx interpolation between
        spaces. [GitHub Copilot / Claude Sonnet 5]
        """
        if not isinstance(field, DOLFINxField):
            raise TypeError("from_field requires another DOLFINxField")
        if field.functionspace == self.functionspace:
            owned = self._owned_scalar_dofs()
            self.f.x.array[:owned] = field.f.x.array[:owned]
        else:
            self.f.interpolate(field.f)
        self.f.x.scatter_forward()
        return self

    def nodal_values(self):
        """Return owned nodal dofs, excluding ghost copies."""
        owned = self.as_array()
        if self.is_scalar_field():
            return owned
        return owned.reshape((-1, self.value_dim()))

    def local_values_with_ghosts(self):
        """Return local nodal storage including read-only-style ghost copies."""
        values = self.f.x.array.copy()
        if self.is_scalar_field():
            return values
        return values.reshape((-1, self.value_dim()))

    def average(self):
        """Return the volume average, matching the legacy ``Field.average`` idea."""
        domain = self.mesh()
        volume = _assemble_scalar(domain, fem.Constant(domain, 1.0) * dx)
        if self.is_scalar_field():
            return _assemble_scalar(domain, self.f * dx) / volume
        return np.array([
            _assemble_scalar(domain, self.f[i] * dx) / volume
            for i in range(self.value_dim())
        ])

    def set_random_values(self, vrange=(-1.0, 1.0)):
        """Fill the field with uniform random values, useful for debugging.

        Each owned raw dof is drawn independently and uniformly from
        ``vrange``; ghosts are then refreshed from their owners. [Codex
        GPT-5.6]
        """
        low, high = vrange
        owned = self._owned_scalar_dofs()
        values = np.random.uniform(low, high, size=owned)
        self.f.x.array[:owned] = values
        self.f.x.scatter_forward()
        return self

    def is_constant(self, eps=1e-14):
        """Return whether the field has a unique constant value everywhere.

        Uses the MPI-reduced global max/min of the raw dof array, matching
        legacy ``Field.is_constant`` (which is only defined for scalar
        fields). [GitHub Copilot / Claude Sonnet 5]
        """
        self._assert_is_scalar_field()
        domain = self.mesh()
        owned = self.as_array()
        local_max = float(np.max(owned)) if owned.size else -np.inf
        local_min = float(np.min(owned)) if owned.size else np.inf
        global_max = domain.comm.allreduce(local_max, op=MPI.MAX)
        global_min = domain.comm.allreduce(local_min, op=MPI.MIN)
        return (global_max - global_min) < eps

    def as_constant(self, eps=1e-14):
        """Return the field's unique constant value, or raise if non-constant."""
        self._assert_is_scalar_field()
        domain = self.mesh()
        owned = self.as_array()
        local_max = float(np.max(owned)) if owned.size else -np.inf
        local_min = float(np.min(owned)) if owned.size else np.inf
        global_max = domain.comm.allreduce(local_max, op=MPI.MAX)
        global_min = domain.comm.allreduce(local_min, op=MPI.MIN)
        if (global_max - global_min) >= eps:
            raise RuntimeError("Field does not have a unique constant value.")
        return global_max

    def _assert_is_scalar_field(self):
        if not self.is_scalar_field():
            raise ValueError("This operation is only defined for scalar fields.")

    def normalise(self):
        """Normalise the field so every nodal vector has unit length.

        Legacy ``Field.normalise`` divides by a norm field assembled via
        Claas Abert's "point measure hack" (``TestFunction``/``dP``
        assembly). This adapter instead normalises the raw per-node nodal
        array directly, which is simpler in DOLFINx and gives the same
        per-node unit-norm result for vector Lagrange-1 fields; only the
        implementation approach differs from legacy, not the outcome.
        [GitHub Copilot / Claude Sonnet 5]
        """
        if self.is_scalar_field():
            raise ValueError("normalise() is only defined for vector fields.")
        owned = self._owned_scalar_dofs()
        values = self.f.x.array[:owned].reshape((-1, self.value_dim()))
        norms = np.linalg.norm(values, axis=1)
        local_has_zero = bool(np.any(norms == 0))
        has_zero = self.mesh().comm.allreduce(local_has_zero, op=MPI.LOR)
        if has_zero:
            raise ValueError("cannot normalise a zero vector field value")
        values[:] = values / norms[:, None]
        self.f.x.scatter_forward()
        return self

    def coords_and_values(self):
        """Return ``(coordinates, values)`` for Lagrange-1 nodal dofs.

        Matches legacy ``Field.coords_and_values``, restricted to the
        one-dof-per-vertex case already required by
        ``get_ordered_numpy_array_xyz``. On MPI ranks it returns owned rows
        only, so callers can gather without duplicating ghosts. [Codex GPT-5.6]
        """
        permutation = _vertex_order_permutation(self.functionspace)
        num_owned_vertices = self.mesh().geometry.index_map().size_local
        geometric_dim = self.mesh().geometry.dim
        coords = self.mesh().geometry.x[
            :num_owned_vertices, :geometric_dim
        ].copy()
        values = self.nodal_values()[permutation]
        return coords, values

    def allclose(self, other, rtol=1e-7, atol=0.0):
        """Return whether two fields are element-wise equal within tolerance.

        Matching owned partitions are compared locally and then reduced with
        MPI logical-AND, excluding duplicate ghost values. [Codex GPT-5.6]
        """
        if not isinstance(other, DOLFINxField):
            raise TypeError("allclose requires another DOLFINxField")
        local_close = np.allclose(
            self.as_array(), other.as_array(), rtol=rtol, atol=atol
        )
        return bool(self.mesh().comm.allreduce(local_close, op=MPI.LAND))

    def save_pvd(self, filename, t=0.0):
        """Write the field to a legacy-analogous ``.pvd``/``.vtu`` time series.

        Uses ``dolfinx.io.VTKFile``, the direct DOLFINx counterpart of
        legacy ``Field.save_pvd`` (``df.File(filename) << self.f``). The
        file handle is kept open across calls (like legacy's hdf5
        open/write pattern) so repeated calls append time steps; call
        ``close_pvd()`` when done. [GitHub Copilot / Claude Sonnet 5]
        """
        if not filename.endswith(".pvd"):
            filename += ".pvd"
        if not hasattr(self, "_pvd_file"):
            self._pvd_file = io.VTKFile(self.mesh().comm, filename, "w")
        self._pvd_file.write_function(self.f, float(t))
        return self

    def close_pvd(self):
        """Close the ``.pvd`` file handle opened by ``save_pvd``."""
        if hasattr(self, "_pvd_file"):
            self._pvd_file.close()
            del self._pvd_file

    def save_xdmf(self, filename, t=0.0):
        """Write the field's mesh and values to an XDMF/HDF5 pair.

        This is the closest DOLFINx-native counterpart to legacy
        ``Field.save_hdf5``, but it is a genuinely different, DOLFINx-native
        format: it uses ``dolfinx.io.XDMFFile`` rather than the external
        ``dolfinh5tools`` package/format, and is write-only here. This
        DOLFINx version (0.10.0) has no ``XDMFFile.read_function``, and
        reading a checkpointed function back would need extra tooling
        (e.g. the external ``adios4dolfinx`` package) not present in this
        environment, so no round-trip read is provided. The written
        ``.xdmf``/``.h5`` files are directly viewable in Paraview. [GitHub
        Copilot / Claude Sonnet 5]
        """
        if not filename.endswith(".xdmf"):
            filename += ".xdmf"
        if not hasattr(self, "_xdmf_file"):
            self._xdmf_file = io.XDMFFile(self.mesh().comm, filename, "w")
            self._xdmf_file.write_mesh(self.mesh())
        self._xdmf_file.write_function(self.f, float(t))
        return self

    def close_xdmf(self):
        """Close the XDMF file handle opened by ``save_xdmf``."""
        if hasattr(self, "_xdmf_file"):
            self._xdmf_file.close()
            del self._xdmf_file

    def from_array(self, arr):
        """Set the raw owned local dof array directly.

        This is the DOLFINx analogue of legacy ``Field.from_array``: it
        assigns the process-owned dofs as-is, without any mesh-vertex
        reordering, then refreshes ghosts. Use
        ``set_with_ordered_numpy_array_xyz`` if the input is ordered by mesh
        vertex instead. [Codex GPT-5.6]
        """
        arr = np.asarray(arr, dtype=np.float64)
        owned = self._owned_scalar_dofs()
        expected_shape = (owned,)
        if arr.shape != expected_shape:
            raise ValueError(
                "from_array expects the raw dof array (owned) shape %s, got %s"
                % (expected_shape, arr.shape)
            )
        self.f.x.array[:owned] = arr
        self.f.x.scatter_forward()
        return self

    def get_ordered_numpy_array_xyz(self):
        """Return a flat owned-local ``[x1, y1, ..., xN, yN, ...]`` array.

        Legacy dolfin's raw vector layout for vector Lagrange spaces is
        component-blocked ("xxx": all x-components, then all y, then all z)
        and unrelated to mesh vertex order, so ``Field`` built explicit
        ``v2d``/``d2v`` permutation maps to recover a vertex-ordered,
        per-node-interleaved ("xyz") view. DOLFINx's raw dof array for a
        blocked vector Lagrange-1 space is already per-node interleaved, but
        dof numbering is still not guaranteed to match mesh vertex numbering
        (e.g. under dof reordering or in parallel), so this builds the
        permutation explicitly from dof/vertex coordinates rather than
        assuming index equality. On MPI ranks this view contains owned mesh
        vertices only; ghost copies are excluded. [Codex GPT-5.6]
        """
        permutation = _vertex_order_permutation(self.functionspace)
        return self.nodal_values()[permutation].reshape(-1)

    def get_ordered_numpy_array_xxx(self):
        """Return the legacy flat component-blocked compatibility view.

        ``xxx`` is not a DOLFINx storage layout. It is an explicit conversion
        from coordinate-ordered ``xyz`` and remains required by the existing
        LLG, Sundials, and NEB array APIs. [Codex GPT-5.6]
        """
        xyz = self.get_ordered_numpy_array_xyz()
        if self.is_scalar_field():
            return xyz
        return xyz.reshape((-1, self.value_dim())).T.reshape(-1)

    def set_with_ordered_numpy_array_xyz(self, ordered_array):
        """Set field values from an array ordered to match mesh vertices.

        See ``get_ordered_numpy_array_xyz`` for the ordering convention and
        its relationship to legacy ``Field.set_with_ordered_numpy_array_xyz``.
        [GitHub Copilot / Claude Sonnet 5]
        """
        permutation = _vertex_order_permutation(self.functionspace)
        ordered_array = np.asarray(ordered_array, dtype=np.float64)
        expected_shape = (permutation.size * self.value_dim(),)
        if ordered_array.shape != expected_shape:
            raise ValueError(
                "set_with_ordered_numpy_array_xyz expects shape %s, got %s"
                % (expected_shape, ordered_array.shape)
            )

        owned = self._owned_scalar_dofs()
        owned_values = self.f.x.array[:owned].reshape(
            (-1, self.value_dim())
        )
        owned_values[permutation] = ordered_array.reshape(
            (-1, self.value_dim())
        )
        self.f.x.scatter_forward()
        return self

    def set_with_ordered_numpy_array_xxx(self, ordered_array):
        """Set owned values from the flat component-blocked compatibility view."""
        ordered_array = np.asarray(ordered_array, dtype=np.float64)
        if self.is_scalar_field():
            xyz = ordered_array
        else:
            if ordered_array.size % self.value_dim() != 0:
                raise ValueError(
                    "component-blocked array size must be divisible by %d"
                    % self.value_dim()
                )
            xyz = ordered_array.reshape((self.value_dim(), -1)).T.reshape(-1)
        return self.set_with_ordered_numpy_array_xyz(xyz)

    def _owned_scalar_dofs(self):
        """Return the number of owned scalar entries in ``Function.x.array``."""
        dofmap = self.functionspace.dofmap
        return dofmap.index_map.size_local * dofmap.index_map_bs

    def _set_constant(self, value, normalised=False):
        """Interpolate a scalar or vector constant into the wrapped function."""
        constant = np.asarray(value, dtype=np.float64)
        if self.is_scalar_field():
            if constant.size != 1:
                raise ValueError("cannot set scalar field with vector value")
            scalar = float(constant.reshape(-1)[0])
            self.f.interpolate(lambda x: np.full(x.shape[1], scalar))
            return

        if constant.ndim != 1 or constant.size != self.value_dim():
            raise ValueError(
                "vector value has %d components, but the field expects %d"
                % (constant.size, self.value_dim())
            )
        if normalised:
            norm = np.linalg.norm(constant)
            if norm == 0:
                raise ValueError("cannot normalise zero vector field value")
            constant = constant / norm

        self.f.interpolate(
            lambda x: np.repeat(constant[:, None], x.shape[1], axis=1)
        )


def _assemble_scalar(domain, form):
    """Assemble and MPI-reduce a scalar DOLFINx form."""
    local_value = fem.assemble_scalar(fem.form(form))
    return domain.comm.allreduce(local_value, op=MPI.SUM)


def _vertex_order_permutation(function_space):
    """Return the permutation mapping mesh-vertex order to nodal dof order.

    ``result[i]`` is the (blocked) dof index whose coordinates match mesh
    vertex ``i``, so ``nodal_values()[result]`` is ordered like
    ``mesh.geometry.x``. Matching is done by coordinate, not by assuming dof
    index equals vertex index, since that is not guaranteed in general (e.g.
    parallel runs or dof reordering). Only Lagrange-1-like spaces with
    exactly one (blocked) dof per mesh vertex are supported; anything else
    (e.g. DG0) raises explicitly.

    Coordinates are matched via a rounded-coordinate dictionary rather than
    lexicographic sorting: an earlier lexsort-based implementation broke
    ties inconsistently between the two coordinate arrays whenever multiple
    points shared a leading coordinate (e.g. a whole mesh column sharing the
    same x value), even though the unsorted arrays already matched row for
    row. [GitHub Copilot / Claude Sonnet 5]
    """
    domain = function_space.mesh
    dof_coords = function_space.tabulate_dof_coordinates()
    vertex_coords = domain.geometry.x

    if dof_coords.shape[0] != vertex_coords.shape[0]:
        raise ValueError(
            "coordinate/value ordering is only supported for function spaces "
            "with exactly one dof per mesh vertex (got %d dofs and %d "
            "vertices)" % (dof_coords.shape[0], vertex_coords.shape[0])
        )

    dof_index_by_key = {}
    for dof_index, point in enumerate(np.round(dof_coords, decimals=9)):
        key = tuple(point)
        if key in dof_index_by_key:
            raise ValueError(
                "coordinate/value ordering requires distinct dof "
                "coordinates; found duplicate coordinate %s" % (key,)
            )
        dof_index_by_key[key] = dof_index

    num_owned_vertices = domain.geometry.index_map().size_local
    num_owned_dofs = function_space.dofmap.index_map.size_local
    vertex_to_dof = np.empty(num_owned_vertices, dtype=np.int64)
    owned_vertex_coords = vertex_coords[:num_owned_vertices]
    for vertex_index, point in enumerate(np.round(owned_vertex_coords, decimals=9)):
        key = tuple(point)
        if key not in dof_index_by_key:
            raise ValueError(
                "could not match DOLFINx dofs to mesh vertices by "
                "coordinate; coordinate/value ordering is not supported for "
                "this space"
            )
        vertex_to_dof[vertex_index] = dof_index_by_key[key]

    if np.any(vertex_to_dof >= num_owned_dofs):
        raise ValueError(
            "owned mesh vertices do not map exclusively to owned dofs"
        )

    return vertex_to_dof
