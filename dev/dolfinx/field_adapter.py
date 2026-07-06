"""Small DOLFINx-backed field adapter for compatibility exploration.

This module probes how much of the legacy ``finmag.field.Field`` shape can be
preserved with DOLFINx. It intentionally lives in ``dev/dolfinx`` and only
covers a narrow subset: construction, constant/callable assignment, scalar vs
vector inspection, nodal values, and volume averages. It is not a production
replacement for ``finmag.field.Field`` yet. [Codex gpt-5.5 high]
"""

import numpy as np
from dolfinx import fem
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

    def set(self, value, normalised=False):
        """Set field values from a constant or DOLFINx-compatible callable."""
        if callable(value):
            self.f.interpolate(value)
        else:
            self._set_constant(value, normalised=normalised)
        self.f.x.scatter_forward()
        return self

    def as_array(self):
        """Return a copy of the underlying DOLFINx local dof array."""
        return self.f.x.array.copy()

    def nodal_values(self):
        """Return scalar or vector nodal dofs in a shape useful for tests."""
        if self.is_scalar_field():
            return self.f.x.array.copy()
        return self.f.x.array.reshape((-1, self.value_dim())).copy()

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

    def from_array(self, arr):
        """Set the raw local dof array directly.

        This is the DOLFINx analogue of legacy ``Field.from_array``: it
        assigns the underlying dof array as-is, without any mesh-vertex
        reordering. Use ``set_with_ordered_numpy_array_xyz`` if the input is
        ordered by mesh vertex instead. [GitHub Copilot / Claude Sonnet 5]
        """
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != self.f.x.array.shape:
            raise ValueError(
                "from_array expects the raw dof array shape %s, got %s"
                % (self.f.x.array.shape, arr.shape)
            )
        self.f.x.array[:] = arr
        self.f.x.scatter_forward()
        return self

    def get_ordered_numpy_array_xyz(self):
        """Return nodal values ordered to match mesh-vertex order.

        Legacy dolfin's raw vector layout for vector Lagrange spaces is
        component-blocked ("xxx": all x-components, then all y, then all z)
        and unrelated to mesh vertex order, so ``Field`` built explicit
        ``v2d``/``d2v`` permutation maps to recover a vertex-ordered,
        per-node-interleaved ("xyz") view. DOLFINx's raw dof array for a
        blocked vector Lagrange-1 space is already per-node interleaved, but
        dof numbering is still not guaranteed to match mesh vertex numbering
        (e.g. under dof reordering or in parallel), so this builds the
        permutation explicitly from dof/vertex coordinates rather than
        assuming index equality. There is no DOLFINx equivalent of legacy's
        component-blocked "xxx" layout for blocked vector spaces, so that
        variant is intentionally not provided here. [GitHub Copilot / Claude
        Sonnet 5]
        """
        permutation = _vertex_order_permutation(self.functionspace)
        return self.nodal_values()[permutation]

    def set_with_ordered_numpy_array_xyz(self, ordered_array):
        """Set field values from an array ordered to match mesh vertices.

        See ``get_ordered_numpy_array_xyz`` for the ordering convention and
        its relationship to legacy ``Field.set_with_ordered_numpy_array_xyz``.
        [GitHub Copilot / Claude Sonnet 5]
        """
        permutation = _vertex_order_permutation(self.functionspace)
        ordered_array = np.asarray(ordered_array, dtype=np.float64)
        expected_shape = (
            (permutation.size,)
            if self.is_scalar_field()
            else (permutation.size, self.value_dim())
        )
        if ordered_array.shape != expected_shape:
            raise ValueError(
                "set_with_ordered_numpy_array_xyz expects shape %s, got %s"
                % (expected_shape, ordered_array.shape)
            )

        if self.is_scalar_field():
            self.f.x.array[permutation] = ordered_array
        else:
            self.f.x.array.reshape((-1, self.value_dim()))[permutation] = (
                ordered_array
            )
        self.f.x.scatter_forward()
        return self

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

    vertex_to_dof = np.empty(vertex_coords.shape[0], dtype=np.int64)
    for vertex_index, point in enumerate(np.round(vertex_coords, decimals=9)):
        key = tuple(point)
        if key not in dof_index_by_key:
            raise ValueError(
                "could not match DOLFINx dofs to mesh vertices by "
                "coordinate; coordinate/value ordering is not supported for "
                "this space"
            )
        vertex_to_dof[vertex_index] = dof_index_by_key[key]

    return vertex_to_dof
