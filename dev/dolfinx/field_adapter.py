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
