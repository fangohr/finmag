"""DOLFINx static Zeeman interaction."""

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from ufl import dx, inner

from finmag.field import Field, associated_scalar_space

from .energy_base import (
    _assemble_scalar,
    _require_cg1_magnetisation,
    mu0,
)


class Zeeman:
    """Static external field in A/m.

    Constants and Python callables use the same assignment contract as
    :class:`finmag.field.Field`. Energy assembly and averages are collective;
    ``compute_field`` returns flat rank-local owned backend-order values.
    """

    def __init__(self, H, name="Zeeman", **kwargs):
        self.H_value = H
        self.name = name
        self.kwargs = kwargs
        self.in_jacobian = False

    def setup(self, m, Ms, unit_length=1.0):
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if not isinstance(Ms, Field):
            raise TypeError("Ms must be a finmag.Field")
        if m.is_scalar_field() or m.value_dim() != 3:
            raise ValueError("Zeeman requires a three-component m Field")
        _require_cg1_magnetisation(m)
        if not Ms.is_scalar_field():
            raise ValueError("Ms must be a scalar Field")
        if m.mesh() is not Ms.mesh():
            raise ValueError("m and Ms must use the same mesh")

        unit_length = float(unit_length)
        if not np.isfinite(unit_length) or unit_length <= 0.0:
            raise ValueError("unit_length must be a positive finite number")
        ms_values = Ms.as_array()
        local_invalid_ms = bool(
            np.any(~np.isfinite(ms_values)) or np.any(ms_values <= 0.0)
        )
        if m.mesh().comm.allreduce(local_invalid_ms, op=MPI.LOR):
            raise ValueError("Ms must be positive")
        if not Ms.is_constant():
            raise NotImplementedError(
                "spatially varying Ms is deferred from the first DOLFINx "
                "energy slice"
            )

        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        self.S1 = associated_scalar_space(m.functionspace)
        if hasattr(self, "H"):
            del self.H
        if hasattr(self, "_energy_density_field"):
            del self._energy_density_field
        self.set_value(self.H_value, **self.kwargs)
        return self

    def set_value(self, value, **kwargs):
        """Set a constant or callable external field after ``setup``."""
        if not hasattr(self, "m"):
            raise RuntimeError("Zeeman.setup must be called before set_value")
        if kwargs:
            raise NotImplementedError(
                "legacy Expression parameters are unavailable; pass a callable"
            )
        if hasattr(self, "H"):
            self.H.set(value)
        else:
            self.H = Field(self.m.functionspace, value, name="H_ext")
        self.H_value = value
        self.value = value
        self.E = -mu0 * self.Ms.f * inner(self.m.f, self.H.f)
        return self

    def compute_field(self):
        """Return flat rank-local owned field coefficients."""
        return self.H.as_array()

    def average_field(self):
        """Collectively return the legacy arithmetic nodal field average."""
        values = self.compute_field().reshape((-1, self.m.value_dim()))
        local_sum = np.sum(values, axis=0)
        global_sum = np.zeros_like(local_sum)
        self.m.mesh().comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        global_count = self.m.mesh().comm.allreduce(values.shape[0], op=MPI.SUM)
        return global_sum / global_count

    def compute_energy(self, dx=dx):
        """Collectively integrate Zeeman energy over the supplied measure."""
        mesh_energy = _assemble_scalar(self.m.mesh(), self.E * dx)
        return mesh_energy * self.unit_length ** self.m.mesh_dim()

    def energy_density(self):
        """Collectively return legacy pointwise nodal density as a Field."""
        values = -mu0 * self.Ms.as_constant() * np.sum(
            self.m.as_array().reshape((-1, 3))
            * self.H.as_array().reshape((-1, 3)),
            axis=1,
        )
        expected_size = self.S1.dofmap.index_map.size_local
        if values.shape != (expected_size,):
            raise ValueError(
                "Zeeman pointwise density requires matching scalar/vector "
                "CG1 ownership"
            )
        if not hasattr(self, "_energy_density_field"):
            self._energy_density_field = Field(
                self.S1, name="{}_energy_density".format(self.name)
            )
        self._energy_density_field.from_array(values)
        return self._energy_density_field

    def energy_density_function(self):
        return self.energy_density().f


class _DeferredZeeman:
    feature_name = "time-dependent Zeeman interaction"

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "{} is deferred from the static Task 5 DOLFINx slice".format(
                self.feature_name
            )
        )


class DipolarField(_DeferredZeeman):
    feature_name = "dipolar point-field construction"


class TimeZeeman(_DeferredZeeman):
    pass


class DiscreteTimeZeeman(_DeferredZeeman):
    pass


class TimeZeemanPython(_DeferredZeeman):
    pass


class OscillatingZeeman(_DeferredZeeman):
    pass
