"""DOLFINx uniaxial-anisotropy interaction."""

import numbers

import numpy as np
from aeon import timer
from dolfinx import fem
from ufl import inner

from finmag.field import Field

from .energy_base import EnergyBase, _require_cg1_magnetisation


class UniaxialAnisotropy(EnergyBase):
    """Constant-coefficient uniaxial anisotropy using box assembly.

    The energy-density convention is retained exactly as

    ``K1 * (1 - (axis . m)**2) - K2 * (axis . m)**4``.

    A non-zero constant axis is normalised during construction, restoring the
    intended legacy contract that the dot product represents an angle cosine.
    Spatially varying coefficients and axes are deferred.
    """

    def __init__(
        self,
        K1,
        axis,
        K2=0,
        method="box-assemble",
        name="Anisotropy",
        assemble=True,
    ):
        if not assemble:
            raise NotImplementedError(
                "the legacy native/direct anisotropy path is not ported; "
                "use box assembly"
            )
        self.K1_value = _constant_scalar_value(K1, "K1")
        self.K2_value = _constant_scalar_value(K2, "K2")
        self.axis_value = _constant_axis(axis)
        self.name = name
        self.assemble = True
        super().__init__(method=method, in_jacobian=True)

    @timer.method
    def setup(self, m, Ms, unit_length=1.0):
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if m.value_dim() != 3 or m.is_scalar_field():
            raise ValueError(
                "UniaxialAnisotropy requires a three-component m Field"
            )
        _require_cg1_magnetisation(m)

        coefficient_space = fem.functionspace(m.mesh(), ("Lagrange", 1))
        vector_space = fem.functionspace(m.mesh(), ("Lagrange", 1, (3,)))
        self.K1 = Field(coefficient_space, self.K1_value, name="K1")
        self.K2 = Field(coefficient_space, self.K2_value, name="K2")
        self.axis = Field(vector_space, self.axis_value, name="axis")

        alignment = inner(self.axis.f, m.f)
        E_integrand = self.K1.f * (1.0 - alignment**2)
        E_integrand -= self.K2.f * alignment**4

        super().setup(E_integrand, m, Ms, unit_length)
        return self


def _constant_scalar_value(value, name):
    if isinstance(value, (Field, fem.Function, str)) or callable(value):
        raise NotImplementedError(
            "spatially varying {} is deferred from the first DOLFINx "
            "anisotropy slice".format(name)
        )
    if isinstance(value, fem.Constant):
        value = value.value
    if isinstance(value, numbers.Real):
        result = float(value)
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise NotImplementedError(
                "spatially varying {} is deferred from the first DOLFINx "
                "anisotropy slice".format(name)
            )
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _constant_axis(axis):
    is_string_expression = isinstance(axis, str) or (
        isinstance(axis, (tuple, list))
        and any(isinstance(component, str) for component in axis)
    )
    if (
        isinstance(axis, (Field, fem.Function))
        or callable(axis)
        or is_string_expression
    ):
        raise NotImplementedError(
            "spatially varying anisotropy axes are deferred from the first "
            "DOLFINx anisotropy slice"
        )
    if isinstance(axis, fem.Constant):
        axis = axis.value
    value = np.asarray(axis, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("anisotropy axis must be a three-component vector")
    if not np.all(np.isfinite(value)):
        raise ValueError("anisotropy axis must contain finite values")
    norm = np.linalg.norm(value)
    if norm == 0.0:
        raise ValueError("anisotropy axis must be non-zero")
    return value / norm
