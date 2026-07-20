"""DOLFINx exchange interaction."""

import numbers

import numpy as np
from aeon import timer
from dolfinx import fem
from ufl import grad, inner

from finmag.field import Field

from .energy_base import EnergyBase, _require_cg1_magnetisation


class Exchange(EnergyBase):
    """Compute exchange energy and its lumped-box effective field.

    This first DOLFINx slice supports a spatially constant scalar ``A`` and the
    ``box-assemble`` method. Coordinates are converted with
    ``unit_length**-2`` inside the gradient energy density.
    """

    def __init__(self, A, method="box-assemble", name="Exchange"):
        self.A_value = _constant_scalar_value(A, "A")
        self.name = name
        super().__init__(method=method, in_jacobian=True)

    @timer.method
    def setup(self, m, Ms, unit_length=1.0):
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if m.value_dim() != 3 or m.is_scalar_field():
            raise ValueError("Exchange requires a three-component m Field")
        _require_cg1_magnetisation(m)

        unit_length = float(unit_length)
        if not np.isfinite(unit_length) or unit_length <= 0.0:
            raise ValueError("unit_length must be a positive finite number")

        coefficient_space = fem.functionspace(m.mesh(), ("DG", 0))
        self.A = Field(coefficient_space, self.A_value, name="A")
        self.exchange_factor = fem.Constant(m.mesh(), 1.0 / unit_length**2)
        E_integrand = (
            self.exchange_factor
            * self.A.f
            * inner(grad(m.f), grad(m.f))
        )
        super().setup(E_integrand, m, Ms, unit_length)
        return self


def _constant_scalar_value(value, name):
    if isinstance(value, (Field, fem.Function, str)) or callable(value):
        raise NotImplementedError(
            "spatially varying {} is deferred from the first DOLFINx "
            "energy slice".format(name)
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
                "energy slice".format(name)
            )
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result
