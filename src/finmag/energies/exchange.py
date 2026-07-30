"""DOLFINx exchange interaction."""

import numpy as np
from aeon import timer
from dolfinx import fem
from ufl import grad, inner

from finmag.field import Field

from .energy_base import EnergyBase, _require_cg1_magnetisation, scalar_coefficient


class Exchange(EnergyBase):
    """Compute exchange energy and its lumped-box effective field.

    Supports a constant scalar ``A`` or a spatially varying ``A`` (a callable,
    :class:`~finmag.field.Field` or ``dolfinx.fem.Function``), placed -- exactly
    as legacy did -- into a **DG0** (cellwise-constant) coefficient space; and
    the ``box-assemble`` method. Coordinates are converted with
    ``unit_length**-2`` inside the gradient energy density.
    """

    def __init__(self, A, method="box-assemble", name="Exchange"):
        self.A_value = scalar_coefficient(A, "A")
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
