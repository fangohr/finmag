"""DOLFINx uniaxial-anisotropy interaction."""

from aeon import timer
from dolfinx import fem
from ufl import inner

from finmag.field import Field

from .energy_base import (
    EnergyBase,
    _require_cg1_magnetisation,
    axis_coefficient,
    scalar_coefficient,
)


class UniaxialAnisotropy(EnergyBase):
    """Uniaxial anisotropy using box assembly.

    The energy-density convention is retained exactly as

    ``K1 * (1 - (axis . m)**2) - K2 * (axis . m)**4``.

    Constant scalar ``K1``/``K2`` or spatially varying ones (callable, Field or
    Function) are supported and placed -- as legacy did -- into a **CG1**
    (nodal) scalar space; the axis into a **CG1** vector space. A non-zero
    *constant* axis is normalised during construction, restoring the intended
    legacy contract that the dot product represents an angle cosine; a
    spatially varying axis is used as given (legacy did not renormalise it).
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
        self.K1_value = scalar_coefficient(K1, "K1")
        self.K2_value = scalar_coefficient(K2, "K2")
        self.axis_value = axis_coefficient(axis, "anisotropy axis")
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
