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

    When ``K2`` is the constant ``0`` -- its default -- the quartic term is left
    out of the energy integrand entirely. This is an internal optimisation with
    no effect on computed values: it avoids the much higher quadrature degree
    FFCx picks for a degree-8 integrand (register **P1**). A spatially varying
    ``K2`` keeps the term, even where it evaluates to zero.
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
        # Only carry the quartic term when K2 actually contributes. With CG1
        # axis and m, ``alignment**4`` is degree 8, so FFCx raises the
        # quadrature degree for the WHOLE integrand -- measured 60.1 ms against
        # 8.5 ms for the K1 term alone at 20787 dofs, a 7.0x cost that is
        # independent of K2's value and therefore pure waste at the default
        # K2 == 0 (register P1). Dropping it also makes dE_dm linear in m,
        # which a precomputed-operator fast path requires.
        #
        # Deliberately only the CONSTANT K2 == 0 case, which is the default and
        # the one that matters: `scalar_coefficient` returns a float for every
        # constant form of K2 and the original object for a callable/Field/
        # Function, so this is exactly "K2 was given as a constant zero". A
        # spatially varying K2 keeps the term even where it evaluates to zero.
        # Testing the constant rather than the assembled values also keeps the
        # decision rank-invariant by construction -- no collective is needed,
        # where a value-based test would have to agree across ranks to avoid
        # building different forms. K2 is only ever supplied at construction and
        # the integrand is rebuilt by every setup(), so there is no later
        # mutation to invalidate. [Claude Opus 5]
        k2_is_zero = isinstance(self.K2_value, float) and self.K2_value == 0.0
        if not k2_is_zero:
            E_integrand -= self.K2.f * alignment**4

        super().setup(E_integrand, m, Ms, unit_length)
        return self
