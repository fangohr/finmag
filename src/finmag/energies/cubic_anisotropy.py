"""DOLFINx cubic-anisotropy interaction."""

import numbers

import numpy as np
from aeon import timer
from dolfinx import fem
from ufl import inner

from finmag.field import Field

from .energy_base import EnergyBase, _require_cg1_magnetisation


class CubicAnisotropy(EnergyBase):
    """Constant-coefficient cubic anisotropy using box assembly.

    With ``a = u1 . m``, ``b = u2 . m``, ``c = u3 . m`` and
    ``u3 = u1 x u2`` (formed exactly as in the legacy class), the energy
    density is

    ``K1*(a**2*b**2 + b**2*c**2 + c**2*a**2) + K2*(a**2*b**2*c**2)
      + K3*(a**4*b**4 + b**4*c**4 + c**4*a**4)``.

    Matching the legacy class exactly, ``u1``/``u2`` are used *as given*:
    they are not renormalised and their orthogonality is not checked (the
    legacy docstring only says they "should be unit vectors"). Only constant
    scalar ``K1``/``K2``/``K3`` and constant ``u1``/``u2`` axes are supported
    in this slice; spatially varying coefficients/axes raise
    ``NotImplementedError`` by name (deferred to a later slice).

    The legacy ``assemble`` flag chose between two different *field*
    computation algorithms; the total *energy* is always box-assembled, in
    both the legacy class and here. ``assemble=True`` used the very same
    box-assemble weak-form derivative this port always uses for
    ``compute_field()``. ``assemble=False`` (the legacy default) used a
    separate native/compiled direct computation
    (``finmag.native.llg.compute_cubic_field``) that is not part of the
    DOLFINx port. Construction and ``compute_energy()`` therefore work
    identically regardless of ``assemble`` (matching the legacy default
    constructor call exactly); ``compute_field()`` -- and anything built on
    it, e.g. ``EffectiveField``/dynamics -- requires ``assemble=True`` and
    raises ``NotImplementedError`` by name otherwise.
    """

    def __init__(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy',
                 assemble=False):
        self.u1_value = _constant_axis(u1, "u1")
        self.u2_value = _constant_axis(u2, "u2")
        self.u3_value = np.cross(self.u1_value, self.u2_value)

        self.K1_value = _constant_scalar_value(K1, "K1")
        self.K2_value = _constant_scalar_value(K2, "K2")
        self.K3_value = _constant_scalar_value(K3, "K3")

        self.name = name
        self.assemble = bool(assemble)
        super(CubicAnisotropy, self).__init__("box-assemble", in_jacobian=True)

    @timer.method
    def setup(self, m, Ms, unit_length=1.0):
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if m.value_dim() != 3 or m.is_scalar_field():
            raise ValueError(
                "CubicAnisotropy requires a three-component m Field")
        _require_cg1_magnetisation(m)

        coefficient_space = fem.functionspace(m.mesh(), ("Lagrange", 1))
        vector_space = fem.functionspace(m.mesh(), ("Lagrange", 1, (3,)))

        self.K1 = Field(coefficient_space, self.K1_value, name="K1")
        self.K2 = Field(coefficient_space, self.K2_value, name="K2")
        self.K3 = Field(coefficient_space, self.K3_value, name="K3")

        self.u1 = Field(vector_space, self.u1_value, name="u1")
        self.u2 = Field(vector_space, self.u2_value, name="u2")
        self.u3 = Field(vector_space, self.u3_value, name="u3")

        a = inner(self.u1.f, m.f)
        b = inner(self.u2.f, m.f)
        c = inner(self.u3.f, m.f)

        E_integrand = self.K1.f * (a**2 * b**2 + b**2 * c**2 + c**2 * a**2)
        E_integrand += self.K2.f * (a**2 * b**2 * c**2)
        E_integrand += self.K3.f * (
            a**4 * b**4 + b**4 * c**4 + c**4 * a**4)

        super(CubicAnisotropy, self).setup(E_integrand, m, Ms, unit_length)

        if not self.assemble:
            self.compute_field = self._compute_field_not_ported
        return self

    def _compute_field_not_ported(self):
        raise NotImplementedError(
            "the legacy native/direct cubic-anisotropy field computation "
            "(assemble=False, the default) is not ported to DOLFINx; "
            "construct CubicAnisotropy(..., assemble=True) to use box "
            "assembly for compute_field()"
        )


def _constant_scalar_value(value, name):
    if isinstance(value, (Field, fem.Function, str)) or callable(value):
        raise NotImplementedError(
            "spatially varying {} is deferred from the first DOLFINx "
            "cubic-anisotropy slice".format(name)
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
                "cubic-anisotropy slice".format(name)
            )
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _constant_axis(value, name):
    is_string_expression = isinstance(value, str) or (
        isinstance(value, (tuple, list))
        and any(isinstance(component, str) for component in value)
    )
    if (
        isinstance(value, (Field, fem.Function))
        or callable(value)
        or is_string_expression
    ):
        raise NotImplementedError(
            "spatially varying {} is deferred from the first DOLFINx "
            "cubic-anisotropy slice".format(name)
        )
    if isinstance(value, fem.Constant):
        value = value.value
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("{} must be a three-component vector".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain finite values".format(name))
    # Deliberately *not* normalised and *not* checked for orthogonality: the
    # legacy class stores u1/u2 exactly as given (see
    # dev/dolfinx/porting_map.md for the axis-handling investigation).
    return array
