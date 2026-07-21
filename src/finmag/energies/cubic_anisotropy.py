"""DOLFINx cubic-anisotropy interaction."""

import numbers

import numpy as np
from aeon import timer
from dolfinx import fem
from ufl import inner

from finmag.field import Field

from .energy_base import EnergyBase, _require_cg1_magnetisation, mu0


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
    both the legacy class and here. ``assemble=True`` uses the box-assemble
    weak-form derivative provided by ``EnergyBase.compute_field``.
    ``assemble=False`` (the legacy default) uses the native/compiled direct
    computation ``finmag.native.llg.compute_cubic_field`` in the legacy code;
    this port reproduces that path in NumPy as ``_compute_field_analytic`` --
    the closed-form nodal field

    ``H = -1/(mu0*Ms) * dE/dm``

    with, for each axis ``u_k`` and its projection ``p_k in {a, b, c}``,

    ``dE/dp1 = 2*K1*a*(b**2+c**2) + 2*K2*a*b**2*c**2 + 4*K3*a**3*(b**4+c**4)``

    (and cyclically for ``p2``/``p3``), giving ``dE/dm = (dE/da)*u1 +
    (dE/db)*u2 + (dE/dc)*u3``. Both flags therefore support ``compute_field``
    (and dynamics via ``EffectiveField``); they differ only in discretisation
    (exact nodal analytic field vs box-assemble weak-form derivative), which
    is the legacy behaviour exactly, and the two agree under mesh refinement.

    Native K2 typo (DELIBERATE DEVIATION, documented): the legacy native
    routine ``native/src/llg/energy.cc:116`` writes the K2 contribution's
    ``hz`` line as ``hz[i] += K2[2]*(...)`` -- a fixed node index ``2`` where
    every other line uses the per-node ``K2[i]``. For the spatially *constant*
    ``K2`` supported by this slice the nodal ``K2`` array is uniform, so
    ``K2[2] == K2[i]`` and the typo is numerically dormant: the native oracle
    reproduces the correct analytic field to ~1e-12 (see
    ``test_native_oracle_*`` and the K2 fixture). This port implements the
    *correct* per-node field unconditionally; it would only diverge from the
    legacy native output under spatially varying ``K2``, which is deferred by
    name. See ``transition-notes.org``/``dev/dolfinx/porting_map.md`` Task 14.
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
            self.compute_field = self._compute_field_analytic
        return self

    def _compute_field_analytic(self):
        """Legacy-default (``assemble=False``) nodal analytic effective field.

        NumPy transcription of the native routine
        ``native/src/llg/energy.cc::compute_cubic_field``: the per-node
        analytic ``H = -1/(mu0 Ms) dE/dm`` obtained by the chain rule through
        ``a = u1.m``, ``b = u2.m``, ``c = u3.m`` (see the class docstring for
        the closed form and the K2 native-typo note). This is the legacy
        default's *own* discretisation -- an exact pointwise field at the CG1
        nodes -- and is deliberately distinct from the box-assemble
        weak-form-derivative field used for ``assemble=True``; the two agree
        in the continuum limit (they converge together under refinement).

        Returned in the same flat, rank-local owned-dof layout as
        ``EnergyBase.compute_field`` so it is a drop-in replacement.
        """
        m_nodes = self.m.as_array().reshape(-1, self.m.value_dim())
        Ms = self.Ms.as_constant()

        u1 = self.u1_value
        u2 = self.u2_value
        u3 = self.u3_value
        a = m_nodes @ u1
        b = m_nodes @ u2
        c = m_nodes @ u3

        K1 = self.K1_value
        K2 = self.K2_value
        K3 = self.K3_value
        # dE/da, dE/db, dE/dc, then dE/dm = (dE/da) u1 + (dE/db) u2 + (dE/dc) u3.
        g1 = (2 * K1 * a * (b**2 + c**2) + 2 * K2 * a * b**2 * c**2
              + 4 * K3 * a**3 * (b**4 + c**4))
        g2 = (2 * K1 * b * (a**2 + c**2) + 2 * K2 * b * a**2 * c**2
              + 4 * K3 * b**3 * (a**4 + c**4))
        g3 = (2 * K1 * c * (a**2 + b**2) + 2 * K2 * c * a**2 * b**2
              + 4 * K3 * c**3 * (a**4 + b**4))
        dEdm = g1[:, None] * u1 + g2[:, None] * u2 + g3[:, None] * u3
        H = -(1.0 / (mu0 * Ms)) * dEdm
        return H.reshape(-1)


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
