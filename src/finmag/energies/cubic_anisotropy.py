"""DOLFINx cubic-anisotropy interaction."""

import numpy as np
from aeon import timer
from dolfinx import fem
from ufl import TestFunction, dx, inner

from finmag.field import Field

from .energy_base import (
    EnergyBase,
    _assemble_vector_owned,
    _require_cg1_magnetisation,
    axis_coefficient,
    mu0,
    scalar_coefficient,
)


class CubicAnisotropy(EnergyBase):
    """Constant-coefficient cubic anisotropy using box assembly.

    With ``a = u1 . m``, ``b = u2 . m``, ``c = u3 . m`` and
    ``u3 = u1 x u2`` (formed exactly as in the legacy class), the energy
    density is

    ``K1*(a**2*b**2 + b**2*c**2 + c**2*a**2) + K2*(a**2*b**2*c**2)
      + K3*(a**4*b**4 + b**4*c**4 + c**4*a**4)``.

    Matching the legacy class exactly, ``u1``/``u2`` are used *as given*:
    they are not renormalised and their orthogonality is not checked (the
    legacy docstring only says they "should be unit vectors"). Constant scalar
    ``K1``/``K2``/``K3`` *and* spatially varying ones (callable/Field/Function,
    placed into a CG1 nodal space exactly as legacy did) are supported.
    Spatially varying ``u1``/``u2`` axes remain deferred by name.

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
    reproduces the correct analytic field to a measured ~1e-15 relative error
    (asserted at the fixture's declared ``rtol=1e-11``; see
    ``test_native_oracle_*`` and the K2 fixture). This port implements the
    *correct* per-node field unconditionally. Task 16 makes spatially varying
    ``K2`` supported, so this deviation is now LIVE (still USER ACCEPTANCE
    PENDING): the port's ``hz`` field diverges from the legacy native output by
    exactly ``-2/(mu0 Ms) (K2[v*] - K2[i]) termz[i]`` (``v*`` = native array
    index 2), pinned in
    ``test_variable_params_dolfinx.py::test_k2_varying_diverges_from_legacy_native_in_hz_only``
    against the spatially-varying-K2 oracle fixture. Energy is box-assembled
    (it never uses the native field path) and matches legacy. See
    ``transition-notes.org``/``dev/dolfinx/porting_map.md`` Tasks 14 & 16.

    Spatially varying ``Ms`` (fix round 1, Finding 1): the native analytic
    field now uses ``Ms`` node-for-node (``_ms_per_node``), matching legacy's
    own ``self.Ms.get_numpy_array_debug()`` usage in
    ``__compute_field_directly``, instead of requiring a single constant
    value. A spatially uniform ``Ms`` (any function space) still takes the
    cheap global-value path. A spatially varying ``Ms`` must be defined on a
    scalar space with the same per-node (CG1) layout as ``m`` -- exactly the
    requirement the legacy native routine's ``Ms_arr.check_shape(nodes, ...)``
    imposed, and which legacy's own tests/oracle generators satisfied by
    passing ``Ms`` on ``m``'s CG1 space directly. A varying ``Ms`` on a
    mismatched space (e.g. the DG0 space ``finmag.sim.sim.Simulation``/
    ``finmag.physics.llg.LLG`` always use) raises a documented
    :class:`ValueError` rather than silently misindexing or letting a
    confusing NumPy broadcast error surface, reproducing -- not silently
    dropping -- legacy's own limitation (verified: legacy raises
    ``ValueError: compute_cubic_field: Ms: Expected array of shape (nodes),
    got (cells)`` in exactly this scenario). See
    ``test_variable_params_dolfinx.py`` for the varying-Ms tests and
    ``cubic_varying_ms_oracle.json`` for the quantitative pin.
    """

    def __init__(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy',
                 assemble=False):
        self.u1_value = _constant_cubic_axis(u1, "u1")
        self.u2_value = _constant_cubic_axis(u2, "u2")
        self.u3_value = np.cross(self.u1_value, self.u2_value)

        self.K1_value = scalar_coefficient(K1, "K1")
        self.K2_value = scalar_coefficient(K2, "K2")
        self.K3_value = scalar_coefficient(K3, "K3")

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
            # Legacy native path feeds the compiled ``compute_cubic_field`` a
            # per-node K array obtained by a mass-lumped assemble of the CG1 K
            # field (``assemble(K_field.f * v * dx).get_local() / volumes``);
            # this reproduces exactly that placement so spatially varying K's
            # match the legacy native field node-for-node (except the
            # deliberately-fixed K2 hz typo). For constant K it equals the
            # scalar, preserving the constant-K native oracle match (measured
            # ~1e-15 relative error, asserted at the fixture's rtol=1e-11).
            scalar_test = TestFunction(self.S1)
            self._K1_nodal = (
                _assemble_vector_owned(self.K1.f * scalar_test * dx, self.S1)
                / self.nodal_volume_S1
            )
            self._K2_nodal = (
                _assemble_vector_owned(self.K2.f * scalar_test * dx, self.S1)
                / self.nodal_volume_S1
            )
            self._K3_nodal = (
                _assemble_vector_owned(self.K3.f * scalar_test * dx, self.S1)
                / self.nodal_volume_S1
            )
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
        Ms = self._ms_per_node(m_nodes.shape[0])

        u1 = self.u1_value
        u2 = self.u2_value
        u3 = self.u3_value
        a = m_nodes @ u1
        b = m_nodes @ u2
        c = m_nodes @ u3

        # Per-node mass-lumped K arrays (aligned with ``m_nodes`` rows: the S1
        # scalar and S3 blocked-vector CG1 spaces enumerate vertices
        # identically). For constant K these are uniform.
        K1 = self._K1_nodal
        K2 = self._K2_nodal
        K3 = self._K3_nodal
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

    def _ms_per_node(self, n_nodes):
        """Per-node ``Ms`` for the native analytic field (fix round 1, Finding 1).

        Legacy's own ``assemble=False`` path (``cubic_anisotropy.py: self.Ms =
        self.Ms.get_numpy_array_debug()``) fed the compiled
        ``compute_cubic_field`` the *raw* (not mass-lumped) nodal array of
        whatever ``Ms`` ``Field`` the caller passed to ``setup``, and the
        native routine hard-requires that array to have exactly as many
        entries as ``m`` has nodes (``native/src/llg/energy.cc``:
        ``Ms_arr.check_shape(nodes, ...)``). Legacy's own working usage (e.g.
        ``cubic_anisotropy_test.py``, and the ``gen_cubic_*_native_oracle.py``
        fixture generators) always passed ``Ms`` on the CG1 scalar space
        matching ``m`` for this reason. Verified empirically against the
        oracle: routing a constant *DG0* ``Ms`` (the space
        ``finmag.sim.sim.Simulation``/``LLG`` always use) through
        ``Simulation.effective_field()`` with ``assemble=False`` cubic
        anisotropy raises ``ValueError: compute_cubic_field: Ms: Expected
        array of shape (nodes), got (cells)`` in legacy -- i.e. legacy itself
        does not support an arbitrary-space ``Ms`` here, constant or varying.

        This port mirrors that exactly: a spatially uniform ``Ms`` (any
        space) still takes the cheap global-value fast path (matching the
        constant-``Ms`` behaviour every existing ``assemble=False`` test,
        including through ``Simulation``, already relies on); a spatially
        *varying* ``Ms`` is used node-for-node via ``Ms.as_array()`` --
        raising a clear, documented error (rather than a confusing NumPy
        broadcast failure or silently-wrong values) if its array length does
        not match the number of ``m`` nodes, reproducing legacy's own
        DG0-vs-CG1 limitation rather than silently dropping it. [Claude
        Sonnet 5]
        """
        if self.Ms.is_constant():
            return self.Ms.as_constant()
        ms_nodal = self.Ms.as_array()
        if ms_nodal.shape[0] != n_nodes:
            raise ValueError(
                "CubicAnisotropy(assemble=False) with a spatially varying "
                "Ms requires Ms's per-node array to align with m's per-node "
                "(CG1) layout, exactly as legacy's native routine required "
                "(native/src/llg/energy.cc: "
                "Ms_arr.check_shape(nodes, ...)); got {} Ms values for {} m "
                "nodes. A DG0 Ms Field (e.g. from finmag.sim.sim.Simulation "
                "or finmag.physics.llg.LLG, which always place Ms in DG0) "
                "has one value per cell, not per node, and hits exactly this "
                "mismatch in legacy too. Pass Ms on a CG1 scalar space "
                "matching m (as finmag.energies.cubic_anisotropy_test.py's "
                "own legacy usage does), or use assemble=True for the "
                "box-assembled field, which supports any positive scalar Ms "
                "placement.".format(ms_nodal.shape[0], n_nodes)
            )
        return ms_nodal[:, None]


def _constant_cubic_axis(value, name):
    """Validate a constant cubic-anisotropy axis (u1/u2).

    Spatially varying cubic axes remain deferred by name in this slice (there
    is no legacy oracle for them and the ``u3 = u1 x u2`` cross product would
    need a per-node evaluation); spatially varying cubic *K*'s are supported.
    The axis is stored exactly as given -- deliberately *not* normalised and
    *not* checked for orthogonality, matching the legacy class.
    """
    if (
        isinstance(value, (Field, fem.Function))
        or callable(value)
        or (isinstance(value, str))
        or (isinstance(value, (tuple, list))
            and any(isinstance(component, str) for component in value))
    ):
        raise NotImplementedError(
            "spatially varying cubic-anisotropy {} is deferred; pass a "
            "constant 3-vector (spatially varying cubic K1/K2/K3 are "
            "supported)".format(name)
        )
    return axis_coefficient(value, name, normalise=False)
