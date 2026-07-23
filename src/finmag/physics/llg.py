"""Deterministic Landau-Lifshitz-Gilbert core, ported directly to DOLFINx.

This is the direct DOLFINx port of the legacy ``finmag.physics.llg.LLG``. It
solves the Landau-Lifshitz form of the LLG equation

.. math::

    \\frac{d\\vec{m}}{dt} = -\\gamma_{LL}\\, \\vec{m}\\times\\vec{H}
        - \\alpha\\gamma_{LL}\\, \\vec{m}\\times(\\vec{m}\\times\\vec{H})
        + c\\,(1 - |\\vec{m}|^2)\\,\\vec{m}

where :math:`\\gamma_{LL} = \\gamma / (1 + \\alpha^2)`. The final term is the
legacy numerical norm-relaxation correction (``relaxation_i`` in
``native/src/llg/llg.cc``, with coefficient ``0.1/char_time == self.c``); it
holds :math:`|\\vec{m}|` at unit length during integration and vanishes when
:math:`|\\vec{m}| = 1`. The precession and damping terms are transcribed
node-for-node from ``calc_llg_dmdt`` (``native/src/llg/llg.cc``); no compiled
extension is used.

Scope of this slice: scalar ``Ms``, scalar or spatially varying ``alpha``
(per-node in both the damping term and ``gamma_LL``), ``gamma``, ``set_m``,
``solve`` and ``solve_for``, driven entirely from the ported ``EffectiveField``
registry. The state vector for ``solve``/``solve_for``/``m`` setters is the
component-blocked, coordinate-ordered ``xxx`` array (rank-local owned dofs);
``H_eff`` is routed into the identical ordering before the node-local update.
Thermal dynamics and multi-rank ODE state are out of scope and raise
``NotImplementedError`` by name when requested.

Task 16 adds spatially varying Gilbert damping ``alpha`` (per-node in both the
damping term and ``gamma_LL``). Task 22 adds the Slonczewski/Xiao and Zhang-Li
spin-transfer torques (``use_slonczewski`` / ``use_zhangli``) as NumPy
transcriptions of the native ``calc_llg_slonczewski_dmdt`` /
``calc_llg_zhang_li_dmdt`` kernels; the compiled STT kernels are not rebuilt for
the DOLFINx lane. [Claude Opus 4.8]
"""

import logging
import math

import numpy as np
import ufl
from dolfinx import fem, la

import finmag.util.consts as consts
from finmag.field import Field, _assemble_scalar
from finmag.physics.effective_field import EffectiveField

# default settings for logger 'finmag' set in __init__.py
logger = logging.getLogger(name="finmag")

# Physical constants transcribed verbatim from ``native/src/llg/llg.cc``
# (lines 21-25) so the STT NumPy transcription reproduces the compiled kernels
# bit-for-bit rather than drifting to scipy's CODATA values. [Claude Opus 4.8]
_E_CHARGE = 1.602176565e-19       # elementary charge, As      (llg.cc:21)
_H_BAR = 1.054571726e-34          # reduced Planck constant, Js (llg.cc:22)
_MU_B = 9.27400968e-24            # Bohr magneton               (llg.cc:24)
_MU_0 = math.pi * 4e-7            # vacuum permeability, Vs/(Am)(llg.cc:25)


def _assemble_owned_vector(form_expr, function_space):
    """Assemble a linear form and return its owned rank-local dof array.

    Mirrors ``finmag.energies.energy_base._assemble_vector_owned`` so the STT
    lumped-box gradient/projection assemblies use the identical owner+ghost
    accumulation the ported energies use.
    """
    vector = fem.assemble_vector(fem.form(form_expr))
    vector.scatter_reverse(la.InsertMode.add)
    vector.scatter_forward()
    dofmap = function_space.dofmap
    owned = dofmap.index_map.size_local * dofmap.index_map_bs
    return vector.array[:owned].copy()


class LLG(object):
    """Solves the Landau-Lifshitz-Gilbert equation (deterministic core)."""

    def __init__(self, S1, S3, do_precession=True, average=False, unit_length=1):
        """*S1* and *S3* are DOLFINx scalar and 3-component vector CG1
        function spaces (``dolfinx.fem.FunctionSpace``). ``do_precession``
        controls whether the precession term is computed. ``average`` is
        accepted for legacy signature compatibility but is unused, exactly as
        in the legacy implementation.
        """
        logger.debug("Creating LLG object.")
        del average  # accepted for signature compatibility only (legacy: unused)
        self.S1 = S1
        self.S3 = S3
        self.mesh = S3.mesh
        self.comm = self.mesh.comm
        self.DG = fem.functionspace(self.mesh, ("DG", 0))

        self.set_default_values()
        self.do_precession = do_precession
        self.unit_length = unit_length
        self.effective_field = EffectiveField(
            self._m_field, self.Ms, self.unit_length
        )
        # will be computed on demand, and carries volume of the mesh
        self.Volume = None

    def set_default_values(self):
        # Gilbert damping lives in a scalar CG1 Field (matching legacy, which
        # stored alpha as a ``df.Function`` on S1), so spatially varying alpha
        # is a per-node array; ``set_alpha`` also caches the node-ordered array
        # used by the RHS. Default is spatially uniform 0.5.
        self._alpha_field = Field(self.S1, name="alpha")
        self.set_alpha(0.5)

        self.gamma = consts.gamma
        self.c = 1e11  # 1/s numerical scaling correction
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self._Ms_dg = Field(self.DG, name="Saturation magnetisation")
        self.Ms = 8.6e5  # A/m saturation magnetisation
        self._m_field = Field(self.S3, name="m")
        self._dmdt = Field(self.S3, name="dmdt")
        self._pins = np.array([], dtype="int")

        # Spin-transfer-torque state (Task 22). Both flags default off, exactly
        # as the legacy ``LLG.__init__`` did; ``use_slonczewski``/``use_zhangli``
        # set the corresponding flag and populate the parameters below.
        self.do_slonczewski = False
        self.do_zhangli = False
        self.fun_slonczewski_time_update = None
        self.fun_zhangli_time_update = None

    # -- pinning ------------------------------------------------------------

    def set_pins(self, nodes):
        """Hold the magnetisation constant at the given owned node indices.

        Indices refer to coordinate-ordered owned nodes (the ``xxx`` state
        ordering), between 0 (inclusive) and the number of owned nodes
        (exclusive). Pinning is retained for serial use only; the multi-rank
        guard on ``solve`` rejects distributed state.
        """
        nodes = np.asarray(list(nodes), dtype="int")
        if nodes.size > 0:
            nb_nodes_mesh = self._m_field.get_ordered_numpy_array_xxx().size // 3
            if nodes.min() < 0 or nodes.max() >= nb_nodes_mesh:
                logger.error(
                    "Indices of pinned nodes should be in [0, {}), were "
                    "[{}, {}].".format(nb_nodes_mesh, nodes.min(), nodes.max())
                )
                raise ValueError(
                    "pinned node indices out of range [0, {})".format(nb_nodes_mesh)
                )
            self._pins = nodes
        else:
            self._pins = np.array([], dtype="int")

    def pins(self):
        return self._pins

    pins = property(pins, set_pins)

    # -- damping ------------------------------------------------------------

    @property
    def alpha(self):
        """Gilbert damping :math:`\\alpha`.

        Returns a plain ``float`` when spatially uniform (the common case,
        preserving the scalar contract), otherwise the per-node coordinate-
        ordered array of nodal values.
        """
        if self._alpha_field.is_constant():
            return float(self._alpha_field.as_constant())
        return self._alpha_field.get_ordered_numpy_array()

    def set_alpha(self, value):
        """Set the Gilbert damping :math:`\\alpha`.

        Accepts (Task 16) a scalar, a per-node NumPy array/list, a Python
        callable ``x -> alpha``, a :class:`~finmag.field.Field`, or a
        ``dolfinx.fem.Function`` -- everything the legacy
        ``helpers.scalar_valued_function(value, S1)`` accepted, placed into the
        scalar CG1 space. Alpha then enters the node-local RHS per node in both
        the damping term and ``gamma_LL = gamma / (1 + alpha**2)`` (precession
        and damping), exactly as the native ``calc_llg_dmdt`` did.
        """
        if np.isscalar(value):
            self._alpha_field.set(float(value))
        else:
            self._alpha_field.set(value)
        # Cache the node-ordered (xyz) array aligned with the m/H columns used
        # by ``_dmdt_numpy`` (a ``(3, N)`` component-blocked layout).
        self._alpha_node = self._alpha_field.get_ordered_numpy_array()

    # -- saturation magnetisation ------------------------------------------

    @property
    def Ms(self):
        return self._Ms_dg

    @Ms.setter
    def Ms(self, value):
        self._Ms_dg.set(value)
        self._Ms_dg.name = "Saturation magnetisation"
        self._Ms = self._Ms_dg.as_array().copy()
        self.Ms_av = float(np.average(self._Ms)) if self._Ms.size else float("nan")

    # -- magnetisation accessors -------------------------------------------

    @property
    def m_field(self):
        """The unit magnetisation Field."""
        return self._m_field

    @property
    def m_numpy(self):
        """The magnetisation as a component-blocked ``xxx`` NumPy array."""
        return self._m_field.get_ordered_numpy_array_xxx()

    @property
    def dmdt(self):
        """Owned rank-local dm/dt dof values (backend order)."""
        return self._dmdt.as_array()

    @property
    def sundials_m(self):
        """The unit magnetisation as the ``xxx`` state vector."""
        return self._m_field.get_ordered_numpy_array_xxx()

    @sundials_m.setter
    def sundials_m(self, value):
        self._require_serial("state-vector assignment")
        self._m_field.set_with_ordered_numpy_array_xxx(value)

    def m_average_fun(self, dx=None):
        """Volume-averaged magnetisation, :math:`\\frac{1}{V}\\int m\\,dV`."""
        if dx is None:
            return self._m_field.average()
        return self._m_field.average(dx=dx)

    m_average = property(m_average_fun)

    @property
    def M(self):
        """The magnetisation in A/m as a component-blocked ``xxx`` array.

        Returns ``Ms(x) * m(x)`` per node: the unit magnetisation scaled by the
        (possibly spatially varying) saturation magnetisation, in the same
        coordinate-ordered component-blocked (``xxx``) layout as
        :attr:`m_numpy` -- so ``M.reshape((3, -1))`` column ``j`` is the A/m
        magnetisation at node ``j``. The per-node ``Ms`` is the lumped-mass
        projection of the DG0 ``Ms`` onto the CG1 nodes (:meth:`_ms_nodal`),
        already node-aligned with the m columns and handling constant and
        spatially varying ``Ms`` identically.

        NOTE (correct physics, register D20): the frozen legacy ``LLG.M`` read
        ``self.m``, which raised ``RuntimeError`` -- it was broken. This port
        implements the intended ``M = Ms * m`` contract.
        """
        m_nodes = self._m_field.get_ordered_numpy_array_xxx().reshape((3, -1))
        Ms_node = self._ms_nodal()
        return (Ms_node * m_nodes).reshape(-1)

    @property
    def M_average(self):
        """The volume-average magnetisation in A/m (Ms-weighted).

        Returns ``(integral Ms*m dV) / (integral dV)`` component-wise -- the
        spatial average of :attr:`M`. For a constant ``Ms`` this equals
        ``Ms * m_average``; for a spatially varying ``Ms`` it is the correctly
        Ms-weighted volume average.

        NOTE (correct physics, register D20): the frozen legacy ``M_average``
        computed ``m_average * volume_Ms / volume`` with ``volume_Ms`` and
        ``volume`` the *identical* integral, collapsing to the dimensionless
        ``m_average`` (a unit bug). This port returns the intended A/m value.
        """
        domain = self.mesh
        volume = _assemble_scalar(domain, fem.Constant(domain, 1.0) * ufl.dx)
        if volume == 0.0:
            raise ValueError("cannot average over a zero-volume mesh")
        Ms = self._Ms_dg.f
        m = self._m_field.f
        return np.array(
            [
                _assemble_scalar(domain, Ms * m[i] * ufl.dx) / volume
                for i in range(3)
            ]
        )

    def set_m(self, value, normalise=True, **kwargs):
        """Set the magnetisation, normalising to unit length by default.

        ``value`` may be a constant tuple/list, a callable ``x -> values``, a
        :class:`~finmag.field.Field`, a ``dolfinx.fem.Function``, or a flat
        NumPy array. A NumPy array is interpreted as the component-blocked
        coordinate-ordered ``xxx`` state vector (matching ``solve_for``), not
        as raw backend dofs. Legacy string ``Expression`` values are not
        supported; pass a callable instead.
        """
        if kwargs:
            raise NotImplementedError(
                "legacy Expression keyword parameters are not supported; "
                "pass a callable"
            )
        if isinstance(value, np.ndarray):
            m0 = np.asarray(value, dtype=np.float64).reshape(-1)
            if np.any(np.isnan(m0)):
                raise ValueError("Attempting to initialise m with NaN(s)")
            self._m_field.set_with_ordered_numpy_array_xxx(m0)
            if normalise:
                self._m_field.normalise()
        else:
            self._m_field.set(value, normalised=normalise)
        return self

    # -- right-hand side ----------------------------------------------------

    def solve_for(self, m, t):
        """Set the ``xxx`` state ``m`` and return dm/dt in the same ordering."""
        self._require_serial("solve_for")
        self._m_field.set_with_ordered_numpy_array_xxx(m)
        return self.solve(t)

    def solve(self, t):
        """Return dm/dt (component-blocked ``xxx`` order) at time ``t``.

        Every field contribution is taken from the complete ``EffectiveField``
        registry; there is no bypass path.
        """
        self._require_serial("solve")

        # Accumulate the total effective field from the registry. Since Task 31
        # every interaction's compute_field() -- and therefore the accumulated
        # H_eff -- is already in the coordinate-ordered component-blocked
        # (``xxx``) layout m uses, so it is consumed directly here with NO
        # re-conversion (the former raw->xxx round-trip would double-convert).
        self.effective_field.update(t)

        m = self._m_field.get_ordered_numpy_array_xxx().reshape((3, -1))
        H = self.effective_field.H_eff.reshape((3, -1))

        # Spin-transfer-torque dispatch, mirroring the legacy ``solve``:
        # Slonczewski and Zhang-Li are mutually exclusive extra torques added
        # on top of the deterministic precession/damping/relaxation update.
        if self.do_slonczewski:
            if self.fun_slonczewski_time_update is not None:
                # Legacy contract: the callback returns a spatially uniform
                # current density (a number), broadcast over every node.
                self.J[:] = self.fun_slonczewski_time_update(t)
            self._Ms_node = self._ms_nodal()
            dmdt = self._dmdt_slonczewski_numpy(m, H)
        elif self.do_zhangli:
            if self.fun_zhangli_time_update is not None:
                # Legacy contract: the callback returns a new J profile; rebuild
                # the current-density field (the discrete gradient below then
                # picks it up, as the legacy gradient-matrix rebuild did).
                self._J.set(self.fun_zhangli_time_update(t))
            self._Ms_node = self._ms_nodal()
            H_gradm = self._compute_zhangli_gradient()
            dmdt = self._dmdt_zhangli_numpy(m, H, H_gradm)
        else:
            dmdt = self._dmdt_numpy(m, H)

        if self._pins.size:
            dmdt[:, self._pins] = 0.0

        dmdt = dmdt.reshape(-1)
        self._dmdt.set_with_ordered_numpy_array_xxx(dmdt)
        return dmdt

    def _dmdt_numpy(self, m, H):
        """Node-local LLG right-hand side; transcribed from ``calc_llg_dmdt``.

        ``m`` and ``H`` are ``(3, N)`` component-blocked owned nodal arrays.
        Returns a fresh ``(3, N)`` dm/dt array. ``alpha`` is per node (a length-
        ``N`` array aligned with the node columns), so ``gamma_LL`` and the
        damping coefficient are per node, exactly as native ``calc_llg_dmdt``.
        """
        alpha = self._alpha_node
        gamma_LL = self.gamma / (1.0 + alpha * alpha)

        m0, m1, m2 = m[0], m[1], m[2]
        h0, h1, h2 = H[0], H[1], H[2]

        mh = m0 * h0 + m1 * h1 + m2 * h2
        mm = m0 * m0 + m1 * m1 + m2 * m2

        # damping: -alpha * gamma_LL * (m x (m x H)) = damping_coeff*(m*mh - H*mm)
        damping_coeff = -alpha * gamma_LL
        dm0 = damping_coeff * (m0 * mh - h0 * mm)
        dm1 = damping_coeff * (m1 * mh - h1 * mm)
        dm2 = damping_coeff * (m2 * mh - h2 * mm)

        # numerical norm relaxation: c * (1 - |m|^2) * m  (coeff 0.1/char_time)
        relax_coeff = self.c * (1.0 - mm)
        dm0 += relax_coeff * m0
        dm1 += relax_coeff * m1
        dm2 += relax_coeff * m2

        # precession: -gamma_LL * (m x H)
        if self.do_precession:
            dm0 += -gamma_LL * (m1 * h2 - m2 * h1)
            dm1 += -gamma_LL * (m2 * h0 - m0 * h2)
            dm2 += -gamma_LL * (m0 * h1 - m1 * h0)

        return np.vstack((dm0, dm1, dm2))

    # -- generic ODE adapter (backend-neutral) ------------------------------

    def sundials_rhs(self, t, y, ydot):
        """Deterministic dm/dt adapter, ``ydot[:] = solve_for(y, t)``.

        This is backend-neutral (it does not touch native CVODE); both the SciPy
        driver slice and the native CVODE ``SundialsIntegrator`` reuse it. ``y``
        and ``ydot`` are the component-blocked coordinate-ordered ``xxx`` state.
        """
        ydot[:] = self.solve_for(y, t)
        return 0

    # -- native CVODE preconditioner / Jacobian-times-vector hooks ----------
    #
    # These reproduce the legacy ``bdf_gmres_prec_id`` default path exactly
    # (see ``finmag.drivers.sundials_integrator``): CVODE with SPGMR + a
    # left identity preconditioner and an analytic Jacobian-times-vector
    # product. ``psetup`` records the linearisation state, ``psolve`` applies
    # the identity preconditioner (``z = r``), and ``jtimes`` is the exact
    # directional derivative of the LLG right-hand side. The three per-node
    # derivative kernels are transcribed from native ``calc_llg_jtimes``
    # (``native/src/llg/llg.cc``: ``dm_precession_i`` / ``dm_damping_i`` /
    # ``dm_relaxation_i``), the same way ``_dmdt_numpy`` transcribes
    # ``calc_llg_dmdt``. [Claude Opus 4.8]

    def sundials_psetup(self, t, m, fy, jok, gamma, tmp1, tmp2, tmp3):
        # Some arguments are unused but must be present so the callback matches
        # the signature CVODE's preconditioner-setup expects. When ``jok`` is
        # false CVODE requests a fresh linearisation, so record the state and
        # signal that the "Jacobian" data was rebuilt (identity here). Returns
        # ``(retval, jcurPtr)`` mirroring the legacy wrapper. [Claude Opus 4.8]
        if not jok:
            self._m_field.set_with_ordered_numpy_array_xxx(m)
            self._reuse_jacobian = True
        return 0, (not jok)

    def sundials_psolve(self, t, y, fy, r, z, gamma, delta, lr, tmp):
        # Identity (left) preconditioner: solve ``P z = r`` with ``P = I``,
        # exactly as the legacy default ``bdf_gmres_prec_id`` path did.
        z[:] = r
        return 0

    def sundials_jtimes(self, mp, J_mp, t, m, fy, tmp):
        """Analytic Jacobian-times-vector product ``J(m,t) mp`` in ``xxx`` order.

        ``J mp = d/da rhs(m + a mp, H(m + a mp))|_{a=0}``. For the linear
        effective-field contributions the directional field derivative is
        ``H' = H(mp)``, obtained from ``EffectiveField.compute_jacobian_only``.
        The node-local Jacobian kernels are the exact derivatives of the
        precession, damping, and relaxation terms in ``_dmdt_numpy``.
        """
        self._require_serial("sundials_jtimes")
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        mp = np.asarray(mp, dtype=np.float64).reshape(-1)

        # H' = dH_eff/da in the direction mp (linear interactions -> H(mp)).
        # compute_jacobian_only / H_eff are already component-blocked (``xxx``)
        # since Task 31, so they are consumed directly with NO re-conversion.
        self._m_field.set_with_ordered_numpy_array_xxx(mp)
        Hp = self.effective_field.compute_jacobian_only(t)

        # Restore the linearisation state m and its effective field H(m, t).
        self._m_field.set_with_ordered_numpy_array_xxx(m)
        self.effective_field.update(t)
        H = self.effective_field.H_eff

        jt = self._jtimes_numpy(
            m.reshape((3, -1)), H.reshape((3, -1)),
            mp.reshape((3, -1)), Hp.reshape((3, -1)))

        if self._pins.size:
            jt[:, self._pins] = 0.0

        J_mp[:] = jt.reshape(-1)
        return 0

    def _jtimes_numpy(self, m, H, mp, Hp):
        """Node-local LLG Jacobian-times-vector kernel; transcribed from the
        native ``dm_precession_i`` / ``dm_damping_i`` / ``dm_relaxation_i``.

        All inputs are ``(3, N)`` component-blocked owned nodal arrays. Returns
        a fresh ``(3, N)`` array ``J mp``. ``alpha`` is per node exactly as in
        ``_dmdt_numpy``.
        """
        alpha = self._alpha_node
        gamma_LL = self.gamma / (1.0 + alpha * alpha)

        m0, m1, m2 = m[0], m[1], m[2]
        mp0, mp1, mp2 = mp[0], mp[1], mp[2]
        h0, h1, h2 = H[0], H[1], H[2]
        hp0, hp1, hp2 = Hp[0], Hp[1], Hp[2]

        jt0 = np.zeros_like(m0)
        jt1 = np.zeros_like(m0)
        jt2 = np.zeros_like(m0)

        # precession derivative: d(m x H) = mp x H + m x Hp
        if self.do_precession:
            jt0 += -gamma_LL * ((mp1 * h2 - mp2 * h1) + (m1 * hp2 - m2 * hp1))
            jt1 += -gamma_LL * ((mp2 * h0 - mp0 * h2) + (m2 * hp0 - m0 * hp2))
            jt2 += -gamma_LL * ((mp0 * h1 - mp1 * h0) + (m0 * hp1 - m1 * hp0))

        # damping derivative: d[(m*H)m - (m*m)H]
        mph_mhp = (mp0 * h0 + mp1 * h1 + mp2 * h2
                   + m0 * hp0 + m1 * hp1 + m2 * hp2)
        mh = m0 * h0 + m1 * h1 + m2 * h2
        mm = m0 * m0 + m1 * m1 + m2 * m2
        mmp = m0 * mp0 + m1 * mp1 + m2 * mp2
        damping_coeff = -alpha * gamma_LL
        jt0 += damping_coeff * (mph_mhp * m0 + mh * mp0 - 2 * mmp * h0 - mm * hp0)
        jt1 += damping_coeff * (mph_mhp * m1 + mh * mp1 - 2 * mmp * h1 - mm * hp1)
        jt2 += damping_coeff * (mph_mhp * m2 + mh * mp2 - 2 * mmp * h2 - mm * hp2)

        # relaxation derivative: d[(1 - m*m) m]; native relax_coeff = 0.1/char_time = c
        relax_coeff = self.c
        jt0 += relax_coeff * (-2 * mmp * m0 + (1.0 - mm) * mp0)
        jt1 += relax_coeff * (-2 * mmp * m1 + (1.0 - mm) * mp1)
        jt2 += relax_coeff * (-2 * mmp * m2 + (1.0 - mm) * mp2)

        return np.vstack((jt0, jt1, jt2))

    # -- spin-transfer torque (Task 22) -------------------------------------
    #
    # NumPy transcription of the native STT kernels (``native/src/llg/llg.cc``);
    # the compiled ``calc_llg_slonczewski_dmdt`` / ``calc_llg_zhang_li_dmdt``
    # kernels are NOT rebuilt for the DOLFINx lane -- the physics is transcribed
    # term-for-term into ``_dmdt_slonczewski_numpy`` / ``_dmdt_zhangli_numpy``
    # (same protocol as ``_dmdt_numpy`` transcribes ``calc_llg_dmdt``). Both
    # kernels add precession unconditionally (llg.cc:289 / :464), so the STT
    # right-hand sides include precession regardless of ``do_precession``,
    # matching the compiled behaviour. [Claude Opus 4.8]

    def _ms_nodal(self):
        """Per-owned-vertex saturation magnetisation, aligned with m columns.

        Transcribes the legacy ``Ms`` setter (``llg.py`` lines 140-142 at the
        oracle): a lumped-mass projection of the DG0 ``Ms`` onto the scalar CG1
        nodes, ``assemble(Ms * v_S1 * dx) / assemble(v_S1 * dx)``. Returned in
        coordinate-ordered (owned-vertex) layout so ``Ms_node[i]`` matches
        node column ``i`` of the ``(3, N)`` m/H arrays. For a spatially uniform
        ``Ms`` this is exactly the constant at every node.
        """
        v1 = ufl.TestFunction(self.S1)
        lumped = _assemble_owned_vector(self._Ms_dg.f * v1 * ufl.dx, self.S1)
        volume = _assemble_owned_vector(v1 * ufl.dx, self.S1)
        ms_dof = lumped / volume
        ms_field = Field(self.S1)
        ms_field.from_array(ms_dof)
        return ms_field.get_ordered_numpy_array()

    def use_slonczewski(self, J, P, d, p, Lambda=2, epsilonprime=0.0,
                        with_time_update=None):
        """Activate the Slonczewski/Xiao spin-transfer torque in the LLG.

        Parameters mirror the legacy ``use_slonczewski`` exactly:

        - ``J``: current density in A/m^2 -- a number, callable ``x -> J``,
          :class:`~finmag.field.Field`, or ``dolfinx.fem.Function`` (placed
          into the scalar CG1 space). Legacy string ``Expression`` values are
          not supported; pass a callable.
        - ``P``: polarisation in [0, 1].
        - ``d``: free-layer thickness in m.
        - ``p``: polarisation direction (3-tuple/callable), normalised to unit
          length per node.
        - ``Lambda``: the Lambda parameter in the Slonczewski/Xiao term.
        - ``epsilonprime``: strength of the secondary (field-like) torque.
        - ``with_time_update``: optional ``J(t)`` returning a spatially uniform
          current density (a number), broadcast over every node each RHS eval.
        """
        if self.do_zhangli:
            raise ValueError(
                "Cannot enable the Slonczewski spin-transfer torque: the "
                "Zhang-Li torque is already active. The two local STT modes "
                "are mutually exclusive; disable Zhang-Li first (e.g. set "
                "llg.do_zhangli = False) before configuring Slonczewski."
            )
        self.do_slonczewski = True
        self.do_zhangli = False
        self.fun_slonczewski_time_update = with_time_update

        self.Lambda = Lambda
        self.epsilonprime = epsilonprime

        J_field = Field(self.S1)
        J_field.set(J)
        self._J_slon = J_field
        # Coordinate-ordered (owned-vertex) current density, mutated in place by
        # the time-update callback, matching the legacy ``self.J[:] = J_new``.
        self.J = J_field.get_ordered_numpy_array()

        assert 0.0 <= P <= 1.0
        self.P = P
        self.d = d

        p_field = Field(self.S3)
        p_field.set(p)
        p_field.normalise()
        self.p = p_field.get_ordered_numpy_array_xxx().reshape((3, -1))
        return self

    def use_zhangli(self, J_profile=(1e10, 0, 0), P=0.5, beta=0.01,
                    using_u0=False, with_time_update=None):
        """Activate the Zhang-Li spin-transfer torque in the LLG.

        Mirrors the legacy ``use_zhangli``: ``J_profile`` is any value accepted
        by :meth:`finmag.field.Field.set` for the vector CG1 space (constant
        triple, callable, Field, Function). ``u0 = P * mu_B / e``; with
        ``using_u0`` false (default) the ``1 / (1 + beta**2)`` factor is applied.
        ``with_time_update`` is an optional ``J(t)`` returning a new J profile.
        """
        if self.do_slonczewski:
            raise ValueError(
                "Cannot enable the Zhang-Li spin-transfer torque: the "
                "Slonczewski torque is already active. The two local STT modes "
                "are mutually exclusive; disable Slonczewski first (e.g. set "
                "llg.do_slonczewski = False) before configuring Zhang-Li."
            )
        self.do_zhangli = True
        self.do_slonczewski = False
        self.fun_zhangli_time_update = with_time_update

        J_field = Field(self.S3)
        J_field.set(J_profile)
        self._J = J_field

        self.P = P
        self.beta = beta

        u0 = P * _MU_B / _E_CHARGE  # P g mu_B / (2 e Ms), g = 2 for electrons
        self.u0 = u0 if using_u0 else u0 / (1.0 + beta ** 2)

        # Precompute the lumped nodal volume used to divide the assembled
        # gradient functional. Legacy: ``nodal_volume(S3) * unit_length`` (a
        # single unit_length power converts the one spatial derivative from
        # mesh units to physical units); see ``compute_gradient_matrix``.
        self.dim = self.mesh.topology.dim
        sigma = ufl.TestFunction(self.S3)
        ones = fem.Constant(self.mesh, np.ones(3))
        raw_volume = _assemble_owned_vector(
            ufl.inner(sigma, ones) * ufl.dx, self.S3
        )
        self._nodal_volume_S3 = raw_volume * self.unit_length
        return self

    def _compute_zhangli_gradient(self):
        """Discrete ``(J . grad) m`` field, transcribing the legacy operator.

        Legacy ``compute_gradient_matrix`` assembles the matrix ``gradM`` from
        ``sum_k J_k * dot(grad(tau)[:, k], sigma) * dx`` and forms
        ``H_gradm = gradM @ m / nodal_volume_S3``. Because ``gradM @ m`` is
        exactly the linear functional evaluated at the current ``m``, this
        assembles that functional directly (no stored matrix), giving an
        identical result and picking up any time-updated ``J`` automatically.

        Returns a ``(3, N)`` coordinate-ordered array aligned with the m/H
        node columns.
        """
        sigma = ufl.TestFunction(self.S3)
        grad_m = ufl.grad(self._m_field.f)  # shape (3, gdim)
        integrand = self._J.f[0] * ufl.dot(grad_m[:, 0], sigma)
        for k in range(1, self.dim):
            integrand = integrand + self._J.f[k] * ufl.dot(grad_m[:, k], sigma)
        assembled = _assemble_owned_vector(integrand * ufl.dx, self.S3)
        h_gradm = assembled / self._nodal_volume_S3
        hg_field = Field(self.S3)
        hg_field.from_array(h_gradm)
        return hg_field.get_ordered_numpy_array_xxx().reshape((3, -1))

    def _dmdt_slonczewski_numpy(self, m, H):
        """Slonczewski dm/dt; transcribed from ``calc_llg_slonczewski_dmdt``
        (llg.cc:252) and ``slonczewski_xiao_i`` (llg.cc:207).

        ``m`` and ``H`` are ``(3, N)`` component-blocked owned nodal arrays.
        The base precession/damping/relaxation terms match ``_dmdt_numpy``
        (precession forced on, as the compiled kernel does at llg.cc:289).
        """
        alpha = self._alpha_node
        gamma_LL = self.gamma / (1.0 + alpha * alpha)

        m0, m1, m2 = m
        h0, h1, h2 = H
        mh = m0 * h0 + m1 * h1 + m2 * h2
        mm = m0 * m0 + m1 * m1 + m2 * m2

        # damping_i (llg.cc:35)
        damping_coeff = -alpha * gamma_LL
        dm0 = damping_coeff * (m0 * mh - h0 * mm)
        dm1 = damping_coeff * (m1 * mh - h1 * mm)
        dm2 = damping_coeff * (m2 * mh - h2 * mm)

        # relaxation_i (llg.cc:105); coeff 0.1/char_time == self.c
        relax_coeff = self.c * (1.0 - mm)
        dm0 += relax_coeff * m0
        dm1 += relax_coeff * m1
        dm2 += relax_coeff * m2

        # precession_i (llg.cc:74) -- unconditional in the STT kernel
        dm0 += -gamma_LL * (m1 * h2 - m2 * h1)
        dm1 += -gamma_LL * (m2 * h0 - m0 * h2)
        dm2 += -gamma_LL * (m0 * h1 - m1 * h0)

        # slonczewski_xiao_i (llg.cc:207)
        p0, p1, p2 = self.p
        Ms = self._Ms_node
        lambda_sq = self.Lambda * self.Lambda
        beta = self.J * _H_BAR / (_MU_0 * Ms * _E_CHARGE * self.d)  # llg.cc:216
        mp = m0 * p0 + m1 * p1 + m2 * p2                            # llg.cc:220
        epsilon = self.P * lambda_sq / (
            lambda_sq + 1.0 + (lambda_sq - 1.0) * mp                # llg.cc:217
        )
        perpendicular = alpha * epsilon - self.epsilonprime          # llg.cc:223
        parallel = epsilon - alpha * self.epsilonprime               # llg.cc:224
        cross0 = m1 * p2 - m2 * p1
        cross1 = m2 * p0 - m0 * p2
        cross2 = m0 * p1 - m1 * p0
        coeff = gamma_LL * beta
        # dm += gamma_LL*beta*(perp * m x p - par * m x (m x p)),
        # with m x (m x p) = mp*m - mm*p  (llg.cc:225-227)
        dm0 += coeff * (perpendicular * cross0 - parallel * (mp * m0 - mm * p0))
        dm1 += coeff * (perpendicular * cross1 - parallel * (mp * m1 - mm * p1))
        dm2 += coeff * (perpendicular * cross2 - parallel * (mp * m2 - mm * p2))

        return np.vstack((dm0, dm1, dm2))

    def _dmdt_zhangli_numpy(self, m, H, H_gradm):
        """Zhang-Li dm/dt; transcribed from ``calc_llg_zhang_li_dmdt``
        (llg.cc:406).

        ``m``, ``H`` and ``H_gradm`` are ``(3, N)`` component-blocked owned
        nodal arrays. The adiabatic + non-adiabatic STT term is written first
        (assignment, llg.cc:457-459), then precession/damping/relaxation are
        added (precession forced on, as the compiled kernel does at llg.cc:464).
        """
        alpha = self._alpha_node
        gamma_LL = self.gamma / (1.0 + alpha * alpha)
        Ms = self._Ms_node

        m0, m1, m2 = m
        h0, h1, h2 = H
        hg0, hg1, hg2 = H_gradm

        # coeff_stt = u0/(1+alpha^2)/Ms, zero where Ms == 0 (llg.cc:439-445)
        coeff_stt = self.u0 / (1.0 + alpha * alpha)
        coeff_stt = np.where(Ms == 0.0, 0.0, coeff_stt / np.where(Ms == 0.0, 1.0, Ms))

        # project H_gradm perpendicular to m (llg.cc:447-451)
        mht = m0 * hg0 + m1 * hg1 + m2 * hg2
        hp0 = hg0 - mht * m0
        hp1 = hg1 - mht * m1
        hp2 = hg2 - mht * m2

        # m x hp (llg.cc:453-455)
        mth0 = m1 * hp2 - m2 * hp1
        mth1 = m2 * hp0 - m0 * hp2
        mth2 = m0 * hp1 - m1 * hp0

        beta = self.beta
        # dm = coeff*((1+alpha*beta) hp - (beta-alpha) m x hp) (llg.cc:457-459)
        dm0 = coeff_stt * ((1.0 + alpha * beta) * hp0 - (beta - alpha) * mth0)
        dm1 = coeff_stt * ((1.0 + alpha * beta) * hp1 - (beta - alpha) * mth1)
        dm2 = coeff_stt * ((1.0 + alpha * beta) * hp2 - (beta - alpha) * mth2)

        # damping_i (llg.cc:462)
        mh = m0 * h0 + m1 * h1 + m2 * h2
        mm = m0 * m0 + m1 * m1 + m2 * m2
        damping_coeff = -alpha * gamma_LL
        dm0 += damping_coeff * (m0 * mh - h0 * mm)
        dm1 += damping_coeff * (m1 * mh - h1 * mm)
        dm2 += damping_coeff * (m2 * mh - h2 * mm)

        # relaxation_i (llg.cc:463)
        relax_coeff = self.c * (1.0 - mm)
        dm0 += relax_coeff * m0
        dm1 += relax_coeff * m1
        dm2 += relax_coeff * m2

        # precession_i (llg.cc:464) -- unconditional in the STT kernel
        dm0 += -gamma_LL * (m1 * h2 - m2 * h1)
        dm1 += -gamma_LL * (m2 * h0 - m0 * h2)
        dm2 += -gamma_LL * (m0 * h1 - m1 * h0)

        return np.vstack((dm0, dm1, dm2))

    def _require_serial(self, what):
        if self.comm.size > 1:
            raise NotImplementedError(
                "{}: multi-rank ODE state is not claimed by the deterministic "
                "DOLFINx LLG slice; the xxx state vector is rank-local "
                "owned-only. Run serially (comm size 1).".format(what)
            )
