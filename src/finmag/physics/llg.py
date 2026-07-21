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

Scope of this slice: scalar ``Ms``, scalar ``alpha``, ``gamma``, ``set_m``,
``solve`` and ``solve_for``, driven entirely from the ported ``EffectiveField``
registry. The state vector for ``solve``/``solve_for``/``m`` setters is the
component-blocked, coordinate-ordered ``xxx`` array (rank-local owned dofs);
``H_eff`` is routed into the identical ordering before the node-local update.
Native Sundials/CVODE preconditioning and Jacobian paths, spin-transfer torque
(Slonczewski, Zhang-Li), thermal dynamics, and multi-rank ODE state are out of
scope and raise ``NotImplementedError`` by name when requested.
"""

import logging

import numpy as np
from dolfinx import fem

import finmag.util.consts as consts
from finmag.field import Field
from finmag.physics.effective_field import EffectiveField

# default settings for logger 'finmag' set in __init__.py
logger = logging.getLogger(name="finmag")


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
        self.alpha = 0.5  # scalar Gilbert damping constant

        self.gamma = consts.gamma
        self.c = 1e11  # 1/s numerical scaling correction
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self._Ms_dg = Field(self.DG, name="Saturation magnetisation")
        self.Ms = 8.6e5  # A/m saturation magnetisation
        self._m_field = Field(self.S3, name="m")
        self._dmdt = Field(self.S3, name="dmdt")
        self._pins = np.array([], dtype="int")

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

    def set_alpha(self, value):
        """Set the scalar Gilbert damping constant :math:`\\alpha`.

        Only spatially uniform (scalar) damping is supported in this slice.
        """
        if not np.isscalar(value):
            raise NotImplementedError(
                "spatially varying alpha is deferred from the deterministic "
                "DOLFINx LLG slice; pass a scalar"
            )
        self.alpha = float(value)

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

        # Accumulate the total effective field from the registry, then route it
        # into the same coordinate-ordered component-blocked layout as m so the
        # node-local update below is unambiguous.
        self.effective_field.update(t)
        H_eff_field = Field(self.S3)
        H_eff_field.from_array(self.effective_field.H_eff)

        m = self._m_field.get_ordered_numpy_array_xxx().reshape((3, -1))
        H = H_eff_field.get_ordered_numpy_array_xxx().reshape((3, -1))

        dmdt = self._dmdt_numpy(m, H)

        if self._pins.size:
            dmdt[:, self._pins] = 0.0

        dmdt = dmdt.reshape(-1)
        self._dmdt.set_with_ordered_numpy_array_xxx(dmdt)
        return dmdt

    def _dmdt_numpy(self, m, H):
        """Node-local LLG right-hand side; transcribed from ``calc_llg_dmdt``.

        ``m`` and ``H`` are ``(3, N)`` component-blocked owned nodal arrays.
        Returns a fresh ``(3, N)`` dm/dt array.
        """
        alpha = self.alpha
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

        This is backend-neutral (it does not touch native CVODE); the SciPy
        driver slice can reuse it. The CVODE-specific preconditioner and
        Jacobian-times-vector callbacks below remain unported.
        """
        ydot[:] = self.solve_for(y, t)
        return 0

    # -- explicitly deferred surfaces --------------------------------------

    def sundials_jtimes(self, *args, **kwargs):
        raise NotImplementedError(
            "native Sundials/CVODE Jacobian-times-vector (calc_llg_jtimes) is "
            "not ported to the deterministic DOLFINx LLG slice"
        )

    def sundials_psetup(self, *args, **kwargs):
        raise NotImplementedError(
            "native Sundials/CVODE preconditioner setup is not ported to the "
            "deterministic DOLFINx LLG slice"
        )

    def sundials_psolve(self, *args, **kwargs):
        raise NotImplementedError(
            "native Sundials/CVODE preconditioner solve is not ported to the "
            "deterministic DOLFINx LLG slice"
        )

    def use_slonczewski(self, *args, **kwargs):
        raise NotImplementedError(
            "Slonczewski spin-transfer torque is out of scope for the "
            "deterministic DOLFINx LLG slice"
        )

    def use_zhangli(self, *args, **kwargs):
        raise NotImplementedError(
            "Zhang-Li spin-transfer torque is out of scope for the "
            "deterministic DOLFINx LLG slice"
        )

    def _require_serial(self, what):
        if self.comm.size > 1:
            raise NotImplementedError(
                "{}: multi-rank ODE state is not claimed by the deterministic "
                "DOLFINx LLG slice; the xxx state vector is rank-local "
                "owned-only. Run serially (comm size 1).".format(what)
            )
