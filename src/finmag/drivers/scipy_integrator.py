"""SciPy VODE/BDF driver, ported directly to DOLFINx.

This is the direct DOLFINx port of the legacy
``finmag.drivers.scipy_integrator.ScipyIntegrator``. The public
``advance_time``/tolerance interface is unchanged, but the ODE state vector
fed to and read back from ``scipy.integrate.ode`` now unambiguously uses the
component-blocked, coordinate-ordered ``xxx`` array
(``Field.get_ordered_numpy_array_xxx`` / ``set_with_ordered_numpy_array_xxx``)
in both directions.

The legacy driver seeded the ODE state with ``m_field.as_array()`` (raw
backend dof order) and wrote results back with ``m_field.from_array(...)``
(also raw order), while ``rhs`` passed the state straight through to
``llg.solve_for``, which has always interpreted its argument as the ``xxx``
ordering. On the legacy DOLFIN meshes these two orderings happened to
coincide often enough that the mismatch went unnoticed; DOLFINx's blocked
dof layout makes the two orderings genuinely different in general (see
``dev/dolfinx/porting_map.md``, "Field `xxx` consumers"). This port resolves
that recorded ambiguity by routing both directions through the explicit
``xxx`` ordering rather than preserving it.

Scope: serial-only (inherited from ``LLG.solve``/``solve_for``'s own
multi-rank guard); no ``solve_ivp`` migration, no new backend-selection
mechanism.
"""

import logging

import numpy as np
from scipy.integrate import ode

EPSILON = 1e-15
log = logging.getLogger(name="finmag")


class ScipyIntegrator(object):
    """Drives an ``LLG`` with ``scipy.integrate.ode`` (VODE/BDF or Adams).

    ``m0`` is the :class:`~finmag.field.Field` whose component-blocked
    ``xxx`` state is the dynamic degrees of freedom; it is read at
    construction/``reinit`` time and written back after every successful
    ``advance_time``.
    """

    def __init__(self, llg, m0, reltol=1e-6, abstol=1e-6, nsteps=10000,
                 method="bdf", tablewriter=None, **kwargs):
        self.llg = llg
        self.m_field = m0
        self.solve_for = llg.solve_for
        self.reltol = reltol
        self.abstol = abstol
        self.nsteps = nsteps
        self.method = method
        self.integrator_kwargs = kwargs
        self.tablewriter = tablewriter

        self.cur_t = 0.0
        self._n_rhs_evals = 0

        self.ode = self._new_ode()
        self.ode.set_initial_value(
            self.m_field.get_ordered_numpy_array_xxx(), self.cur_t)

    def _new_ode(self):
        """Build a fresh ``scipy.integrate.ode`` object with our settings."""
        solver = ode(self.rhs, jac=None)
        solver.set_integrator(
            "vode", method=self.method, rtol=self.reltol, atol=self.abstol,
            nsteps=self.nsteps, **self.integrator_kwargs)
        return solver

    n_rhs_evals = property(
        lambda self: self._n_rhs_evals, "Number of function evaluations performed")

    def rhs(self, t, y):
        self._n_rhs_evals += 1
        return self.solve_for(y, t)

    def advance_time(self, t):
        if t < self.cur_t:
            raise ValueError(
                "{}: backward integration is not supported (requested "
                "t={}, current t={}).".format(
                    self.__class__.__name__, t, self.cur_t))

        if t == 0 and abs(t - self.cur_t) < EPSILON:
            # like sundials, scipy doesn't like integrating to 0 when
            # it was initialized with t = 0
            return True

        new_m = self.ode.integrate(t)
        if not self.ode.successful():
            raise RuntimeError(
                "{}: scipy.integrate.ode (method={!r}) failed to "
                "integrate to t={}.".format(
                    self.__class__.__name__, self.method, t))

        self.m_field.set_with_ordered_numpy_array_xxx(new_m)
        self.cur_t = t
        return True

    def reinit(self):
        """Re-seed the ODE integrator from the current field state.

        The dynamic degrees of freedom (the ``xxx`` state) are read fresh
        from ``self.m_field`` at the current time, so any change made to the
        field from outside this driver (e.g. a modified applied field
        propagating into ``H_eff``) takes effect on the next
        ``advance_time`` call. This mirrors the Sundials reinit contract
        (see ``sundials_reinit_test.py``): the state itself is unchanged
        unless it was externally modified, but the integrator forgets its
        internal step-size/order history and starts exploring the
        right-hand side afresh. ``scipy.integrate.ode`` has no reinit
        primitive of its own, so this rebuilds the underlying VODE
        integrator object.
        """
        log.debug(
            "{}: reinitialising from the current field state at t={}.".format(
                self.__class__.__name__, self.cur_t))
        self.ode = self._new_ode()
        self.ode.set_initial_value(
            self.m_field.get_ordered_numpy_array_xxx(), self.cur_t)
