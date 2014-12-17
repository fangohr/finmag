import logging
import numpy as np
from scipy.integrate import ode

EPSILON = 1e-15
log = logging.getLogger(name="finmag")


class ScipyIntegrator(object):
    def __init__(self, llg, m0, reltol=1e-6, abstol=1e-6, nsteps=10000, method="bdf", tablewriter=None, **kwargs):
        self.m_field = m0
        self.solve_for = llg.solve_for
        self.cur_t = 0.0
        self.ode = ode(self.rhs, jac=None)
        self.ode.set_integrator(
            "vode", method=method, rtol=reltol, atol=abstol, nsteps=nsteps, **kwargs)
        self.ode.set_initial_value(self.m_field.as_array(), 0)
        self._n_rhs_evals = 0
        self.tablewriter = tablewriter

    n_rhs_evals = property(
        lambda self: self._n_rhs_evals, "Number of function evaluations performed")

    def rhs(self, t, y):
        self._n_rhs_evals += 1
        return self.solve_for(y, t)

    def advance_time(self, t):
        if t == 0 and abs(t - self.cur_t) < EPSILON:
            # like sundials, scipy doesn't like integrating to 0 when
            # it was initialized with t = 0
            return

        new_m = self.ode.integrate(t)
        assert self.ode.successful()
        self.m_field.from_array(new_m)
        self.cur_t = t

    def reinit(self):
        log.debug("{}: This integrator doesn't support reinitialisation.".format(
            self.__class__.__name__))
