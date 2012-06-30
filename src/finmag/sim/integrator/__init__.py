# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import logging
import scipy.integrate
import numpy as np
from finmag.native import sundials

log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


def LLGIntegrator(llg, m0, backend="sundials", **kwargs):
    log.debug("Creating LLGIntegrator with backend {}.".format(backend))
    if backend == "scipy":
        return ScipyIntegrator(llg, m0, **kwargs)
    elif backend == "sundials":
        return SundialsIntegrator(llg, m0, **kwargs)
    else:
        raise ValueError("backend must be either scipy or sundials")


class BaseIntegrator(object):
    def run_until_relaxation(self, stopping_dmdt=ONE_DEGREE_PER_NS):
        # TODO: use the characteristic time here
        dt = 1e-15
        while True:
            # Why is self.cur_t alias CVodeGetCurrentTime not updated?
            log.debug("{}: at t={:.2}, will integrate for dt={:.2}.".format(self.__class__.__name__, self.llg.t, dt))
            prev_m = self.llg.m.copy()

            self.run_until(self.llg.t + dt)

            dm = np.abs(self.m - prev_m).reshape((3, -1))
            dm_norm = np.sqrt(dm[0] ** 2 + dm[1] ** 2 + dm[2] ** 2)
            max_dmdt_norm = float(np.max(dm_norm) / dt)
            log.debug("{}: max_dmdt_norm = {:.2} * stopping_dmdt.".format(
                self.__class__.__name__, max_dmdt_norm / stopping_dmdt))

            if max_dmdt_norm < stopping_dmdt:
                log.debug("{}: at t={:.2}, max_dmdt_norm={}<{}=stopping_dmdt\n\t->break".format(
                        self.__class__.__name__, self.llg.t, max_dmdt_norm, stopping_dmdt))
                break

            if dt < 1e-10:
                dt = dt * 1.5


class ScipyIntegrator(BaseIntegrator):
    def __init__(self, llg, m0, reltol=1e-8, abstol=1e-8, nsteps=10000, method="bdf", **kwargs):
        self.llg = llg
        self.cur_t = 0.0
        self.ode = scipy.integrate.ode(self.rhs, jac=None)
        self.m = self.llg.m[:] = m0
        self.ode.set_integrator("vode", method=method, rtol=reltol, atol=abstol, nsteps=nsteps, **kwargs)
        self.ode.set_initial_value(m0, 0)
        self._n_rhs_evals = 0

    n_rhs_evals = property(lambda self: self._n_rhs_evals, "Number of function evaluations performed")

    def rhs(self, t, y):
        self._n_rhs_evals += 1
        return self.llg.solve_for(y, t)

    def run_until(self, t):
        if t <= self.cur_t:
            return

        new_m = self.ode.integrate(t)
        assert self.ode.successful()

        self.m = new_m


class SundialsIntegrator(BaseIntegrator):
    def __init__(self, llg, m0, reltol=1e-8, abstol=1e-8,
                 nsteps=10000, method="bdf_gmres_prec_id"):
        assert method in ("adams", "bdf_diag",
                          "bdf_gmres_no_prec", "bdf_gmres_prec_id")
        self.llg = llg
        self.cur_t = 0.0
        self.m = m0.copy()

        if method == "adams":
            integrator = sundials.cvode(sundials.CV_ADAMS, sundials.CV_FUNCTIONAL)
        else:
            integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        self.integrator = integrator

        integrator.init(llg.sundials_rhs, 0, self.m)

        if method == "bdf_diag":
            integrator.set_linear_solver_diag()
        elif method == "bdf_gmres_no_prec":
            integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
            integrator.set_spils_jac_times_vec_fn(self.llg.sundials_jtimes)
        elif method == "bdf_gmres_prec_id":
            integrator.set_linear_solver_sp_gmr(sundials.PREC_LEFT)
            integrator.set_spils_jac_times_vec_fn(self.llg.sundials_jtimes)
            integrator.set_spils_preconditioner(llg.sundials_psetup, llg.sundials_psolve)

        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)

    def run_until(self, t):
        if t <= self.cur_t:
            return

        self.integrator.advance_time(t, self.m)
        self.llg.m = self.m

    def reinit(self):
        """reinit() calls CVodeReInit.

        Useful if there is a drastic (non-continuous) change in the right hand side of the ODE.
        By calling this function, we inform the integrator that it should not assuming smoothness
        of the RHS. Should be called when we change the applied field, abruptly, for example.
        """
        self.integrator.reinit(self.integrator.get_current_time(), self.m)

    n_rhs_evals = property(lambda self: self.integrator.get_num_rhs_evals(), "Number of function evaluations performed")
