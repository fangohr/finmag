# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import scipy.integrate
import numpy as np

def LLGIntegrator(llg, m0, backend="scipy", **kwargs):
    if backend=="scipy":
        return ScipyIntegrator(llg, m0, **kwargs)
    elif backend=="sundials":
        return SundialsIntegrator(llg, m0, **kwargs)
    else:
        raise ValueError("backend must be either scipy or sundials")

class BaseIntegrator(object):
    def run_until_relaxation(self, stop_tol=1e-4):
        # TODO: use the characteristic time here
        next_t = 1e-15
        prev_m = self.llg.m.copy()
        while True:
            dt = next_t * 0.1
            next_t += dt
            self.run_until(next_t)
            # TODO: use a better norm for stopping
            diff = np.max(np.abs(prev_m - self.dof))
            prev_m = self.m.copy()
            if diff < stop_tol:
                break

class ScipyIntegrator(BaseIntegrator):
    def __init__(self, llg, m0, reltol=1e-8, abstol=1e-8, nsteps=10000, method="bdf", **kwargs):
        self.llg = llg
        self.cur_t = 0
        self.ode = scipy.integrate.ode(llg.solve_for, jac=None)
        self.llg.m[:] = m0
        self.ode.set_integrator("vode", method=method, rtol=reltol, atol=abstol, nsteps=nsteps, **kwargs)
        self.ode.set_initial_value(m0, 0)

    def run_until(self, t):
        if t <= self.cur_t:
            return

        new_m = self.ode.integrate(t)
        assert self.ode.successful()

        self.m = new_m

class SundialsIntegrator(BaseIntegrator):
    def __init__(self, llg, m0, reltol=1e-8, abstol=1e-8, nsteps=10000, method="bdf_gmres_no_prec"):
        assert method in ("adams", "bdf_diag", "bdf_gmres_no_prec")
        self.llg = llg
        self.cur_t = 0
        self.m = m0.copy()

        if method == "adams":
            integrator = sundials.cvode(sundials.CV_ADAMS, sundials.CV_FUNCTIONAL)
        else:
            integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        self.integrator = integrator

        if method=="bdf_diag":
            integrator.init(scipy_to_cvode_rhs(self.scipy_rhs), 0, self.llg.m.copy())
            integrator.set_linear_solver_diag()
        elif method=="bdf_gmres_no_prec":
            integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
            integrator.set_splis_jac_times_vec_fn(self.llg.sundials_jtimes)

        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)

    def run_until(self, t):
        if t <= self.cur_t:
            return

        integrator.advance_time(t, self.m)
