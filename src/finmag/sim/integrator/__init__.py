# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import os
import logging
import itertools
import scipy.integrate
import numpy as np
import dolfin as df
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
    def run_until_relaxation(self, save_snapshots=False, filename=None, save_every=100e-12, save_final_snapshot=True,
                             stopping_dmdt=ONE_DEGREE_PER_NS, dmdt_increased_counter_limit=20, dt_limit=1e-10):
        """
        Run integration until the maximum |dm/dt| is smaller than the
        threshold value stopping_dmdt (which is one degree per
        nanosecond per default).

        As a precaution against running an infinite amount of time when
        |dm/dt| - stopping_dmdt doesn't convergence (because of badly
        chosen tolerances?), the integration will stop if |dm/dt|
        increases `dmdt_increased_counter_limit` times during the integration
        (default value: 20). The maximum allowed timestep per integration step
        can be controlled via `dt_limit`.

        If save_snapshots is True (default: False) then a series of snapshots
        is saved to `filename` (which must be specified in this case). If
        `filename` contains directory components then these are created if they
        do not already exist. A snapshot is saved every `save_every` seconds
        (default: 100e-12, i.e. every 100 picoseconds). It should be noted that
	the true timestep at which the snapshot is saved may deviate from slightly
 	from the exact value due to the way the time integrators work.
        Usually, one last snapshot is saved after the relaxation is finished (or
        was stopped). This can be disabled by setting save_final_snapshot to False
        (default: True).

        """
        if save_snapshots == True:
            if filename == '':
                raise ValueError("If save_snapshots is True, filename must be a non-empty string.")
            else:
                ext = os.path.splitext(filename)[1]
                if ext != '.pvd':
                    raise ValueError("File extension for vtk snapshot file must be '.pvd', but got: '{}'".format(ext))
            f = df.File(filename, 'compressed')
        else:
            if filename != '':
                log.warning("Value of save_snapshot is False, but filename is given anyway: '{}'. Ignoring...".format(filename))

        dt = 1e-14 # initial timestep (TODO: use the characteristic time here)

        dt_increment_multi = 1.5;
        dmdt_increased_counter = 0;

        #ct = itertools.count()  # we need a possibly unlimited counter for saving snapshots
        #cur_count = ct.next()
	cur_count = 0

        def _do_save_snapshot():
            log.debug("Saving snapshot at timestep t={:.4g} to file '{}' (snapshot #{})".format(self.llg.t, filename, cur_count))
            # TODO: Can we somehow store information about the current timestep in either the .pvd/.vtu file itself, or in the filenames?
            #       Unfortunately, it seems as if the filenames of the *.vtu files are generated automatically.
            f << self.llg._m

        last_max_dmdt_norm = 1e99
        while True:
            prev_m = self.llg.m.copy()
	    next_stop = self.llg.t + dt

            # If in the next step we would cross a timestep where a snapshot should be saved, run until
            # that timestep, save the snapshot, and then continue.
            while save_snapshots and (next_stop >= cur_count*save_every):
                self.run_until(cur_count*save_every)
                _do_save_snapshot()
                cur_count += 1

            # Why is self.cur_t alias CVodeGetCurrentTime not updated?
            self.run_until(next_stop)

            dm = np.abs(self.m - prev_m).reshape((3, -1))
            dm_norm = np.sqrt(dm[0] ** 2 + dm[1] ** 2 + dm[2] ** 2)
            max_dmdt_norm = float(np.max(dm_norm) / dt)

            if max_dmdt_norm < stopping_dmdt:
                log.debug("{}: Stopping at t={:.3g}, with last_dmdt={:.3g}, smaller than stopping_dmdt={:.3g}.".format(
                    self.__class__.__name__, self.llg.t, max_dmdt_norm, float(stopping_dmdt)))
                break

            if dt < dt_limit / dt_increment_multi:
                dt *= dt_increment_multi
            else:
                dt = dt_limit

            log.debug("{}: t={:.3g}, last_dmdt={:.3g} * stopping_dmdt, next dt={:.3g}.".format(
                self.__class__.__name__, self.llg.t, max_dmdt_norm/stopping_dmdt, dt))

            if max_dmdt_norm > last_max_dmdt_norm:
                dmdt_increased_counter += 1
                log.debug("{}: dmdt {:.2f} times larger than last time (counting {}/{}).".format(
                    self.__class__.__name__, max_dmdt_norm/last_max_dmdt_norm,
                    dmdt_increased_counter, dmdt_increased_counter_limit))
            last_max_dmdt_norm = max_dmdt_norm
            if dmdt_increased_counter >= dmdt_increased_counter_limit:
                log.warning("{}: Stopping after it increased {} times.".format(
                    self.__class__.__name__, dmdt_increased_counter_limit))
                break

        if save_snapshots and save_final_snapshot:
            _do_save_snapshot()

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
