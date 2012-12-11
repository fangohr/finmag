import time
import logging
import scipy.integrate
import numpy as np
import dolfin as df
from finmag.native import sundials

log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


def LLGIntegrator(llg, m0, backend="sundials", tablewriter=None, **kwargs):
    # XXX TODO: Passing the tablewriter argument on like this is a
    #           complete hack and this should be refactored. The same
    #           is true with saving snapshots. Neither saving average
    #           fields nor VTK snapshots should probably happen in
    #           this class but rather in the Simulation class (?).
    #             -- Max, 11.12.2012
    #
    log.debug("Creating LLGIntegrator with backend {}.".format(backend))
    if backend == "scipy":
        return ScipyIntegrator(llg, m0, tablewriter=tablewriter, **kwargs)
    elif backend == "sundials":
        return SundialsIntegrator(llg, m0, tablewriter=tablewriter, **kwargs)
    else:
        raise ValueError("backend must be either scipy or sundials")


class BaseIntegrator(object):
    def run_until_relaxation(self, save_snapshots=False, filename=None, save_every=100e-12, save_final_snapshot=True,
                             stopping_dmdt=ONE_DEGREE_PER_NS, dmdt_increased_counter_limit=50, dt_limit=1e-10):
        """
        Run integration until the maximum |dm/dt| is smaller than the
        threshold value stopping_dmdt (which is one degree per
        nanosecond per default).

        As a precaution against running an infinite amount of time when
        |dm/dt| - stopping_dmdt doesn't convergence (because of badly
        chosen tolerances?), the integration will stop if |dm/dt|
        increases `dmdt_increased_counter_limit` times during the integration
        (default value: 50). The maximum allowed timestep per integration step
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
        dt = 1e-14 # initial timestep (TODO: use the characteristic time here)

        dt_increment_multi = 1.5;
        dmdt_increased_counter = 0;

        if save_snapshots:
            f = df.File(filename, 'compressed')

        cur_count = 0  # current snapshot count
        start_time = self.llg.t  # start time of the integration; needed for snapshot saving
        last_max_dmdt_norm = 1e99
        while True:
            prev_m = self.llg.m.copy()
            next_stop = self.llg.t + dt

            # If in the next step we would cross a timestep where a snapshot should be saved, run until
            # that timestep, save the snapshot, and then continue.
            while save_snapshots and (next_stop >= start_time+cur_count*save_every):
                self.run_until(cur_count*save_every)
                self._do_save_snapshot(f, cur_count, filename, save_averages=True)
                cur_count += 1

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
            self._do_save_snapshot(f, cur_count, filename, save_averages=True)

    def reinit(self):
        raise NotImplementedError("No reinit() method is implemented for this integrator: {}".format(self.__class__.__name__))

    def _do_save_snapshot(self, f, cur_count, filename, save_averages=True):
        # TODO: Can we somehow store information about the current timestep in either the .pvd/.vtu file itself, or in the filenames?
        #       Unfortunately, it seems as if the filenames of the *.vtu files are generated automatically.
        t0 = time.time()
        f << self.llg._m
        t1 = time.time()
        log.debug("Saving snapshot #{} at timestep t={:.4g} to file '{}' (saving took {:.3g} seconds).".format(cur_count, self.llg.t, filename, t1-t0))
        if save_averages:
            if self.tablewriter:
                log.debug("Saving average field values (in integrator).")
                self.tablewriter.save()
            else:
                log.warning("Cannot save average fields because no Tablewriter is present in integrator.")


class ScipyIntegrator(BaseIntegrator):
    def __init__(self, llg, m0, reltol=1e-8, abstol=1e-8, nsteps=10000, method="bdf", tablewriter=None, **kwargs):
        self.llg = llg
        self.cur_t = 0.0
        self.ode = scipy.integrate.ode(self.rhs, jac=None)
        self.m = self.llg.m[:] = m0
        self.ode.set_integrator("vode", method=method, rtol=reltol, atol=abstol, nsteps=nsteps, **kwargs)
        self.ode.set_initial_value(m0, 0)
        self._n_rhs_evals = 0
        self.tablewriter = tablewriter

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
        self.cur_t = t

class SundialsIntegrator(BaseIntegrator):
    """
    Sundials time integrator.

    Attributes:
        cur_t       The current internal time reached by the solver.
                    It's not meant to represent the actual simulation time.

    """
    def __init__(self, llg, m0, reltol=1e-8, abstol=1e-8,
                 nsteps=10000, method="bdf_gmres_prec_id", tablewriter=None):
        assert method in ("adams", "bdf_diag",
                          "bdf_gmres_no_prec", "bdf_gmres_prec_id")
        self.llg = llg
        self.cur_t = 0.0
        self.m = m0.copy()
        self.tablewriter = tablewriter

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
        self.cur_t = self.integrator.get_current_time()
        self.llg.m = self.m

    def reinit(self):
        """reinit() calls CVodeReInit.

        Useful if there is a drastic (non-continuous) change in the right hand side of the ODE.
        By calling this function, we inform the integrator that it should not assuming smoothness
        of the RHS. Should be called when we change the applied field, abruptly, for example.
        """
        self.integrator.reinit(self.integrator.get_current_time(), self.m)

    n_rhs_evals = property(lambda self: self.integrator.get_num_rhs_evals(), "Number of function evaluations performed")
