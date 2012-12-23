import logging
from finmag.native import sundials
from finmag.integrators.common import run_until_relaxation

log = logging.getLogger(name='finmag')


class SundialsIntegrator(object):
    """
    Sundials time integrator. We always start integration from t = 0.

    Attributes:
        cur_t       The time up to which integration has been carried out.
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

        self.set_max_steps(nsteps)

    def set_max_steps(self, nsteps):
        """Sets the maximum number of steps that will be done for time integration."""
        self.nsteps = nsteps
        self.integrator.set_max_num_steps(self.nsteps)

    def get_max_steps(self):
        """Sets the maximum number of steps that will be done for time integration."""
        return self.nsteps

    def run_until(self, t, max_steps=None):
        """
        *Arguments*

        t : float

            Target time to integrate to

        max_steps : integer

            maximum number of steps for time integration

        Returns ``True`` or ``False`` depending on whether target time ``t``
        has been reached.

        Given a target time t, this function integrates towards ``t``. If
        ``max_steps`` is given and the number of ``max_steps`` steps
        for the integration are reached, we interrupt the calculation and
        return False.

        If tout is reached within the number of allowed steps, it will return
        True.

        If max_steps was not provided, and the maximum number of steps is
        reached, we raise a RuntimeError (as this is almost certainly not
        what the user intended).
        """

        # The following check is required because sundials does not like to
        # integrate up to t=0, if the cvode solver was initialised for t=0.
        if t <= self.cur_t and t == 0:
            return
        # if t <= self.cur_t and this is not the value with which we started,
        # we should complain:
        elif t <= self.cur_t:
            raise RuntimeError("t=%g, self.cur_t=%g -- why are we integrating into the past?")

        # if max_steps given, set this with the integrator, otherwise use
        # value we have currently (Not sure this is good. Maybe should use
        # default otherwise. actually, would be better to keep attribute in
        # integrator class that keeps track of current max_steps. XXX)

        # save current max steps
        previous_max_steps = self.get_max_steps()

        # if given, set new value of max_steps
        if max_steps != None:
            self.set_max_steps(max_steps)

        try:
            self.integrator.advance_time(t, self.m)
        except RuntimeError, msg:
            # if we have reached max_num_steps, the error message will read
            # something like expected_error = "Error in CVODE:CVode
            # (CV_TOO_MUCH_WORK): At t = 0.258733, mxstep steps taken before
            # reaching tout.'"
            if "CV_TOO_MUCH_WORK" in msg.message:
                if max_steps != None:
                    # max_steps has been given by the user
                    reached_tout = False
                    # in this case, we have integrated up to cvode's inernal time.
                    # So we need to get this value:
                    self.cur_t = self.integrator.get_current_time()
                else:  # max_steps not given
                    #  If we get into this branch, it means cvode has returned
                    #  after nsteps, but the user has not given a particular
                    #  value of nsteps. This isgenerally not desired, and
                    #  we warn quite strongly about it:
                    msg = "The integrator has reached nsteps=%d " % \
                        self.get_max_steps()
                    msg += "and the corresponding time is %g whereas " %\
                        self.integrator.get_current_time()
                    msg += "the requested time is %g. " % t
                    msg += "You can increase the number of maximum steps " +\
                        "if that is really what you need using " +\
                        "integrator.set_max_steps()."
                    self.cur_t = self.integrator.get_current_time()
                    reached_tout = False  # not used, but this would be the rigth value
                    raise RuntimeError(msg)
            else:  # Any other exception is unexpected, so raise error again
                raise
        else:  # if we succeeded with time integration to t
            self.cur_t = t
            reached_tout = True

        # in any case: put integrated degrees of freedom from cvode object
        # back into llg object
        self.llg.m = self.m

        # set previous value of max_steps again
        self.set_max_steps(previous_max_steps)

        return reached_tout

    def reinit(self):
        """
        Reinitialise memory for CVODE.

        Useful if there is a drastic (non-continuous) change in the right hand side of the ODE.
        By calling this function, we inform the integrator that it should not assuming smoothness
        of the RHS. Should be called when we change the applied field, abruptly, for example.
        """
        log.debug("Re-initialising CVODE integrator")
        self.integrator.reinit(self.integrator.get_current_time(), self.m)

    n_rhs_evals = property(lambda self: self.integrator.get_num_rhs_evals(), "Number of function evaluations performed")

    def stats(self):
        """ Return integrator stats as dictionary. Keys are
        nsteps, nfevals, nlinsetups, netfails, qlast, qcur, hinused, hlast, hcur,
        tcur

        and the meanings are (from CVODE 2.7 documentation, section 4.5, page 46)

        nsteps                   (long int) number of steps taken by cvode.
        nfevals                  (long int) number of calls to the user's f function.
        nlinsetups               (long int) number of calls made to the linear solver setup function.
        netfails                 (long int) number of error test failures.
        qlast                    (int) method order used on the last internal step.
        qcur                     (int) method order to be used on the next internal step.
        hinused                  (realtype) actual value of initial step size.
        hlast                    (realtype) step size taken on the last internal step.
        hcur                     (realtype) step size to be attempted on the next internal step.
        tcur                     (realtype) current internal time reached.

        """

        stats = self.integrator.get_integrator_stats()
        nsteps, nfevals, nlinsetups, netfails, qlast, qcur, hinused, hlast, hcur, tcur = stats
        d = {'nsteps': nsteps,
             'nfevals': nfevals,
             'nlinsetups': nlinsetups,
             'netfails': netfails,
             'qlast': qlast,
             'qcur': qcur,
             'hinused': hinused,
             'hlast': hlast,
             'hcur': hcur,
             'tcur': tcur
             }
        return d

    run_until_relaxation = run_until_relaxation
