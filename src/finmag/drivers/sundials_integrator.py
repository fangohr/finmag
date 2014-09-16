import logging
from finmag.native import sundials

EPSILON = 1e-15

log = logging.getLogger(name='finmag')


class SundialsIntegrator(object):

    """
    Sundials time integrator. We always start integration from t = 0.

    Attributes:
        cur_t       The time up to which integration has been carried out.
    """

    def __init__(self, llg, m0, t0=0.0, reltol=1e-6, abstol=1e-6,
                 nsteps=10000, method="bdf_gmres_prec_id", tablewriter=None):
        assert method in ("adams", "bdf_diag",
                          "bdf_gmres_no_prec", "bdf_gmres_prec_id")
        self.llg = llg
        self.cur_t = t0
        self.m = m0.copy()
        self.tablewriter = tablewriter

        if method == "adams":
            integrator = sundials.cvode(
                sundials.CV_ADAMS, sundials.CV_FUNCTIONAL)
        else:
            integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        self.integrator = integrator

        integrator.init(llg.sundials_rhs, self.cur_t, self.m)

        if method == "bdf_diag":
            integrator.set_linear_solver_diag()
        elif method == "bdf_gmres_no_prec":
            integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
            integrator.set_spils_jac_times_vec_fn(self.llg.sundials_jtimes)
        elif method == "bdf_gmres_prec_id":
            integrator.set_linear_solver_sp_gmr(sundials.PREC_LEFT)
            integrator.set_spils_jac_times_vec_fn(self.llg.sundials_jtimes)
            integrator.set_spils_preconditioner(
                llg.sundials_psetup, llg.sundials_psolve)

        integrator.set_scalar_tolerances(reltol, abstol)
        self.max_steps = nsteps

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value):
        self._max_steps = value
        self.integrator.set_max_num_steps(value)

    def advance_time(self, t):
        """
        *Arguments*

        t : float

            Target time to integrate to

        Returns ``True`` or ``False`` depending on whether target time ``t``
        has been reached.

        Given a target time ``t``, this function integrates towards ``t``. If
        ``max_steps`` was set and the number of steps for the integration are
        reached, we interrupt the calculation and return False.

        If ``t`` is reached within the number of allowed steps, it will return
        True.

        """
        # The following check is required because sundials does not like to
        # integrate up to t=0, if the cvode solver was initialised for t=0.
        if t == 0 and abs(t - self.cur_t) < EPSILON:
            return True
        # if t <= self.cur_t and this is not the value with which we started,
        # we should complain:
        elif t <= self.cur_t:
            raise RuntimeError(
                "t={:.3g}, self.cur_t={:.3g} -- why are we integrating "
                "into the past?".format(t, self.cur_t))

        try:
            self.integrator.advance_time(t, self.m)
        except RuntimeError, msg:
            # if we have reached max_num_steps, the error message will read
            # something like "Error in CVODE:CVode (CV_TOO_MUCH_WORK):
            # At t = 0.258733, mxstep steps taken before reaching tout."
            if "CV_TOO_MUCH_WORK" in msg.message:
                # we have integrated up to cvode's internal time
                self.cur_t = self.integrator.get_current_time()

                log.error("The integrator has reached its maximum of {} steps.\n"
                          "The time is t = {} whereas you requested t = {}.\n"
                          "You can increase the maximum number of steps if "
                          "you really need to with integrator.max_steps = n.".format(
                              self.max_steps, self.integrator.get_current_time(), t))
                # not used, but this would be the right value
                reached_tout = False
                raise
            else:
                reached_tout = False
                raise
        else:
            self.cur_t = t
            reached_tout = True

        # in any case: put integrated degrees of freedom from cvode object
        # back into llg object
        # Weiwei: change the default m to sundials_m since sometimes we need to
        # extend the default equation.
        self.llg.sundials_m = self.m
        return reached_tout

    def advance_steps(self, steps):
        """
        Run the integrator for `steps` internal steps.

        """
        old_max_steps = self.max_steps
        self.max_steps = steps
        try:
            # we can't tell sundials to run a certain number of steps
            # so we try integrating for a very long time but set it to
            # stop after the specified number of steps
            self.integrator.advance_time(self.cur_t + 1, self.m)
        except RuntimeError, msg:
            if "CV_TOO_MUCH_WORK" in msg.message:
                pass  # this is the error we expect
            else:
                raise
        self.cur_t = self.integrator.get_current_time()
        self.llg.sundials_m = self.m
        self.max_steps = old_max_steps

    def reinit(self):
        """
        Reinitialise memory for CVODE.

        Useful if there is a drastic (non-continuous) change in the right hand side of the ODE.
        By calling this function, we inform the integrator that it should not assuming smoothness
        of the RHS. Should be called when we change the applied field, abruptly, for example.
        """
        log.debug("Re-initialising CVODE integrator.")
        self.integrator.reinit(self.cur_t, self.m)

    n_rhs_evals = property(lambda self: self.integrator.get_num_rhs_evals(
    ), "Number of function evaluations performed")

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
