# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import unittest
import numpy as np
import finmag.native.sundials as sundials
from finmag.util.ode import scipy_to_cvode_jtimes, scipy_to_cvode_rhs
import robertson_ode
from robertson_ode import robertson_jacobean, robertson_rhs, robertson_reset_n_evals

ROBERTSON_Y0 = np.array([1., 0., 0.])

class SundialsStiffOdeTests(unittest.TestCase):
    def test_robertson_scipy(self):
        import scipy.integrate
        y_tmp = ROBERTSON_Y0.copy()
        robertson_reset_n_evals()
        integrator = scipy.integrate.ode(robertson_rhs, jac=robertson_jacobean)
        integrator.set_initial_value(ROBERTSON_Y0)
        integrator.set_integrator("vode", method="bdf", nsteps=5000)
        integrator.integrate(1e8)
        print "Integration of the Robertson ODE until t=1e8 with scipy VODE, BDF method: %d steps" % (robertson_ode.n_rhs_evals,)
        self.assertLess(robertson_ode.n_rhs_evals, 5000)

    def test_robertson_scipy_transposed(self):
        import scipy.integrate

        y_tmp = ROBERTSON_Y0.copy()
        robertson_reset_n_evals()
        integrator = scipy.integrate.ode(robertson_rhs, jac=lambda t, y: robertson_jacobean(t,y).T)
        integrator.set_initial_value(ROBERTSON_Y0)
        integrator.set_integrator("vode", method="bdf", nsteps=5000)
        integrator.integrate(1e8)
        self.assertGreater(robertson_ode.n_rhs_evals, 5000)

    def test_robertson_sundials(self):
        robertson_reset_n_evals()
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(scipy_to_cvode_rhs(robertson_rhs), 0, ROBERTSON_Y0.copy())

        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_spils_jac_times_vec_fn(scipy_to_cvode_jtimes(robertson_jacobean))
        integrator.set_scalar_tolerances(1e-8, 1e-8)
        integrator.set_max_num_steps(5000)
        yout = np.zeros(3)
        integrator.advance_time(1e8, yout)
        print "Integration of the Robertson ODE until t=1e8 with CVODE, BDF method: %d steps" % (robertson_ode.n_rhs_evals,)

    def test_robertson_sundials_transposed(self):
        robertson_reset_n_evals()
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(scipy_to_cvode_rhs(robertson_rhs), 0, ROBERTSON_Y0.copy())

        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_spils_jac_times_vec_fn(scipy_to_cvode_jtimes(lambda t, y: robertson_jacobean(t, y).T))
        integrator.set_scalar_tolerances(1e-8, 1e-8)
        integrator.set_max_num_steps(5000)
        yout = np.zeros(3)
        try:
            integrator.advance_time(1e8, yout)
        except RuntimeError, ex:
            self.assertGreater(robertson_ode.n_rhs_evals, 5000)
            assert ex.message.find("CV_TOO_MUCH_WORK") >= 0

if __name__ == '__main__':
    unittest.main()
