# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import unittest
import numpy as np
import warnings
import pytest
import finmag.native.sundials as sundials
from finmag.util.ode import scipy_to_cvode_jtimes, scipy_to_cvode_rhs
from . import robertson_ode
from .robertson_ode import robertson_jacobean, robertson_rhs, robertson_reset_n_evals

ROBERTSON_Y0 = np.array([1., 0., 0.])


class SundialsStiffOdeTests(unittest.TestCase):

    def test_robertson_scipy(self):
        import scipy.integrate
        robertson_reset_n_evals()
        # Legacy SciPy 0.19 solved this with ode(..., "vode"), but modern
        # SciPy's real-valued VODE path no longer behaves reliably on this
        # Robertson problem. Use solve_ivp's current stiff BDF path as the
        # SciPy reference instead of preserving the old wrapper-specific
        # behaviour. [Codex GPT-5.4]
        sol = scipy.integrate.solve_ivp(
            robertson_rhs,
            (0, 1e8),
            ROBERTSON_Y0,
            method="BDF",
            jac=robertson_jacobean)
        print("Integration of the Robertson ODE until t=1e8 with scipy solve_ivp/BDF: %d steps" % (robertson_ode.n_rhs_evals,))
        self.assertTrue(sol.success)
        self.assertAlmostEqual(sol.t[-1], 1e8)
        self.assertAlmostEqual(np.sum(sol.y[:, -1]), 1.0, places=10)
        # Keep a loose work-budget sanity check on the modern SciPy stiff
        # solver path without depending on the legacy VODE-specific step
        # counter semantics. [Codex GPT-5.4]
        self.assertLess(sol.nfev, 2000)
        self.assertTrue(np.allclose(
            sol.y[:, -1],
            np.array([2.08e-05, 8.33e-11, 9.99979176e-01]),
            rtol=5e-3,
            atol=1e-12))

    @pytest.mark.xfail(
        strict=True,
        reason="Known pathological case: transposing the Robertson Jacobian breaks SciPy VODE/BDF convergence on the current stack")
    def test_robertson_scipy_transposed_fails_with_excess_work(self):
        import scipy.integrate

        robertson_reset_n_evals()
        integrator = scipy.integrate.ode(
            robertson_rhs, jac=lambda t, y: robertson_jacobean(t, y).T)
        integrator.set_initial_value(ROBERTSON_Y0)
        integrator.set_integrator("vode", method="bdf", nsteps=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yout = integrator.integrate(1e8)
        # This asserts the "best case" contract; the xfail records that the
        # transposed Jacobian currently does not satisfy it. [Codex GPT-5.4]
        self.assertTrue(integrator.successful())
        self.assertAlmostEqual(integrator.t, 1e8)
        self.assertAlmostEqual(np.sum(yout), 1.0, places=10)
        self.assertLess(robertson_ode.n_rhs_evals, 5000)
        self.assertTrue(np.allclose(
            yout,
            np.array([2.08e-05, 8.33e-11, 9.99979176e-01]),
            rtol=5e-3,
            atol=1e-12))

    def test_robertson_sundials(self):
        robertson_reset_n_evals()
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(
            scipy_to_cvode_rhs(robertson_rhs), 0, ROBERTSON_Y0.copy())

        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_spils_jac_times_vec_fn(
            scipy_to_cvode_jtimes(robertson_jacobean))
        integrator.set_scalar_tolerances(1e-8, 1e-8)
        integrator.set_max_num_steps(5000)
        yout = np.zeros(3)
        integrator.advance_time(1e8, yout)
        print("Integration of the Robertson ODE until t=1e8 with CVODE, BDF method: %d steps" % (robertson_ode.n_rhs_evals,))
        self.assertLess(robertson_ode.n_rhs_evals, 5000)
        self.assertAlmostEqual(np.sum(yout), 1.0, places=10)
        self.assertTrue(np.allclose(
            yout,
            np.array([2.08e-05, 8.33e-11, 9.99979176e-01]),
            rtol=5e-3,
            atol=1e-12))

    @pytest.mark.xfail(
        strict=True,
        reason="Known pathological case: transposing the Robertson Jacobian triggers CVODE work exhaustion on the current stack")
    def test_robertson_sundials_transposed_fails_with_excess_work(self):
        robertson_reset_n_evals()
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(
            scipy_to_cvode_rhs(robertson_rhs), 0, ROBERTSON_Y0.copy())

        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_spils_jac_times_vec_fn(
            scipy_to_cvode_jtimes(lambda t, y: robertson_jacobean(t, y).T))
        integrator.set_scalar_tolerances(1e-8, 1e-8)
        integrator.set_max_num_steps(5000)
        yout = np.zeros(3)
        integrator.advance_time(1e8, yout)
        # This asserts the "best case" contract; the xfail records that the
        # transposed Jacobian currently does not satisfy it. [Codex GPT-5.4]
        self.assertAlmostEqual(np.sum(yout), 1.0, places=10)
        self.assertLess(robertson_ode.n_rhs_evals, 5000)
        self.assertTrue(np.allclose(
            yout,
            np.array([2.08e-05, 8.33e-11, 9.99979176e-01]),
            rtol=5e-3,
            atol=1e-12))

if __name__ == '__main__':
    unittest.main()
