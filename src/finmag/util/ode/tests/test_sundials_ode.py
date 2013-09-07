# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import unittest
import math
import numpy as np
import finmag.native.sundials as sundials
from finmag.util.helpers import ignored

class OdeSundialsTests(unittest.TestCase):
    def test_errors(self):
        integrator = sundials.cvode(sundials.CV_ADAMS, sundials.CV_FUNCTIONAL)
        y = np.zeros((5,))
        try:
            integrator.advance_time(1, y)
            self.fail("Exception was not raised")
            pass
        except RuntimeError, ex:
            print ex

    def test_simple_1d_scipy(self):
        import scipy.integrate
        integrator = scipy.integrate.ode(lambda t, y: 0.5 * y)
        integrator.set_integrator('vode', rtol=1e-8, atol=1e-8)
        integrator.set_initial_value(np.array([1.]), 0)
        reference = lambda t: [math.exp(0.5*t)]
        ts = np.linspace(0.001, 3, 100)
        ys = np.array([integrator.integrate(t) for t in ts])
        ref_ys = np.array([reference(t)  for t in ts])
        assert np.max(np.abs(ys - ref_ys)) < 1e-6

    def test_simple_1d(self):
        integrator = sundials.cvode(sundials.CV_ADAMS, sundials.CV_FUNCTIONAL)
        self.init_simple_test(integrator)
        self.run_simple_test(integrator)
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_FUNCTIONAL)
        self.init_simple_test(integrator)
        self.run_simple_test(integrator)

    def test_simple_1d_diag(self):
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        self.init_simple_test(integrator)
        integrator.set_linear_solver_diag()
        self.run_simple_test(integrator)

    def init_simple_test(self, integrator):
        def rhs(t, y, ydot):
            ydot[:] = 0.5 * y
            return 0

        integrator.init(rhs, 0, np.array([1.]))
        integrator.set_scalar_tolerances(1e-9, 1e-9)

    def run_simple_test(self, integrator):
        reference = lambda t: [math.exp(0.5 * t)]
        yout = np.zeros(1)
        ts = np.linspace(0.001, 3, 100)
        ys = np.zeros((100, 1))
        for i, t in enumerate(ts):
            integrator.advance_time(t, yout)
            ys[i] = yout.copy()
        ref_ys = np.array([reference(t)  for t in ts])
        assert np.max(np.abs(ys - ref_ys)) < 1e-6

    def test_stiff_sp_gmr(self):
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        self.init_simple_test(integrator)
        def jtimes(v, Jv, t, y, fy, tmp):
            return 0
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_spils_jac_times_vec_fn(jtimes)
        self.run_simple_test(integrator)

    def test_jtimes_ex(self):
        class MyException(Exception):
            pass

        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        self.init_simple_test(integrator)
        def jtimes(v, Jv, t, y, fy, tmp):
            raise MyException()
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_spils_jac_times_vec_fn(jtimes)
        yout = np.zeros(1)
        with ignored(MyException):
            integrator.advance_time(1, yout)
            self.fail("Exception was not raised")
            pass

if __name__ == '__main__':
    unittest.main()
