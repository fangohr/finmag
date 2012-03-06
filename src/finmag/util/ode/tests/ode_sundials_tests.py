# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

from finmag.util.ode import cvode
import unittest
import math
import numpy as np
import finmag.native.sundials as sundials

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
        self.run_simple_test(integrator)
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_FUNCTIONAL)
        self.run_simple_test(integrator)

    def run_simple_test(self, integrator):
        def rhs(t, y, ydot):
            ydot[:] = 0.5 * y
            return 0

        integrator.init(rhs, 0, np.array([1.]))
        integrator.set_scalar_tolerances(1e-9, 1e-9)
        reference = lambda t: [math.exp(0.5 * t)]
        yout = np.zeros(1)
        ts = np.linspace(0.001, 3, 100)
        ys = np.zeros((100, 1))
        for i, t in enumerate(ts):
            integrator.advance_time(t, yout)
            ys[i] = yout.copy()
        ref_ys = np.array([reference(t)  for t in ts])
        assert np.max(np.abs(ys - ref_ys)) < 1e-6

if __name__ == '__main__':
    unittest.main()
