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
        integrator = sundials.sundials_cvode(sundials.CV_ADAMS, sundials.CV_FUNCTIONAL)
        y = np.zeros((5,))
        try:
            integrator.advance_time(1, y)
            self.fail("Exception was not raised")
            pass
        except RuntimeError, ex:
            print ex

    def atest_simple_1d(self):
        integrator.set_integrator('vode', rtol=1e-8, atol=1e-8)
        integrator.set_initial_value(np.array([1.]), 0)
        reference = lambda t: [math.exp(0.5*t)]
        ts = np.linspace(0.001, 3, 100)
        ys = np.array([integrator.integrate(t) for t in ts])
        ref_ys = np.array([reference(t)  for t in ts])
        assert np.max(np.abs(ys - ref_ys)) < 1e-6

if __name__ == '__main__':
    unittest.main()
