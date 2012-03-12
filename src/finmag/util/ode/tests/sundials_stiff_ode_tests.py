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

n_rhs_evals = 0
n_jac_evals = 0

def robertson_rhs_sundials(t, y, yout):
    global n_rhs_evals
    n_rhs_evals += 1
    yout[0] = -0.04 * y[0] + 1e4 * y[1] * y[2]
    yout[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]
    yout[2] = 3e7 * y[1] * y[1]
    return 0

def robertson_jacobean_scipy(t, y):
    global n_jac_evals
    n_jac_evals += 1
    jac = np.zeros((3,3))
    # jac[i,j] = d f[i] / d y[j]
    jac[0,0] = -0.04
    jac[0,1] = 1e4*y[2]
    jac[0,2] = 1e4*y[1]
    jac[1,0] = 0.04
    jac[1,1] = -1e4*y[2] - 6e7*y[1]
    jac[1,2] = -1e4*y[1]
    jac[2,0] = 0
    jac[2,1] = 6e7*y[1]
    jac[2,2] = 0
    return jac

def robertson_rhs_scipy(t, y):
    yout = np.zeros(3)
    robertson_rhs_sundials(t, y, yout)
    return yout

ROBERTSON_Y0 = np.array([1., 0., 0.])

class SundialsStiffOdeTests(unittest.TestCase):
    def test_robertson_scipy(self):
        import scipy.integrate
        y_tmp = ROBERTSON_Y0.copy()
        global n_rhs_evals
        n_rhs_evals = 0
        integrator = scipy.integrate.ode(robertson_rhs_scipy, jac=robertson_jacobean_scipy)
        integrator.set_initial_value(ROBERTSON_Y0)
        integrator.set_integrator("vode", method="bdf", nsteps=10000)
        integrator.integrate(1e8)
        print "Integration of the Robertson ODE until t=1e8 with scipy VODE, BDF method: %d steps" % (n_rhs_evals,)

if __name__ == '__main__':
    unittest.main()
