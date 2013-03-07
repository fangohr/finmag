# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import numpy as np
import scipy.integrate
import unittest
import py
from domain_wall_cobalt import setup_domain_wall_cobalt, domain_wall_error
from finmag.native import sundials
from finmag.util.ode import scipy_to_cvode_rhs
from datetime import datetime

NODE_COUNT = 100
END_TIME = 1e-10

@py.test.mark.slow
class JacobeanIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.n_rhs_evals = 0

    def scipy_rhs(self, t, y):
        self.n_rhs_evals += 1
        return self.llg.solve_for(y, t)

    def run_scipy_test(self, method):
        self.llg = setup_domain_wall_cobalt(node_count=NODE_COUNT)
        integrator = scipy.integrate.ode(self.scipy_rhs)
        integrator.set_integrator("vode", method=method, atol=1e-8, rtol=1e-8, nsteps=40000)
        integrator.set_initial_value(self.llg.m)
        t = datetime.now()
        ys = integrator.integrate(END_TIME)
        dt = datetime.now() - t
        print "scipy integration: method=%s, n_rhs_evals=%d, error=%g, elapsed time=%s" % (method, self.n_rhs_evals, domain_wall_error(ys, NODE_COUNT), dt)

    def test_scipy_bdf(self):
        self.run_scipy_test("bdf")

    def test_scipy_adams(self):
        self.run_scipy_test("adams")

    def run_sundials_test_no_jacobean(self, method):
        self.llg = setup_domain_wall_cobalt(node_count=NODE_COUNT)
        if method=="bdf":
            integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        else:
            assert method=="adams"
            integrator = sundials.cvode(sundials.CV_ADAMS, sundials.CV_FUNCTIONAL)
        integrator.init(scipy_to_cvode_rhs(self.scipy_rhs), 0, self.llg.m.copy())
        if method == "bdf":
            integrator.set_linear_solver_diag()
        integrator.set_scalar_tolerances(1e-8, 1e-8)
        integrator.set_max_num_steps(40000)
        ys = np.zeros(self.llg.m.shape)
        t = datetime.now()
        integrator.advance_time(END_TIME, ys)
        dt = datetime.now() - t
        print "sundials integration, no jacobean (%s, diagonal): n_rhs_evals=%d, error=%g, elapsed time=%s" % (method, self.n_rhs_evals, domain_wall_error(ys, NODE_COUNT), dt  )

    def test_sundials_diag_bdf(self):
        self.run_sundials_test_no_jacobean("bdf")

    def test_sundials_diag_adams(self):
        self.run_sundials_test_no_jacobean("adams")

    def test_sundials_test_with_jacobean(self):
        self.llg = setup_domain_wall_cobalt(node_count=NODE_COUNT)

        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(scipy_to_cvode_rhs(self.scipy_rhs), 0, self.llg.m.copy())
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_spils_jac_times_vec_fn(self.llg.sundials_jtimes)
        integrator.set_scalar_tolerances(1e-8, 1e-8)
        integrator.set_max_num_steps(40000)
        ys = np.zeros(self.llg.m.shape)
        t = datetime.now()
        integrator.advance_time(END_TIME, ys)
        dt = datetime.now() - t
        print "sundials integration, with jacobean: n_rhs_evals=%d, error=%g, elapsed time=%s" % (self.n_rhs_evals, domain_wall_error(ys, NODE_COUNT), dt)

if __name__ == '__main__':
    unittest.main()
