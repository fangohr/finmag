# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import unittest
from finmag.tests.jacobean.domain_wall_cobalt import setup_domain_wall_cobalt, domain_wall_error
from finmag.drivers.llg_integrator import llg_integrator
from datetime import datetime

NODE_COUNT = 100
END_TIME = 1e-10

class IntegratorTests(unittest.TestCase):
    def run_test(self, backend, method, nsteps=40000):
        llg = setup_domain_wall_cobalt(node_count=NODE_COUNT)
        integrator = llg_integrator(llg, llg.m, backend, method=method, nsteps=nsteps)
        t = datetime.now()
        integrator.advance_time(END_TIME)
        dt = datetime.now() - t
        print "backend=%s, method=%s: elapsed time=%s, n_rhs_evals=%s, error=%g" % (
                backend,
                method,
                dt,
                integrator.n_rhs_evals,
                domain_wall_error(integrator.m, NODE_COUNT))

    def test_scipy_bdf(self):
        self.run_test("scipy", "bdf")

    def test_scipy_adams(self):
        self.run_test("scipy", "adams")

    def test_sundials_adams(self):
        self.run_test("sundials", "bdf_diag")

    def test_sundials_bdf_diag(self):
        self.run_test("sundials", "adams")

    def test_sundials_bdf_gmres_no_prec(self):
        self.run_test("sundials", "bdf_gmres_no_prec")

    def test_sundials_bdf_gmres_prec_id(self):
        self.run_test("sundials", "bdf_gmres_prec_id")
