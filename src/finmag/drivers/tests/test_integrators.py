# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import unittest
import pytest  # D33: added for the not_ported/xfail markers below (master had no pytest import)
try:  # master: from finmag.tests.jacobean.domain_wall_cobalt import setup_domain_wall_cobalt, domain_wall_error
    from finmag.tests.jacobean.domain_wall_cobalt import setup_domain_wall_cobalt, domain_wall_error
except ImportError:
    # not ported (D33): tests below xfail
    setup_domain_wall_cobalt = None
    domain_wall_error = None
from finmag.drivers.llg_integrator import llg_integrator
from datetime import datetime

NODE_COUNT = 100
END_TIME = 1e-10


class IntegratorTests(unittest.TestCase):

    def run_test(self, backend, method, nsteps=40000):
        # This stays as a broad smoke comparison across backends rather than a tight numerical benchmark. [Codex GPT-5.4]
        llg = setup_domain_wall_cobalt(node_count=NODE_COUNT)
        integrator = llg_integrator(
            llg, llg.m_field, backend, method=method, nsteps=nsteps)
        t = datetime.now()
        integrator.advance_time(END_TIME)
        dt = datetime.now() - t
        print("backend=%s, method=%s: elapsed time=%s, n_rhs_evals=%s, error=%g" % (
            backend,
            method,
            dt,
            integrator.n_rhs_evals,
            domain_wall_error(llg.m_field.as_array(), NODE_COUNT)))

    @pytest.mark.not_ported
    @pytest.mark.xfail(reason="not ported: legacy dolfin-API integrator-backend matrix, full legacy integrator matrix is not mapped (D33)", strict=True)
    def test_scipy_bdf(self):
        self.run_test("scipy", "bdf")

    @pytest.mark.not_ported
    @pytest.mark.xfail(reason="not ported: legacy dolfin-API integrator-backend matrix, full legacy integrator matrix is not mapped (D33)", strict=True)
    def test_scipy_adams(self):
        self.run_test("scipy", "adams")

    @pytest.mark.not_ported
    @pytest.mark.xfail(reason="not ported: legacy dolfin-API integrator-backend matrix, full legacy integrator matrix is not mapped (D33)", strict=True)
    def test_sundials_adams(self):
        self.run_test("sundials", "bdf_diag")

    @pytest.mark.not_ported
    @pytest.mark.xfail(reason="not ported: legacy dolfin-API integrator-backend matrix, full legacy integrator matrix is not mapped (D33)", strict=True)
    def test_sundials_bdf_diag(self):
        self.run_test("sundials", "adams")

    @pytest.mark.not_ported
    @pytest.mark.xfail(reason="not ported: legacy dolfin-API integrator-backend matrix, full legacy integrator matrix is not mapped (D33)", strict=True)
    def test_sundials_bdf_gmres_no_prec(self):
        self.run_test("sundials", "bdf_gmres_no_prec")

    @pytest.mark.not_ported
    @pytest.mark.xfail(reason="not ported: legacy dolfin-API integrator-backend matrix, full legacy integrator matrix is not mapped (D33)", strict=True)
    def test_sundials_bdf_gmres_prec_id(self):
        self.run_test("sundials", "bdf_gmres_prec_id")
