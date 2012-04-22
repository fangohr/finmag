import numpy as np
import dolfin as df
import unittest
import math
from finmag.sim.llg import LLG
from domain_wall_cobalt import setup_domain_wall_cobalt, domain_wall_error
from finmag.native import llg as native_llg
from jacobean_computation_tests import setup_llg_params_near_one
from finmag.util.time_counter import counter

class NativeLlgTests(unittest.TestCase):
    def test_compute_llg(self):
        llg_instant, m = setup_llg_params_near_one(node_count=30, use_instant=True)
        llg_instant.pins = [0, 2, 3]
        llg_native, m2 = setup_llg_params_near_one(node_count=30, use_instant=False)
        llg_native.pins = llg_instant.pins
        self.assertEquals(np.max(np.abs(m-m2)), 0)
        y_instant = llg_instant.solve_for(m, 0)
        y_native = llg_native.solve_for(m, 0)
        self.assertLess(np.max(np.abs(y_instant - y_native)), 1e-12)
        # Check that we are actually using different implementations
        self.assertGreater(np.max(np.abs(y_instant - y_native)), 0)

    def test_llg_performance(self):
        # Unfortunately 100000 nodes is not enough to even fill the L3 cache
        # TODO: Increase the number of nodes when this is fast enough
        llg, m = setup_llg_params_near_one(node_count=200000, use_instant=True)
        # Calculate H_eff etc
        llg.solve_for(m, 0)
        # Profile the LLG computation using pyinstant
        c = counter()
        while c.next():
            llg._solve(llg.gamma, llg.c, llg.alpha_vec, llg.m, llg.H_eff, llg.m.shape[0], llg.pins, llg.do_precession)
        print "Computing dm/dt via pyinstant", c
        c = counter()
        H_eff = llg.H_eff.reshape((3, -1))
        m.shape = (3, -1)
        dmdt = np.zeros(m.shape)
        while c.next():
            native_llg.calc_llg_dmdt(m, H_eff, llg.t, dmdt, llg.gamma / (1. + llg.alpha ** 2), llg.alpha, 0.1/llg.c, llg.do_precession)
        print "Computing dm/dt via native C++ code", c

if __name__=="__main__":
    unittest.main()
