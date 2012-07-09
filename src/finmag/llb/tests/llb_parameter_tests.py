import unittest
import numpy as np
from finmag.native.llb import LLBFePt

class LLBParameterTests(unittest.TestCase):
    def test_simple(self):
        material = LLBFePt()
        T = np.array([100., 300., 600., 1000.])
        # if T is an vector of length n,
        # compute_parameters will return 4 x n array
        # m_e, A, inv_chi_perp, inv_chi_par
        parameters = material.compute_parameters(T)
        print "T:", T
        print "Computed LLB material parameters:", parameters