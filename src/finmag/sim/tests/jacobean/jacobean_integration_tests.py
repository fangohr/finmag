# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import numpy as np
import scipy.integrate
import dolfin as df
import unittest
import math
from finmag.sim.llg import LLG

# Material parameters
Ms = 1400e3 # A/m
K1 = 520e3 # A/m
A = 30e-12 # J/m

LENGTH = 100e-9
NODE_COUNT = 200

# Initial m
def initial_m(xi):
    mz = 1. - 2. * xi / (NODE_COUNT - 1)
    my = math.sqrt(1 - mz * mz)
    return [0, my, mz]

class JacobeanIntegrationTests(unittest.TestCase):
    def test_scipy(self):
        mesh = df.Interval(NODE_COUNT-1, 0, LENGTH)

        llg = LLG(mesh)
        llg.set_m0(np.array([initial_m(xi) for xi in xrange(NODE_COUNT)]).T.reshape((-1,)))
        llg.setup()
        llg.pins = [0, 10]

if __name__ == '__main__':
    unittest.main()
