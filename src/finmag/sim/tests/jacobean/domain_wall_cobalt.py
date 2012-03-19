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
import scipy.integrate

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

def compute_domain_wall_cobalt(end_time=1e-9):
    mesh = df.Interval(NODE_COUNT - 1, 0, LENGTH)

    llg = LLG(mesh)
    llg.C = 2*A
    llg.Ms = Ms
    llg.add_uniaxial_anisotropy(K1, df.Constant((0, 0, 1)))
    llg.set_m0(np.array([initial_m(xi) for xi in xrange(NODE_COUNT)]).T.reshape((-1,)))
    llg.setup()
    llg.pins = [0, NODE_COUNT-1]
    ys, infodict = scipy.integrate.odeint(llg.solve_for, llg.m, [0, end_time], full_output=True)
    print "Used", infodict["nfe"][-1], "function evaluations."
    return ys[1].reshape((3, -1))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mx, my, mz = compute_domain_wall_cobalt()
    xs = np.linspace(0, LENGTH, NODE_COUNT)
    plt.plot(xs, mx, 'r', label='$m_x$')
    plt.plot(xs, my, 'g', label='$m_y$')
    plt.plot(xs, mz, 'b', label='$m_z$')
    plt.xlabel('x [m]')
    plt.ylabel('m')
    plt.title('Domain wall in Co')
    plt.grid()
    plt.legend()
    plt.show()


