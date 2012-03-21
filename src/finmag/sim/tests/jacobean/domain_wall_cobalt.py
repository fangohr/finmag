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
Ms_Co = 1400e3 # A/m
K1_Co = 520e3 # A/m
A_Co = 30e-12 # J/m

LENGTH = 100e-9
NODE_COUNT = 200

# Initial m
def initial_m(xi, node_count):
    mz = 1. - 2. * xi / (node_count - 1)
    my = math.sqrt(1 - mz * mz)
    return [0, my, mz]

# Analytical solution for the relaxed mz
def reference_mz(x):
    return math.cos(math.pi / 2 + math.atan(math.sinh((x - LENGTH / 2) / math.sqrt(A_Co / K1_Co))))

def setup_domain_wall_cobalt(node_count=NODE_COUNT, A=A_Co, Ms=Ms_Co, K1=K1_Co, length=LENGTH, use_instant=True, do_precession=True):
    mesh = df.Interval(node_count - 1, 0, length)
    llg = LLG(mesh, use_instant_llg=use_instant, do_precession=do_precession)
    llg.C = 2 * A
    llg.Ms = Ms
    llg.add_uniaxial_anisotropy(K1, df.Constant((0, 0, 1)))
    llg.set_m0(np.array([initial_m(xi, node_count) for xi in xrange(node_count)]).T.reshape((-1,)))
    llg.setup()
    llg.pins = [0, node_count - 1]
    return llg

def domain_wall_error(ys, node_count):
    m = ys.view()
    m.shape = (3, -1)
    return np.max(np.abs(m[2] - [reference_mz(x) for x in np.linspace(0, LENGTH, node_count)]))

def compute_domain_wall_cobalt(end_time=1e-9):
    llg = setup_domain_wall_cobalt()
    integrator = scipy.integrate.ode(lambda t, y: llg.solve_for(y, t))
    integrator.set_integrator("vode", method="bdf", atol=1e-8, rtol=1e-8, nsteps=120000)
    integrator.set_initial_value(llg.m)
    ys = integrator.integrate(end_time)
    return np.linspace(0, LENGTH, NODE_COUNT), ys.reshape((3, -1))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xs, m = compute_domain_wall_cobalt()
    print "max difference between simulation and reference: ", domain_wall_error(m, NODE_COUNT)
    xs = np.linspace(0, LENGTH, NODE_COUNT)
    plt.plot(xs, np.transpose([m[2], [reference_mz(x) for x in xs]]), label=['Simulation', 'Reference'])
    plt.xlabel('x [m]')
    plt.ylabel('m')
    plt.title('Domain wall in Co')
    plt.show()


