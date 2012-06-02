import os, sys
import dolfin as df
import numpy as np
from finmag import Simulation
from finmag.energies import Zeeman, TimeZeeman, Demag, Exchange
import run_init as problem

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
average_m_file = os.path.join(MODULE_DIR, "m_averages.txt")

"""
Micromag Standard Problem #4

specification:
    http://www.ctcms.nist.gov/~rdm/mumag.org.html

"""

if not os.path.exists(problem.initial_m_file):
    print "No starting magnetisation found in {}. Creating it.".format(problem.initial_m_file)
    problem.initialise_m()

print "Now running simulation."
sim = Simulation(problem.mesh, problem.Ms)
sim.set_m(np.loadtxt(problem.initial_m_file))
sim.alpha = problem.alpha
sim.gamma = problem.gamma
sim.add(Demag())
sim.add(Exchange(problem.A))

"""
Field 1, ~25mT in direction of 170 degrees.

mu0 * Hx  = -24.6 mT
          = -24.6 * 10^-3 Vs/m^2 divide by mu0
Hx = -24.6 / 4PI * 10^-3 * 10*7 Am/Vs * Vs/m^2
   = -24.6 / 4PI * 10^4 A/m
the same way
Hy = 4.3mT = 4.3 / 4PI * 10^4 A/m
and Hz = 0.

"""
Hx = -24.6e4 / (4 * np.pi)
Hy = 4.3e4 / (4 * np.pi)
Hz = 0
sim.add(Zeeman((Hx, Hy, Hz)))

f = open(MODULE_DIR + "/m_averages.txt", "w")
t = 0; t_max = 2e-9; dt = 1e-11;
while t <= t_max:
    sim.run_until(t_max)
    mx, my, mz = sim.m_average
    f.write("{} {} {} {}\n".format(t, mx, my, mz))
    t += dt
f.close()

print "Done. Average magnetisations saved to {}.".format(average_m_file)
