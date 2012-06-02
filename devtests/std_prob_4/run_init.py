import os
import dolfin as df
import numpy as np
from finmag import Simulation
from finmag.energies import TimeZeeman, Demag, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
initial_m_file = os.path.join(MODULE_DIR, "m_init.txt")

"""
Create a s-state for the starting magnetisation.

"""

L = 500e-9; W = 125e-9; H = 3e-9; # dimensions of film          m
Ms = 8.0e5                        # saturation magnetisation    A/m
A = 1.3e-11                       # exchange coupling strength  J/m
alpha = 0.02
gamma = 2.211e5                   # in m/(As)

mesh = df.Box(0, 0, 0, L, W, H, 166, 41, 1)

def initialise_m():
    # A saturating field in the [1, 1, 1] direction,
    # that is gradually reduced every 10 picoseconds
    # until it vanishes after one nanosecond.
    t_off = 1e-9; t_update = 1e-12;
    f_expr = df.Expression(tuple(3 * ["(t_off - t)*H"]), t=0, t_off=t_off, H=Ms)
    saturating_field = TimeZeeman(f_expr, np.arange(0, t_off, t_update))

    sim = Simulation(mesh, Ms)
    sim.alpha = 1
    sim.gamma = gamma
    sim.llg.do_precession = False
    sim.set_m((1, 1, 1))
    sim.add(Demag())
    sim.add(Exchange(A))
    sim.add(saturating_field, update_func=saturating_field.update)
    sim.run_until(t_off)
    sim.relax()

    np.savetxt("m_init.txt", sim.m)
    print "Saved magnetisation to {}.".format(initial_m_file)

if __name__ == "__main__":
    initialise_m()
