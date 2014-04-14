import os
import dolfin as df
import numpy as np
from finmag import Simulation as Sim
from finmag.energies import Zeeman

alpha = 0.01
Ms = 8.6e5

def test_time_run_until():
    mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    sim = Sim(mesh, Ms, unit_length=1e-9)
    sim.alpha = alpha
    sim.set_m((1, 0, 0))
    sim.set_tol(1e-10, 1e-14)
    sim.add(Zeeman((0, 0, 1e5)))
    
    t = 2e-8; 
    sim.run_until(t)
    print "Asked to run until t = {} s, internal time is now t = {} s.".format(t, sim.t)
    assert abs(sim.t - t) < 1e-13

def test_time_advance_time():
    mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    sim = Sim(mesh, Ms, unit_length=1e-9)
    sim.alpha = alpha
    sim.set_m((1, 0, 0))
    sim.set_tol(1e-10, 1e-14)
    sim.add(Zeeman((0, 0, 1e5)))

    t = 2e-8
    sim.advance_time(t)
    print "Asked to run until t = {} s, internal time is now t = {} s.".format(t, sim.t)
    assert abs(sim.t - t) < 1e-13
