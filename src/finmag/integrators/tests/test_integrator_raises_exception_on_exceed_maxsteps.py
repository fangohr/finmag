import os
import dolfin as df
import numpy as np
import finmag
from finmag import Simulation as Sim
from finmag.energies import Zeeman

import pytest

alpha = 0.01
Ms = 8.6e5


def test_maxsteps_default_is_10000():
    sim = finmag.example.barmini()
    # attempt time integration to create integrator
    sim.advance_time(1e-18)
    assert sim.integrator.get_max_steps() == 10000

def test_time_run_until():
    mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    sim = Sim(mesh, Ms, unit_length=1e-9)
    sim.alpha = alpha
    sim.set_m((1, 0, 0))
    sim.set_tol(1e-10, 1e-14)
    sim.add(Zeeman((0, 0, 1e5)))
    
    t = 2e-8;
    with pytest.raises(RuntimeError) as exc_info:
        sim.run_until(t)
    # Learn something about the exception:
    # print("Exception is")
    # print(exc_info)
    # print(dir(exc_info))
    #
    # for attr in dir(exc_info):
    #     print("{} : {}".format(attr, getattr(exc_info, attr)))
    # # -> so I learn that the actual exception is in '.value'
    # for attr in dir(exc_info.value):
    #      print("{} : {}".format(attr, getattr(exc_info.value, attr)))
    # # -> and the message is in value.message:

    # Check that the first line of the value of the exception is what we expect:
    # (This test will need updating whenever we change the message.)
    assert exc_info.value.message.split('\n')[0] \
       == "The integrator has reached its maximum of 10000 steps."

def test_time_advance_time():
    mesh = df.BoxMesh(0, 0, 0, 2, 2, 2, 1, 1, 1)
    sim = Sim(mesh, Ms, unit_length=1e-9)
    sim.alpha = alpha
    sim.set_m((1, 0, 0))
    sim.set_tol(1e-10, 1e-14)
    sim.add(Zeeman((0, 0, 1e5)))

    t = 2e-8
    with pytest.raises(RuntimeError) as exc_info:
        sim.advance_time(t)
    assert exc_info.value.message.split('\n')[0] \
       == "The integrator has reached its maximum of 10000 steps."
        
    print "Asked to run until t = {} s, internal time is now t = {} s.".format(t, sim.t)
    # this will fail because we have exceeded the number of iterations:
    #assert abs(sim.t - t) < 1e-13

if __name__ == "__main__":
    #pytest.main(["-k test_time_run_until"])
    #pytest.main(["-k test_time_advance_time"])
    pass
