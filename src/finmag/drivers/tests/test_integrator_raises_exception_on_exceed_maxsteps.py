import os
import pytest
import finmag

steps = 10

def test_can_change_maxsteps():
    sim = finmag.example.barmini()
    sim.create_integrator()
    sim.integrator.max_steps = steps
    assert sim.integrator.max_steps == steps


def test_time_run_until():
    sim = finmag.example.barmini()
    sim.create_integrator()
    sim.integrator.max_steps = steps
    t = 1e-9;
    with pytest.raises(RuntimeError) as exc_info:
        sim.run_until(t)
    assert sim.t < t


def test_time_advance_time():
    sim = finmag.example.barmini()
    sim.create_integrator()
    sim.integrator.max_steps = steps
    t = 1e-9
    with pytest.raises(RuntimeError) as exc_info:
        sim.advance_time(t)
    assert sim.t < t  # check that integration was aborted


def test_time_default_max_steps():
    """The idea for this test was to check the default max_steps, but 
    the simulation for this runs about 12 minutes. So I have changed 
    the code below, to stop after 10 steps. HF, Sept 2014
    """
    sim = finmag.example.barmini()
    t = 20e-9;
    # create integrator
    sim.create_integrator()
    # set steps to a small number
    sim.integrator.max_steps = 10

    # now run until we exceed 10 steps
    with pytest.raises(RuntimeError) as exc_info:
        sim.advance_time(t)
    assert sim.t < t
