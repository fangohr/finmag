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


@pytest.mark.slow
def test_time_default_max_steps():
    sim = finmag.example.barmini()
    t = 20e-9;
    with pytest.raises(RuntimeError) as exc_info:
        sim.advance_time(t)
    assert sim.t < t
