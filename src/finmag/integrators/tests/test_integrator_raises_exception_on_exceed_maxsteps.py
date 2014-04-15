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
    # Check that the first line of the value of the exception is what we expect:
    # (This test will need updating whenever we change the message.)
    assert exc_info.value.message.split('\n')[0] \
       == "The integrator has reached its maximum of {} steps.".format(steps)
    assert sim.t < t


def test_time_advance_time():
    sim = finmag.example.barmini()
    sim.create_integrator()
    sim.integrator.max_steps = steps
    t = 1e-9
    with pytest.raises(RuntimeError) as exc_info:
        sim.advance_time(t)
    assert exc_info.value.message.split('\n')[0] \
       == "The integrator has reached its maximum of {} steps.".format(steps)
    assert sim.t < t  # check that integration was aborted


@pytest.mark.slow
def test_time_default_max_steps():
    sim = finmag.example.barmini()
    t = 20e-9;
    with pytest.raises(RuntimeError) as exc_info:
        sim.advance_time(t)
    # Check that the first line of the value of the exception is what we expect:
    # (This test will need updating whenever we change the message.)
    assert exc_info.value.message.split('\n')[0] \
       == "The integrator has reached its maximum of {} steps.".format(
               sim.integrator.max_steps)
    assert sim.t < t
