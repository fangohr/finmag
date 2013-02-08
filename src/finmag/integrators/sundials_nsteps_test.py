import finmag
import pytest

def test_integrator_get_set_max_steps():
    """Tests setting and getting of nsteps"""
    sim = finmag.example.barmini()
    sim.run_until(0) # create integrator object
    steps = sim.integrator.get_max_steps()
    assert steps != 42  # would be an odd default value
    sim.integrator.set_max_steps(42)
    steps2 = sim.integrator.get_max_steps()
    assert steps2 == 42
    sim.integrator.set_max_steps(steps)
    assert steps == sim.integrator.get_max_steps()


def test_integrator_stats():
    """Tests the stats"""
    sim = finmag.example.barmini()
    sim.run_until(0)  # create integrator object
    stats = sim.integrator.stats()
    # All stats should be zero before we do any work
    for key in stats:
        assert stats[key] == 0.0


def test_integrator_n_steps_only():
    """Test integration for a few steps only"""
    sim = finmag.example.barmini()
    sim.run_until(0)  # create integrator object
    assert sim.integrator.stats()['nsteps'] == 0
    sim.integrator.set_max_steps(1)
    ret_val = sim.integrator.advance_time(1e-12)
    assert ret_val == False
    assert sim.integrator.stats()['nsteps'] == 1
    # check also value of cur_t is up-to-date
    assert sim.integrator.cur_t == sim.integrator.stats()['tcur']

    # expect also last time step size to be the same as current time
    # because we have only done one step
    assert sim.integrator.stats()['tcur'] == sim.integrator.stats()['hlast']

    sim.integrator.set_max_steps(1)
    ret_val = sim.integrator.advance_time(1e-12)
    assert sim.integrator.stats()['nsteps'] == 2
    assert ret_val == False
    sim.integrator.set_max_steps(2)
    ret_val = sim.integrator.advance_time(1e-12)
    assert sim.integrator.stats()['nsteps'] == 4
    assert ret_val == False
    # check also value of cur_t is up-to-date
    assert sim.integrator.cur_t == sim.integrator.stats()['tcur']


def test_integrator_run_until_return_value():
    sim = finmag.example.barmini()
    sim.run_until(0) # to create integrator object
    assert sim.integrator.stats()['nsteps'] == 0
    sim.integrator.set_max_steps(1)
    ret_val = sim.integrator.advance_time(1e-15)
    assert sim.integrator.stats()['nsteps'] == 1
    assert ret_val == False
    # check also value of cur_t is up-to-date
    assert sim.integrator.cur_t == sim.integrator.stats()['tcur']

    # if integration succeeds, we should get True back
    sim.integrator.set_max_steps(500) # default value 
    ret_val = sim.integrator.advance_time(1e-15)
    print("For information: nsteps = {nsteps}".format(**sim.integrator.stats()))  # about 6 steps
    assert ret_val == True

    # check also value of cur_t is up-to-date (which here means
    # carries the desired value of 1e-15 as the integration
    # succeeded)
    assert sim.integrator.cur_t == 1e-15
