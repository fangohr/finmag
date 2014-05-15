import finmag
import os


def test_integrator_get_set_max_steps(tmpdir):
    """
    Tests setting and getting of nsteps

    """
    os.chdir(str(tmpdir))
    sim = finmag.example.barmini()
    sim.run_until(0) # create integrator object
    steps = sim.integrator.max_steps
    assert steps != 42  # would be an odd default value
    sim.integrator.max_steps = 42
    steps2 = sim.integrator.max_steps
    assert steps2 == 42
    sim.integrator.max_steps = steps
    assert steps == sim.integrator.max_steps


def test_integrator_stats(tmpdir):
    """
    Tests the stats

    """
    os.chdir(str(tmpdir))
    sim = finmag.example.barmini()
    sim.run_until(0)  # create integrator object
    stats = sim.integrator.stats()
    # All stats should be zero before we do any work
    for key in stats:
        assert stats[key] == 0.0


def test_integrator_n_steps_only(tmpdir):
    """
    Test integration for a few steps only

    """
    os.chdir(str(tmpdir))
    sim = finmag.example.barmini()
    sim.create_integrator()
    assert sim.integrator.stats()['nsteps'] == 0
    sim.integrator.advance_steps(1)
    assert sim.integrator.stats()['nsteps'] == 1
    # check also value of cur_t is up-to-date
    assert sim.integrator.cur_t == sim.integrator.stats()['tcur']

    # expect also last time step size to be the same as current time
    # because we have only done one step
    assert sim.integrator.stats()['tcur'] == sim.integrator.stats()['hlast']

    sim.integrator.advance_steps(1)
    assert sim.integrator.stats()['nsteps'] == 2
    sim.integrator.advance_steps(2)
    assert sim.integrator.stats()['nsteps'] == 4
    # check also value of cur_t is up-to-date
    assert sim.integrator.cur_t == sim.integrator.stats()['tcur']
