import datetime
import numpy as np
import finmag
import restart

def test_can_read_restart_file():
    sim = finmag.example.barmini()
    sim.run_until(1e-21)  # create integrator
    restart.save_restart_data(sim)
    data = restart.load_restart_data(sim)

    assert data['simname'] == sim.name
    assert data['simtime'] == sim.t
    assert data['stats'] == sim.integrator.stats()
    assert np.all(data['m'] == sim.integrator.llg.m)
    #writing and reading the data should take less than 10 seconds
    assert datetime.datetime.now() - data['datetime'] < datetime.timedelta(0, 10)

def test_try_to_restart_a_simulation():
    sim1 = finmag.example.barmini()
    sim1.run_until(10e-12)  # create integrator
    restart.save_restart_data(sim1)
    # continue up to 2e-12
    sim1.run_until(20e-12)
    print("Stats for sim1 at t=2e-12: %s" % sim1.integrator.stats())
    # now create a new simulation object
    sim2 = finmag.example.barmini()
    # and try to bring it back into the restart state.
    # To be done.

    data = restart.load_restart_data(sim2)
    print("Have reloaded data: %s" % data)
    sim2.llg._m.vector()[:] = data['m']
    from finmag.integrators.llg_integrator import llg_integrator
    sim2.integrator = llg_integrator(sim2.llg, sim2.llg.m, 
        backend=sim2.integrator_backend, t0=data['simtime'])

    # at this point, we should have re-instantiated sim2 to the same state (nearly)
    # that sim1 was when cur_t was 1e-12.
    print "Need to call run_until to update stats (not yet done):"
    print sim2.integrator.stats()
    sim2.run_until(sim2.t + 1e-20)
    print("Have called run_until, stats now are")
    print sim2.integrator.stats()

    # complete time integration
    sim2.run_until(20e-12)
    print sim2.integrator.stats()

    print sim2.m_average, sim2.t
    print sim1.m_average, sim1.t
    print("Max deviation: = %g" % (max(sim2.m - sim1.m)))
    assert max(sim2.m - sim1.m) < 3e-08
    assert sim1.t == sim2.t

    # for the second 10e-12 seconds, we needed much fewer steps. Check:
    assert sim2.integrator.stats()['nsteps'] * 4 < sim1.integrator.stats()['nsteps']



if __name__ == '__main__':
    test_try_to_restart_a_simulation()
