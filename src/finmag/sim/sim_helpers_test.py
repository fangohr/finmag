import os
import shutil
import numpy as np
import sim_helpers
from datetime import datetime, timedelta
from finmag.example import barmini
from finmag.integrators.llg_integrator import llg_integrator


def test_can_read_restart_file():
    sim = barmini()
    sim.run_until(1e-21)  # create integrator
    sim_helpers.save_restart_data(sim)
    data = sim_helpers.load_restart_data(sim)

    assert data['simname'] == sim.name
    assert data['simtime'] == sim.t
    assert data['stats'] == sim.integrator.stats()
    assert np.all(data['m'] == sim.integrator.llg.m)
    #writing and reading the data should take less than 10 seconds
    assert datetime.now() - data['datetime'] < timedelta(0, 10)


def test_try_to_restart_a_simulation():
    sim1 = barmini()
    sim1.run_until(10e-12)  # create integrator
    sim_helpers.save_restart_data(sim1)
    # continue up to 2e-12
    sim1.run_until(20e-12)
    print("Stats for sim1 at t=2e-12: %s" % sim1.integrator.stats())
    # now create a new simulation object
    sim2 = barmini()
    # and try to bring it back into the restart state.
    # To be done.

    data = sim_helpers.load_restart_data(sim2)
    print("Have reloaded data: %s" % data)
    sim2.llg._m.vector()[:] = data['m']
    sim2.integrator = llg_integrator(sim2.llg, sim2.llg.m,
        backend=sim2.integrator_backend, t0=data['simtime'])

    # At this point, we should have re-instantiated sim2 to the same
    # state (nearly) that sim1 was when cur_t was 1e-12.
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

    # For the second 10e-12 seconds, we needed much fewer steps. Check:
    stats1 = sim1.integrator.stats()
    stats2 = sim2.integrator.stats()
    assert stats2['nsteps'] * 4 < stats1['nsteps']


def test_create_backup_if_file_exists():
    # remove file
    testfilename = 'tmp-testfile.txt'
    if os.path.exists(testfilename):
        os.remove(testfilename)

    backupfilename = 'tmp-testfile.txt.backup'
    if os.path.exists(backupfilename):
        os.remove(backupfilename)

    assert os.path.exists(testfilename) == False
    assert os.path.exists(backupfilename) == False

    # create file
    os.system('echo "Hello World" > ' + testfilename)

    assert os.path.exists(testfilename) == True
    assert open(testfilename).readline()[:-1] == "Hello World"
    assert os.path.exists(backupfilename) == False

    sim_helpers.create_backup_file_if_file_exists(testfilename)
    assert os.path.exists(backupfilename) == True

    assert open(backupfilename).readline()[:-1] == "Hello World"

    assert open(backupfilename).read() == open(testfilename).read()


if __name__ == '__main__':
    test_try_to_restart_a_simulation()
