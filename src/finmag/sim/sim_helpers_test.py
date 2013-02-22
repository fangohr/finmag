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
    t0 = 10e-12
    t1 = 20e-12

    # Create simulation, integrate until t0, save restart data,
    # and integrate until t1.
    sim1 = barmini()
    sim1.run_until(t0)
    print("Stats for sim1 at t = {} s:\n{}.".format(sim1.t, sim1.integrator.stats()))
    sim_helpers.save_restart_data(sim1)
    sim1.run_until(t1)
    print("Stats for sim1 at t = {} s:\n{}.".format(sim1.t, sim1.integrator.stats()))

    # Bring new simulation object into previously saved
    # state (which is at t0)...
    data = sim_helpers.load_restart_data(sim1)
    sim2 = barmini()
    sim2.set_m(data['m'])
    sim2.integrator = llg_integrator(sim2.llg, sim2.llg.m,
                                     backend=sim2.integrator_backend, t0=data['simtime'])
    # ... and integrate until t1.
    sim2.run_until(t1)
    print("Stats for sim2 at t = {} s:\n{}.".format(sim2.t, sim2.integrator.stats()))

    # Check that we have the same data in both simulation objects.
    print "Time for sim1: {} s, time for sim2: {} s.".format(sim1.t, sim2.t)
    assert abs(sim1.t - sim2.t) < 1e-16
    print "Average magnetisation for sim1:\n\t{}\nfor sim2:\n\t{}.".format(
                        sim1.m_average, sim2.m_average)
    assert np.allclose(sim1.m, sim2.m, atol=5e-6, rtol=1e-8)

    # Check that sim2 had less work to do, since it got the
    # results up to t0 for free.
    stats1 = sim1.integrator.stats()
    stats2 = sim2.integrator.stats()
    assert stats2['nsteps'] < stats1['nsteps']


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
