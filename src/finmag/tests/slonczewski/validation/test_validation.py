import os
import run as sim
import numpy as np

epsilon = 1e-16
tolerance = 1e-4
nmag_file = os.path.join(sim.MODULE_DIR, "averages_nmag5.txt")

def test_oscillator():
    sim.run_simulation()

    averages = np.loadtxt(sim.averages_file)
    nmag_avg = np.loadtxt(nmag_file)
   
    diff = np.abs(np.array(averages) - np.array(nmag_avg))
    assert np.max(diff[:,0]) < epsilon # compare times
    assert np.max(diff[:,1:]) < tolerance
