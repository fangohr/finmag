import os
import run_validation_sim as vsim
import numpy as np

def test_validation():
    nmag_file = os.path.join(vsim.MODULE_DIR, "averages_nmag5.txt")
    epsilon = 1e-16

    #short run:
    tmax = 0.05e-9; tolerance = 2e-6

    #long run
    #tmax = 10e-9; tolerance = 1e-4

    n = vsim.run_simulation(t_max=tmax)

    averages = np.loadtxt(vsim.averages_file)
    nmag_avg = np.loadtxt(nmag_file)[:n, :]

    diff = np.abs(np.array(averages) - np.array(nmag_avg))
    print("Deviation is %s" % (np.max(diff)))
    assert np.max(diff[:, 0]) < epsilon # compare times
    assert np.max(diff[:, 1:]) < tolerance


# Note: On osiris, the time integration in this example seems to get
# progressively slower. Is that right? HF, 7 Dec 2012
