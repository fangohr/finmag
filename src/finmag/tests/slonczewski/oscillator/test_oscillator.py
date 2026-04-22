import os
import numpy as np
import pytest

epsilon = 1e-16
tolerance = 1e-3


@pytest.mark.skip(reason="long Slonczewski oscillator validation; not part of the Python 3 core gate yet")
def test_oscillator():
    import run as sim

    nmag_file = os.path.join(sim.MODULE_DIR, "averages_nmag5.txt")
    if not os.path.exists(sim.initial_m_file):
        sim.create_initial_state()
    sim.run_simulation()

    averages = np.loadtxt(sim.averages_file)
    nmag_avg = np.loadtxt(nmag_file)

    diff = np.abs(np.array(averages) - np.array(nmag_avg))
    assert np.max(diff[:, 0]) < epsilon  # compare times
    assert np.max(diff[:, 1:]) < tolerance
