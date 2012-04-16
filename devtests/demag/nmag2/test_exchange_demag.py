import os
import pytest
import numpy as np
import finmag.sim.helpers as h

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 3e-2

@pytest.mark.skipif("not os.path.exists(MODULE_DIR+ '/averages.txt')")
def test_compare_averages():
    ref = np.array(h.read_float_data(MODULE_DIR + "/averages_ref.txt"))
    computed = np.array(h.read_float_data(MODULE_DIR + "/averages.txt"))

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / np.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2))

    print "test_averages, max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)

    err = np.nanmax(rel_diff)
    if err > 1e-3:
        print "nmag:\n", ref
        print "finmag:\n", computed
    assert np.nanmax(rel_diff) < REL_TOLERANCE
