import finmag
import os
import numpy as np
from finmag.util.fileio import Tablereader

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
reference_file = os.path.join(MODULE_DIR, "barmini_test.ndt.ref")


def test_write_ndt_file(tmpdir):
    os.chdir(str(tmpdir))

    sim = finmag.example.barmini(name="barmini_test")
    for time in np.linspace(0, 1e-10, 21):
        sim.advance_time(time)
        sim.save_averages()
    print("Done.")

    # We used to do a file comparison here, but we had to fall back on
    # a numerical comparison since the integration times returned from
    # Sundials are slightly different for each run (which might be
    # worth investigating, since it means that our simulations runs
    # are not 100% reproducible)
    f_out = Tablereader("barmini_test.ndt")
    f_ref = Tablereader(reference_file)
    a_out = np.array(f_out['time', 'm_x', 'm_y', 'm_z'])
    a_ref = np.array(f_ref['time', 'm_x', 'm_y', 'm_z'])

    diff = np.abs(a_out - a_ref)
    print "Maximum difference: {}.".format(np.max(diff))
    assert np.allclose(a_out, a_ref, atol=5e-6, rtol=1e-8)
