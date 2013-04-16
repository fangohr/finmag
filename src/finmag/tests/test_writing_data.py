import finmag
import os
import numpy as np
from finmag.util.fileio import Tablereader

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(MODULE_DIR, "barmini_test.ndt")
reference_file = os.path.join(MODULE_DIR, "barmini_test.ndt.ref")


def test_write_ndt_file():
    cwd_backup = os.getcwd()
    # Change into directory of the test so that the .ndt file is saved
    # there and the output file is found.
    os.chdir(MODULE_DIR)

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
    f_out = Tablereader(output_file)
    f_ref = Tablereader(reference_file)
    a_out = f_out['time', 'm_x', 'm_y', 'm_z']
    a_ref = f_ref['time', 'm_x', 'm_y', 'm_z']

    diff = np.abs(a_out - a_ref)
    print "Maximum difference: {}.".format(np.max(diff))
    assert np.allclose(a_out, a_ref, atol=5e-6, rtol=1e-8)

    os.chdir(cwd_backup)
