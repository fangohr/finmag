import finmag
import os
import numpy as np

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
    a_out = np.loadtxt(output_file)
    a_ref = np.loadtxt(reference_file)

    diff = np.abs(a_out - a_ref)
    print "Maximum difference: {}.".format(np.max(diff))
    assert np.allclose(a_out, a_ref, atol=5e-6, rtol=1e-8)

    os.chdir(cwd_backup)
