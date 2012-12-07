import finmag
import filecmp
import os
from numpy import linspace

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_write_ndt_file():
    # Change into the directory of the test so that the .ndt file is saved there
    # and the reference file is found.
    # TODO: This might not be the best solution, should revise this at some point.
    os.chdir(MODULE_DIR)

    sim = finmag.example.barmini(name="barmini_test")
    for time in linspace(0, 1e-10, 20):
        print("Integrating towards t = %gs" % time)
        sim.run_until(time, save_averages=True)  # True is the default for save_averages
                                                 # but we provide it for clarity.
    print("Done")
    assert(filecmp.cmp("barmini_test.ndt", "barmini_test.ndt.ref", shallow=False))
