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

    output_file = "barmini_test.ndt"
    reference_file = "barmini_test.ndt.ref"

    sim = finmag.example.barmini(name="barmini_test")
    for time in linspace(0, 1e-10, 20):
        print("Integrating towards t = %gs" % time)
        sim.run_until(time, save_averages=True)  # True is the default for save_averages
                                                 # but we provide it for clarity.
    print("Done")
    result = filecmp.cmp(output_file, reference_file, shallow=False)

    if result == False:
        import difflib

        lines1 = open(output_file).readlines()
        lines2 = open(reference_file).readlines()

        print("Output differs from reference file:")
        print "".join(difflib.unified_diff(lines1, lines2))

    assert(result)
