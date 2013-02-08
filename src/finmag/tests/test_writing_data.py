import finmag
import filecmp
import os
import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_write_ndt_file():
    # Change into the directory of the test so that the .ndt file is saved there
    # and the reference file is found.
    # TODO: This might not be the best solution, should revise this at some point.
    os.chdir(MODULE_DIR)

    #FIXME:change back to 1e-12 later
    RTOL = 1e-10 
    
    

    output_file = "barmini_test.ndt"
    reference_file = "barmini_test.ndt.ref"

    sim = finmag.example.barmini(name="barmini_test")
    for time in np.linspace(0, 1e-10, 21):
        print("Integrating towards t = %gs" % time)
        sim.advance_time(time)
        sim.save_averages()
    print("Done")

    # We used to do a file comparison here, but we had to fall back on
    # a numerical comparison since the integration times returned from
    # Sundials are slightly different for each run (which might be
    # worth investigating, since it means that our simulations runs
    # are not 100% reproducible)
    a_out = np.loadtxt(output_file)
    a_ref = np.loadtxt(reference_file)
    result = np.allclose(a_out, a_ref, atol=0, rtol=RTOL)

    if result == False:
        import difflib
        lines1 = open(output_file).readlines()
        lines2 = open(reference_file).readlines()
        print("Output differs from reference file:")
        print "".join(difflib.unified_diff(lines1, lines2))

    assert(result)
