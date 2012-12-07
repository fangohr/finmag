import finmag
import filecmp
from numpy import linspace

def test_write_ndt_file():
    sim = finmag.example.barmini(name="barmini_test.ndt")
    for time in linspace(0, 1e-10, 20):
        print("Integrating towards t = %gs" % time)
        sim.run_until(time)
        sim.save_averages()
    print("Done")
    assert(filecmp.cmp("barmini_test.ndt", "barmini_test.ndt.ref", shallow=False))
