import os
import cProfile
import pstats
import pytest
import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 3e-4

def setup_module(module):
    import run_finmag as f
    f.run_simulation()

@pytest.mark.slow
def test_compare_averages():
    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = np.loadtxt(os.path.join(MODULE_DIR, "averages.txt"))

    highest_diff = 0
    for i in range(len(computed)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = computed[i]

        dx = abs(mx - mx_ref); dy = abs(my - my_ref); dz = abs(mz - mz_ref);
        d = max([dx, dy, dz])

        if d > highest_diff:
            highest_diff = d

        assert d < TOLERANCE
    print "Highest difference was {0}.".format(highest_diff)
        

if __name__ == "__main__":
    def do_it():
        import run_finmag as f
        f.run_simulation()
        test_compare_averages()
    cProfile.run("do_it()", "test_profile")
    p = pstats.Stats("test_profile")
    print "TOP10 Cumulative time:"
    p.sort_stats("cumulative").print_stats(10)
    print "TOP10 Time inside a function:"
    p.sort_stats("time").print_stats(10)
