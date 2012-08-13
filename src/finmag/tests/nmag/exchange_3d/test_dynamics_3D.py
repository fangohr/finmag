import os
import cProfile
import pstats
import finmag.util.helpers as h

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 8e-3

def setup_module(module):
    import run_dolfin as s
    s.run_simulation()

def test_compare_averages():
    ref = h.read_float_data(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = h.read_float_data(os.path.join(MODULE_DIR, "averages.txt"))

    for i in range(len(computed)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = computed[i]

        assert abs(mx - mx_ref) < TOLERANCE
        assert abs(my - my_ref) < TOLERANCE
        assert abs(mz - mz_ref) < TOLERANCE

if __name__ == "__main__":
    def do_it():
        import run_dolfin as s
        s.run_simulation()
        test_compare_averages()
    cProfile.run("do_it()", "test_profile")
    p = pstats.Stats("test_profile")
    print "TOP10 Cumulative time:"
    p.sort_stats("cumulative").print_stats(10)
    print "TOP10 Time inside a function:"
    p.sort_stats("time").print_stats(10)

