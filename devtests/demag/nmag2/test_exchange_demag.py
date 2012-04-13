import os
import pytest
import finmag.sim.helpers as h

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 1e-2

@pytest.mark.skipif("not os.path.exists(MODULE_DIR+ '/averages.txt')")
def test_compare_averages():
    ref = h.read_float_data(MODULE_DIR + "/averages_ref.txt")
    computed = h.read_float_data(MODULE_DIR + "/averages.txt")

    #TODO: Remove this
    ERR = 0

    for i in range(len(computed)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = computed[i]
        
        assert abs(t  - t_ref)  < 1e-15

        assert abs(mx - mx_ref) < TOLERANCE
        assert abs(my - my_ref) < TOLERANCE
        assert abs(mz - mz_ref) < TOLERANCE

        ####
        # DEBUGGING, TODO: REMOVE THE REST OF THE FILE
        ####

        err = max(abs(mx-mx_ref), abs(my-my_ref), abs(mz-mz_ref))
        if err > ERR:
            ERR = err
    print ERR

if __name__ == '__main__':
    test_compare_averages()

