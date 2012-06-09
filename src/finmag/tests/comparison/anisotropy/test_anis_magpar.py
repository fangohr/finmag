import numpy as np
import conftest as test
import finmag.util.magpar as magpar
from finmag.sim.helpers import stats

def test_against_magpar(finmag):

    REL_TOLERANCE = 5e-7

    magpar_nodes, magpar_anis = magpar.compute_anis_magpar(finmag["m"],
            K1=test.K1, a=test.a, Ms=test.Ms)
    _, _, diff, rel_diff = magpar.compare_field_directly(
            finmag["S3"].mesh().coordinates(), finmag["H"].vector().array(),
            magpar_nodes, magpar_anis)

    print "comparison with magpar, H, relative_difference:"
    print stats(rel_diff)

    finmag["table"] += test.table_entry("magpar", REL_TOLERANCE, rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    finmag = test.setup()
    test_against_magpar(finmag)
    test.teardown(finmag)
