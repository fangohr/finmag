import numpy as np
import conftest as test
from finmag.sim.helpers import vectors, stats

REL_TOLERANCE = 7e-2

def test_against_nmag(finmag):
    m_ref = np.genfromtxt(test.MODULE_DIR + "m0_nmag.txt")
    m_computed = vectors(finmag["m"].vector().array())
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(test.MODULE_DIR + "H_anis_nmag.txt")
    H_computed = vectors(finmag["H"].vector().array())
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    mxH_ref = np.cross(m_ref, H_ref)
    mxH_computed = np.cross(m_computed, H_computed)

    diff = np.abs(mxH_computed - mxH_ref)
    rel_diff = diff/ np.sqrt(np.max(mxH_ref[0]**2 + mxH_ref[1]**2 + mxH_ref[2]**2))

    print "comparison with nmag, m x H, difference:"
    print stats(diff)
    print "comparison with nmag, m x H, relative difference:"
    print stats(rel_diff)

    finmag["table"] += test.table_entry("nmag", REL_TOLERANCE, rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    finmag = test.setup()
    test_against_nmag(finmag)
    test.teardown(finmag)
