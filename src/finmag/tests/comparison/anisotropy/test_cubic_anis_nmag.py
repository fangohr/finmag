import numpy as np
import conftest
import os
from finmag.util.helpers import vectors, stats

def test_cubic_against_nmag(finmag=conftest.setup_cubic()):

    REL_TOLERANCE = 1e-6

    m_ref = np.genfromtxt(os.path.join(conftest.MODULE_DIR, "m0_nmag.txt"))
    m_computed = vectors(finmag["m"].vector().array())
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(os.path.join(conftest.MODULE_DIR, "H_cubic_anis_nmag.txt"))
    H_computed = vectors(finmag["H"].vector().array())
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    mxH_ref = np.cross(m_ref, H_ref)
    mxH_computed = np.cross(m_computed, H_computed)
    print mxH_ref
    print mxH_computed

    diff = np.abs(mxH_computed - mxH_ref)
    rel_diff = diff/ np.sqrt(np.max(mxH_ref[0]**2 + mxH_ref[1]**2 + mxH_ref[2]**2))

    print "comparison with nmag, m x H, difference:"
    print stats(diff)
    print "comparison with nmag, m x H, relative difference:"
    print stats(rel_diff)

    finmag["table"] += conftest.table_entry("nmag", REL_TOLERANCE, rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    finmag = conftest.setup_cubic()
    test_cubic_against_nmag(finmag)
    conftest.teardown(finmag)
