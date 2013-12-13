import numpy as np
import conftest
from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf
from finmag.util.oommf import mesh, oommf_cubic_anisotropy
from finmag.util.helpers import stats

def test_against_oommf(finmag=conftest.setup_cubic()):

    REL_TOLERANCE = 7e-2

    oommf_mesh = mesh.Mesh((20, 20, 20), size=(conftest.x1, conftest.y1, conftest.z1))
    #FIXME: why our result is three times of oommf's??
    oommf_anis  = 3*oommf_cubic_anisotropy(m0=oommf_m0(conftest.m_gen, oommf_mesh),
            Ms=conftest.Ms, K1=conftest.K1, K2=conftest.K2, u1=conftest.u1, u2=conftest.u2).flat
    finmag_anis = finmag_to_oommf(finmag["H"], oommf_mesh, dims=3)

    assert oommf_anis.shape == finmag_anis.shape
    diff = np.abs(oommf_anis - finmag_anis)
    print diff
    rel_diff = diff / np.sqrt((np.max(oommf_anis[0]**2 + oommf_anis[1]**2 + oommf_anis[2]**2)))

    print "comparison with oommf, H, relative_difference:"
    print stats(rel_diff)

    finmag["table"] += conftest.table_entry("oommf", REL_TOLERANCE, rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    finmag = conftest.setup_cubic()
    test_against_oommf(finmag)
    conftest.teardown(finmag)
