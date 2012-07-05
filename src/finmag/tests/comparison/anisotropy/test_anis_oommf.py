import numpy as np
import conftest as test
from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf
from finmag.util.oommf import mesh, oommf_uniaxial_anisotropy
from finmag.util.helpers import stats

def test_against_oommf(finmag):

    REL_TOLERANCE = 7e-2

    oommf_mesh = mesh.Mesh((20, 20, 20), size=(test.x1, test.y1, test.z1))
    oommf_anis  = oommf_uniaxial_anisotropy(oommf_m0(test.m_gen, oommf_mesh),
            test.Ms, test.K1, test.a).flat
    finmag_anis = finmag_to_oommf(finmag["H"], oommf_mesh, dims=3)

    assert oommf_anis.shape == finmag_anis.shape
    diff = np.abs(oommf_anis - finmag_anis)
    rel_diff = diff / np.sqrt((np.max(oommf_anis[0]**2 + oommf_anis[1]**2 + oommf_anis[2]**2)))

    print "comparison with oommf, H, relative_difference:"
    print stats(rel_diff)

    finmag["table"] += test.table_entry("oommf", REL_TOLERANCE, rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    finmag = test.setup()
    test_against_oommf(finmag)
    test.teardown(finmag)
