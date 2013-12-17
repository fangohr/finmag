import numpy as np
import conftest
from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf
from finmag.util.oommf import mesh, oommf_uniaxial_anisotropy
from finmag.util.helpers import stats

def test_against_oommf(finmag=conftest.setup(K2=0)):

    REL_TOLERANCE = 7e-2

    oommf_mesh = mesh.Mesh((20, 20, 20), size=(conftest.x1, conftest.y1, conftest.z1))
    oommf_anis  = oommf_uniaxial_anisotropy(oommf_m0(conftest.m_gen, oommf_mesh),
            conftest.Ms, conftest.K1, conftest.u1).flat
    finmag_anis = finmag_to_oommf(finmag["H"], oommf_mesh, dims=3)

    assert oommf_anis.shape == finmag_anis.shape
    diff = np.abs(oommf_anis - finmag_anis)
    rel_diff = diff / np.sqrt((np.max(oommf_anis[0]**2 + oommf_anis[1]**2 + oommf_anis[2]**2)))

    print "comparison with oommf, H, relative_difference:"
    print stats(rel_diff)

    finmag["table"] += conftest.table_entry("oommf", REL_TOLERANCE, rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    finmag = conftest.setup(K2=0)
    test_against_oommf(finmag)
    conftest.teardown(finmag)
