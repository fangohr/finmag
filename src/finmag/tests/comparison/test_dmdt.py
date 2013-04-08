import pytest
import subprocess
import dolfin as df
import numpy as np
from finmag.sim.llg import LLG
from finmag.energies import Zeeman
from finmag.util.oommf import mesh, oommf_dmdt
from finmag.util.helpers import stats

TOLERANCE = 3e-16

L  = 20e-9; W = 10e-9; H = 1e-9;
nL = 20;   nW = 10;   nH = 1;
msh = df.BoxMesh(0, 0, 0, L, W, H, nL, nW, nH)
S1 = df.FunctionSpace(msh, "Lagrange", 1)
S3 = df.VectorFunctionSpace(msh, "Lagrange", 1)

@pytest.mark.skipif('subprocess.call(["which", "oommf"]) != 0')
def test_dmdt_computation_with_oommf():
    # set up finmag
    llg = LLG(S1, S3)
    llg.set_m((-3, -2, 1))

    Ms=llg.Ms.vector().array()[0]
    Ms=float(Ms)
    h = Ms/2
    H_app = (h/np.sqrt(3), h/np.sqrt(3), h/np.sqrt(3))
    zeeman = Zeeman(H_app)
    zeeman.setup(S3, llg._m, llg.Ms, 1)
    llg.effective_field.add(zeeman)

    dmdt_finmag = df.Function(llg.S3)
    dmdt_finmag.vector()[:] = llg.solve(0)

    # set up oommf
    msh = mesh.Mesh((nL, nW, nH), size=(L, W, H))
    m0 = msh.new_field(3)
    m0.flat[0] += -3
    m0.flat[1] += -2
    m0.flat[2] += 1
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

    dmdt_oommf = oommf_dmdt(m0, Ms, A=0, H=H_app, alpha=llg.alpha, gamma_G=llg.gamma).flat

    # extract finmag data for comparison with oommf
    dmdt_finmag_like_oommf = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        dmdt_x, dmdt_y, dmdt_z = dmdt_finmag(x, y, z)
        dmdt_finmag_like_oommf.flat[0,i] = dmdt_x
        dmdt_finmag_like_oommf.flat[1,i] = dmdt_y
        dmdt_finmag_like_oommf.flat[2,i] = dmdt_z

    # compare
    difference = np.abs(dmdt_finmag_like_oommf.flat - dmdt_oommf)
    relative_difference = difference / np.max(np.sqrt(dmdt_oommf[0]**2+
        dmdt_oommf[1]**2+dmdt_oommf[2]**2))
    print "comparison with oommf, dm/dt, relative difference:"
    print stats(relative_difference)
    assert np.max(relative_difference) < TOLERANCE

    return difference, relative_difference

if __name__ == '__main__':
    test_dmdt_computation_with_oommf()
