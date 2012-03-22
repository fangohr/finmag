import dolfin as df
import numpy as np

from finmag.sim.llg import LLG
from finmag.util.oommf import mesh, oommf_dmdt

TOLERANCE = 5e-16

L  = 20e-9; W = 10e-9; H = 1e-9;
nL = 20;   nW = 10;   nH = 1;

def test_dmdt_computation_with_oommf():
    # set up finmag 
    llg = LLG(df.Box(0, 0, 0, L, W, H, nL, nW, nH))
    llg.set_m0((-3, -2, 1))

    h = llg.Ms/2
    H_app = (h/np.sqrt(3), h/np.sqrt(3), h/np.sqrt(3))
    llg.H_app = H_app

    llg.setup(exchange_flag=False)

    dmdt_finmag = df.Function(llg.V)
    dmdt_finmag.vector()[:] = llg.solve()

    # set up oommf
    msh = mesh.Mesh((nL, nW, nH), size=(L, W, H))
    m0 = msh.new_field(3)
    m0.flat[0] += -3
    m0.flat[1] += -2
    m0.flat[2] += 1
    m0.flat /= np.sqrt(m0.flat[0]*m0.flat[0] + m0.flat[1]*m0.flat[1] + m0.flat[2]*m0.flat[2])

    dmdt_oommf = oommf_dmdt(m0, llg.Ms, A=llg.C, H=H_app, alpha=llg.alpha, gamma_G=llg.gamma).flat

    # extract finmag data for comparison with oommf
    dmdt_finmag_like_oommf = msh.new_field(3)
    for i, (x, y, z) in enumerate(msh.iter_coords()):
        dmdt_x, dmdt_y, dmdt_z = dmdt_finmag(x, y, z)
        dmdt_finmag_like_oommf.flat[0,i] = dmdt_x
        dmdt_finmag_like_oommf.flat[1,i] = dmdt_y
        dmdt_finmag_like_oommf.flat[2,i] = dmdt_z

    # compare
    difference = dmdt_finmag_like_oommf.flat - dmdt_oommf
    relative_difference = np.abs(difference / dmdt_oommf)
    assert np.max(relative_difference) < TOLERANCE

    return difference, relative_difference

if __name__ == '__main__':
    difference, relative_difference = test_dmdt_computation_with_oommf()
    print "difference"
    print difference
    print "absolute relative difference"
    print relative_difference
    print "absolute relative difference:"
    print "  median", np.median(relative_difference, axis=1)
    print "  average", np.mean(relative_difference, axis=1)
    print "  minimum", np.min(relative_difference, axis=1)
    print "  maximum", np.max(relative_difference, axis=1)
    print "  spread", np.std(relative_difference, axis=1)


