import os
import dolfin as df
import numpy as np
import magpar
from finmag.sim.anisotropy import UniaxialAnisotropy

from finmag.sim.helpers import normed_func

#df.parameters["allow_extrapolation"] = True

#REL_TOLERANCE = 1.5e-5 #passes with 'normal magpar'
REL_TOLERANCE = 7e-7 #needs higher accuracy patch 
                     #for saved files to pass
                     #install magpar via finmag/install/magpar.sh to get this.

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"


def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE


def three_dimensional_problem():
    x_max = 10; y_max = 1; z_max = 1;
    mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 2, 2)

    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    K = 520e3 # For Co (J/m3)
    Ms=45e4

    a = (0,0,1) # Easy axis in z-direction

    m=normed_func((1, 1, 1), V)

    u_anis = UniaxialAnisotropy(V, m, K, df.Constant(a), Ms)
    finmag_anis = u_anis.compute_field()
    nodes, magpar_anis = magpar.compute_anis_magpar(m, K1=K, a=a, Ms=Ms)
    
    finmag_anis,magpar_anis, \
        diff,rel_diff=magpar.compare_field_directly( \
            mesh.coordinates(),finmag_anis,\
            nodes, magpar_anis)

    return dict( m0=m.vector().array(),
                 mesh=mesh,
                 anis=finmag_anis,
                 magpar_anis=magpar_anis,
                 diff=diff, 
                 rel_diff=rel_diff)


if __name__ == '__main__':

    res = three_dimensional_problem()
   
    print "finmag:",res["anis"]
    print "magpar:",res["magpar_anis"]
    print "rel_diff:",res["rel_diff"]
    print "max rel_diff",np.max(res["rel_diff"])

    """
    prefix = MODULE_DIR + "_anis_"
    quiver(res["m0"], res["mesh"], prefix+"m0.png")
    quiver(res["anis"], res["mesh"], prefix+"anis.png")
    quiver(res["diff"], res["mesh"], prefix+"diff.png")
    boxplot(res["diff"], prefix+"rel_diff_box.png")
    """
