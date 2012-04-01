import os
import dolfin as df
import numpy as np
import magpar
from finmag.sim.anisotropy import UniaxialAnisotropy

from finmag.sim.helpers import quiver, boxplot, stats


#df.parameters["allow_extrapolation"] = True


REL_TOLERANCE = 2e-1 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"



def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE



def three_dimensional_problem():
    x_max = 10e-9; y_max = 1e-9; z_max = 1e-9;
    mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 2, 2)

    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    K = 520e3 # For Co (J/m3)
    Ms=45e4

    a = (0,0,1) # Easy axis in z-direction

    m0_x = "pow(sin(0.2*x[0]*1e9), 2)"
    m0_y = "0"
    m0_z = "pow(cos(0.2*x[0]*1e9), 2)"
    m=magpar.set_inital_m0(V,(m0_x,m0_y, m0_z))

    u_anis = UniaxialAnisotropy(V, m, K, df.Constant(a), Ms)
    finmag_anis = u_anis.compute_field()
    nodes, magpar_anis = magpar.compute_anis_magpar(V, m, K, a, Ms)
    
    #Because magpar have changed the order of the nodes!!!
    
    tmp=df.Function(V)
    tmp_c = mesh.coordinates()
    mesh.coordinates()[:]=tmp_c*1e9
    
    tmp.vector()[:]=finmag_anis
    
    nodes_tmp=[(i[0],i[1],i[2]) for i in nodes]

    finmag_anis_ordered=[tmp(i) for i in nodes_tmp]
    finmag_anis_ordered=np.array(finmag_anis_ordered)
    finmag_anis_ordered=finmag_anis_ordered.reshape(1,-1)[0]
    

    difference = np.abs(finmag_anis_ordered - magpar_anis)
    relative_difference = difference / np.sqrt(
        magpar_anis[0]**2 + magpar_anis[1]**2 + magpar_anis[2]**2)

    return dict( m0=m.vector().array(),
                 mesh=mesh,
                 anis=finmag_anis_ordered,
                 magpar_anis=magpar_anis,
                 diff=difference, 
                 rel_diff=relative_difference)


if __name__ == '__main__':

    res = three_dimensional_problem()
   
    print "finmag:",res["anis"]
    print "magpar:",res["magpar_anis"]
    print "rel_diff:",res["rel_diff"]
    print "max rel_diff",max(res["rel_diff"])
     
    prefix = MODULE_DIR + "_anis_"
    quiver(res["m0"], res["mesh"], prefix+"m0.png")
    quiver(res["anis"], res["mesh"], prefix+"anis.png")
    quiver(res["diff"], res["mesh"], prefix+"diff.png")
    boxplot(res["diff"], prefix+"rel_diff_box.png")
        
