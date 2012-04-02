import os
import dolfin as df
import numpy as np
import magpar
from finmag.sim.exchange import Exchange

from finmag.sim.helpers import quiver, boxplot, stats


#df.parameters["allow_extrapolation"] = True


REL_TOLERANCE = 2.2e-6 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"



def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE



def three_dimensional_problem():
    x_max = 10e-9; y_max = 1e-9; z_max = 1e-9;
    mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 2, 2)

    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
   
    Ms=8.6e5

    m0_x = "pow(sin(0.2*x[0]*1e9), 2)"
    m0_y = "0"
    m0_z = "pow(cos(0.2*x[0]*1e9), 2)"
    m=magpar.set_inital_m0(V,(m0_x,m0_y, m0_z))

    C=1.3e-11

    u_exch = Exchange(V, m, C, Ms)
    finmag_exch = u_exch.compute_field()
    nodes, magpar_exch = magpar.compute_exch_magpar(V, m, C, Ms)
     
    
    #Because magpar have changed the order of the nodes!!!
    
    tmp=df.Function(V)
    tmp_c = mesh.coordinates()
    mesh.coordinates()[:]=tmp_c*1e9
    
    tmp.vector()[:]=finmag_exch
    
    nodes_tmp=[(i[0],i[1],i[2]) for i in nodes]

    tmp2=[tmp(i) for i in nodes_tmp]
    finmag_exch_ordered=np.array(tmp2)
    finmag_exch_ordered=finmag_exch_ordered.reshape(1,-1)[0]
    

    difference = np.abs(finmag_exch_ordered - magpar_exch)
    relative_difference = difference / np.sqrt(
        magpar_exch[0]**2 + magpar_exch[1]**2 + magpar_exch[2]**2)

    return dict( m0=m.vector().array(),
                 mesh=mesh,
                 exch=finmag_exch_ordered,
                 magpar_exch=magpar_exch,
                 diff=difference, 
                 rel_diff=relative_difference)


if __name__ == '__main__':

    res = three_dimensional_problem()
   
    print "finmag:",res["exch"]
    print "magpar:",res["magpar_exch"]
    print "rel_diff:",res["rel_diff"]
    print "max rel_diff",max(res["rel_diff"])
     
    prefix = MODULE_DIR + "_exch_"
    quiver(res["m0"], res["mesh"], prefix+"m0.png")
    quiver(res["exch"], res["mesh"], prefix+"exch.png")
    quiver(res["diff"], res["mesh"], prefix+"diff.png")
    boxplot(res["diff"], prefix+"rel_diff_box.png")
        
