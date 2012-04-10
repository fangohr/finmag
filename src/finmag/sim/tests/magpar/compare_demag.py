import os
import dolfin as df
import numpy as np
import magpar
from finmag.demag.demag_solver import Demag
from finmag.demag.solver_fk_test import SimpleFKSolver

from finmag.sim.helpers import quiver, boxplot, stats


#df.parameters["allow_extrapolation"] = True


REL_TOLERANCE = 2e-1 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"



def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def three_dimensional_problem():
    x_max = 10e-9; y_max = 1e-9; z_max = 1e-9;
    mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 2, 2, 2)
    mesh = df.UnitSphere(10)

    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    Ms=8.6e5

    m0_x = "pow(sin(0.2*x[0]*1e9), 2)"
    m0_y = "0.2"
    m0_z = "1"

    m=magpar.set_inital_m0(V,(1,0,0))
    #mesh.coordinates()[:]=mesh.coordinates()*1e9                                                                                                                                    
    print 'the number of nodes in the mesh: ',mesh.num_cells()
    print 'the theoretical demag field: (%e %e %e) '%(-Ms/3,0,0)
    demag_tmp = Demag(V, m, Ms,method="GCR")
    finmag_demag = demag_tmp.compute_field()
    mesh.coordinates()[:]=mesh.coordinates()*1e-9
    nodes, magpar_demag = magpar.compute_demag_magpar(V, m, Ms)

    # Magpar have changed the order of the nodes just because the mesh extracted from dolfin can not work directly, but how to sort them ???                                         

    tmp=df.Function(V)
    tmp_c = mesh.coordinates()
    mesh.coordinates()[:]=tmp_c*1e9

    finmag_demag,magpar_demag, \
        diff,rel_diff=magpar.compare_field_directly( \
            mesh.coordinates(),finmag_demag,\
            nodes, magpar_demag)

    return dict( m0=m.vector().array(),
                 mesh=mesh,
                 demag=finmag_demag,
                 magpar_demag=magpar_demag,
                 diff=diff,
                 rel_diff=rel_diff)

def three_dimensional_problem2():
    #x_max = 10e-9; y_max = 1e-9; z_max = 1e-9;
    #mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 2, 2, 2)
    #mesh = df.UnitSphere(10)
    from finmag.demag.problems.prob_fembem_testcases import MagSphere
    mesh = MagSphere(10,2.).mesh
    print mesh


    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    Ms=8.6e5

    m0_x = "pow(sin(0.2*x[0]*1e9), 2)"
    m0_y = "0.2"
    m0_z = "1"

    m=magpar.set_inital_m0(V,(1,0,0))
    #mesh.coordinates()[:]=mesh.coordinates()*1e9
    print 'the number of nodes in the mesh: ',mesh.num_cells()
    print 'the theoretical demag field: (%e %e %e) '%(-Ms/3,0,0)
    demag_tmp = SimpleFKSolver(V, m, Ms)
    finmag_demag = demag_tmp.compute_field()
    mesh.coordinates()[:]=mesh.coordinates()*1e-9
    nodes, magpar_demag = magpar.compute_demag_magpar(V, m, Ms)
    
    # Magpar have changed the order of the nodes just because the mesh extracted from dolfin can not work directly, but how to sort them ???
    
    tmp=df.Function(V)
    tmp_c = mesh.coordinates()
    mesh.coordinates()[:]=tmp_c*1e9

    finmag_demag,magpar_demag, \
        diff,rel_diff=magpar.compare_field_directly( \
            mesh.coordinates(),finmag_demag,\
            nodes, magpar_demag)
    
    return dict( m0=m.vector().array(),
                 mesh=mesh,
                 demag=finmag_demag,
                 magpar_demag=magpar_demag,
                 diff=diff, 
                 rel_diff=rel_diff)


if __name__ == '__main__':

    res = three_dimensional_problem()
    
    print "finmag:",res["demag"]
    print "magpar:",res["magpar_demag"]
    print "rel_diff:",res["rel_diff"]
    print "max rel_diff",np.max(res["rel_diff"])


    res2 = three_dimensional_problem2()
    print "testttttt fk "*10

    print "finmag_fk_test:",res2["demag"]
    print "magpar:",res2["magpar_demag"]
    print "rel_diff:",res2["rel_diff"]
    print "max rel_diff",np.max(res2["rel_diff"])
    print 'mean rel_diff',np.mean(res2["rel_diff"])
