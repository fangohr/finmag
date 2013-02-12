import pytest
import numpy as np
import dolfin as df
from finmag.energies import DMI,Exchange
from finmag import Simulation
from finmag.util.helpers import vector_valued_function

def test_dmi_pbc2d():
    mesh = df.BoxMesh(0,0,0,1,1,0.1,5, 5, 1)
     

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    expr = df.Expression(("0", "0", "1"))
    
    m = df.interpolate(expr, S3)
    
    dmi = DMI(1,pbc2d=True)
    dmi.setup(S3, m, 1)
    field=dmi.compute_field()
    assert np.max(field)<1e-15


def test_dmi_pbc2d_1D(plot=False):

    def m_init_fun(p):
        if p[0]<10:
            return [0.5,0,1]
        else:
            return [-0.5,0,-1]

    mesh = df.RectangleMesh(0,0,20,2,10,1)
    m_init = vector_valued_function(m_init_fun, mesh)

    Ms = 8.6e5
    sim = Simulation(mesh, Ms, pbc2d=True,unit_length=1e-9)
    sim.set_m(m_init)

    A = 1.3e-11
    D = 4e-3
    sim.add(Exchange(A,pbc2d=True))
    sim.add(DMI(D,pbc2d=True))
    
    sim.relax()

    if plot:
        df.plot(sim.llg._m)
        df.interactive()
    
    mx=[sim.llg._m(x+0.5,1)[0] for x in range(20)]
    assert np.max(np.abs(mx))<6e-7


if __name__ == "__main__":
   
   test_dmi_pbc2d_1D(plot=True)
    
    

