import pytest
import numpy as np
import dolfin as df
from distutils.version import StrictVersion
from finmag.energies import DMI,Exchange
from finmag import Simulation
from finmag.util.helpers import vector_valued_function
from finmag.util.pbc2d import PeriodicBoundary2D


def test_dmi_pbc2d():
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 0.1, 2, 2, 1)

    pbc = PeriodicBoundary2D(mesh)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)
    expr = df.Expression(("0", "0", "1"))

    m = df.interpolate(expr, S3)

    dmi = DMI(1)
    dmi.setup(S3, m, 1)
    field = dmi.compute_field()
    
    assert np.max(field) < 1e-15



def test_dmi_pbc2d_1D(plot=False):

    def m_init_fun(p):
        if p[0]<10:
            return [0.5,0,1]
        else:
            return [-0.5,0,-1]

    mesh = df.RectangleMesh(0,0,20,2,10,1)
    m_init = vector_valued_function(m_init_fun, mesh)

    Ms = 8.6e5
    sim = Simulation(mesh, Ms, pbc='2d',unit_length=1e-9)
    sim.set_m(m_init_fun)

    A = 1.3e-11
    D = 4e-3
    sim.add(Exchange(A))
    sim.add(DMI(D))

    #have no idea why we need so small stopping_dmdt
    sim.relax(stopping_dmdt=0.0005)

    if plot:
        df.plot(sim.llg._m)
        df.interactive()

    mx=[sim.llg._m(x+0.5,1)[0] for x in range(20)]

    assert np.max(np.abs(mx)) < 1e-6


if __name__ == "__main__":
    #test_dmi_pbc2d()
    test_dmi_pbc2d_1D(plot=True)
