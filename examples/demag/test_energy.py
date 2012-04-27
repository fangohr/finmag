from numpy import pi, sqrt
from dolfin import UnitSphere
from finmag.sim.llg import LLG

TOL = 1.5e-2

# Using unit sphere mesh
mesh = UnitSphere(10)
mesh_units = 1
llg = LLG(mesh, mesh_units=mesh_units)
llg.set_m((1, 0, 0))
llg.Ms = 1e5
llg.setup(use_demag=True)

def test_energy():
    """
    Test the demag energy.

    Read the corresponding documentation for explanation.

    """
    E_demag = llg.demag.compute_energy()
    print "Demag energy:", E_demag

    vol = 4*pi/3
    mu0 = 4*pi*10**-7
    E_exact = 1./6*mu0*llg.Ms**2*vol
    print "Exact solution:", E_exact

    diff = abs(E_demag - E_exact)
    rel_error =  diff/sqrt(E_exact**2 + E_demag**2)
    print "Relative error:", rel_error
    assert rel_error < TOL, "Relative error is %g, should be zero" % rel_error

if __name__ == '__main__':
    test_energy()
