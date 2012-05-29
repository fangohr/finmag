import os
import dolfin as df
from numpy import pi, sqrt
from finmag.energies import Demag

TOL = 1.5e-2
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
Ms = 1e5

def test_energy():

    """
    Test the demag energy.

    Read the corresponding documentation for explanation.

    """

    # The dolfin UnitSphere gives a coarse and low quality mesh.
    # Use this instead when the lindholm formulation is implemented.
    # Then we can also set TOL = 1.5e-3
    #mesh = df.Mesh(convert_mesh(MODULE_DIR + "/sphere_fine.geo"))

    # Using unit sphere mesh
    mesh = df.UnitSphere(10)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.interpolate(df.Constant((1, 0, 0)), S3)

    demag = Demag()
    demag.setup(S3, m, Ms, unit_length=1)

    E_demag = demag.demag.compute_energy()
    print "Demag energy:", E_demag
    print "Numerical solution on the netgen mesh: 8758.92651323"

    vol = 4*pi/3
    mu0 = 4*pi*10**-7
    E_exact = 1./6*mu0*Ms**2*vol
    print "Exact solution:", E_exact

    diff = abs(E_demag - E_exact)
    rel_error =  diff/sqrt(E_exact**2 + E_demag**2)
    print "Relative error:", rel_error
    assert rel_error < TOL, "Relative error is %g, should be zero" % rel_error

if __name__ == '__main__':
    test_energy()
