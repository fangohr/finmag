import os
import dolfin as df
from numpy import pi, sqrt
from finmag.energies import Demag
from finmag.util.meshes import from_geofile

TOL = 1.9e-2
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
    mesh = from_geofile(os.path.join(MODULE_DIR, "sphere_fine.geo"))

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.interpolate(df.Constant((1, 0, 0)), S3)

    vol = 4*pi/3
    mu0 = 4*pi*10**-7
    E_exact = 1./6*mu0*Ms**2*vol
    print "Exact solution:", E_exact
    print "Numerical solution on the netgen mesh: 8758.92651323\n"

    res = {"FK":{},"GCR":{}}
    for demagtype in ["FK","GCR"]:
        demag = Demag(demagtype)
        demag.setup(S3, m, Ms, unit_length=1)

        E_demag = demag.demag.compute_energy()
        print "\n%s Demag energy:"%demagtype, E_demag

        diff = abs(E_demag - E_exact)
        rel_error =  diff/sqrt(E_exact**2 + E_demag**2)
        print "Relative error:", rel_error
        res[demagtype]["relE"] = rel_error
        res[demagtype]["nrg"] = E_exact
        assert rel_error < TOL, "Relative error is %g, should be zero" % rel_error

    output = open(os.path.join(MODULE_DIR, "demagenergies.txt"), "w")
    for demagtype in ["FK","GCR"]:
        output.write("%s Demag energy %s\n"%(demagtype,str(res[demagtype]["nrg"])))
        output.write("%s Relative error %s\n"%(demagtype,str(res[demagtype]["relE"])))                  
    output.close()
                  
if __name__ == '__main__':
    test_energy()
