import os
from dolfin import Mesh
from numpy import average
from finmag.sim.llg import LLG
from finmag.util.convert_mesh import convert_mesh

TOL = 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Using mesh with radius 10 nm (nmag ex. 1)
mesh = Mesh(convert_mesh(MODULE_DIR + "/sphere1.geo"))
unit_length = 1e-9
llg = LLG(mesh, unit_length=unit_length)
llg.set_m((1, 0, 0))
llg.Ms = 1e6
llg.setup(use_demag=True)

def test_field():
    """
    Test the demag field.

    H_demag should be equal to -1/3 M, and with m = (1, 0 ,0)
    and Ms = 1,this should give H_demag = (-1/3, 0, 0).

    """
    # Compute demag field
    H_demag = llg.demag.compute_field()
    H_demag.shape = (3, -1)
    x, y, z = H_demag[0], H_demag[1], H_demag[2]

    print "Max values in direction:"
    print "x: %g,  y: %g,  z: %g" % (max(x), max(y), max(z))
    print "Min values in direction:"
    print "x: %g,  y: %g,  z: %g" % (min(x), min(y), min(z))

    x, y, z = average(x), average(y), average(z)
    print "Average values in direction"
    print "x: %g,  y: %g,  z: %g" % (x, y, z)

    # Compute relative erros
    x = abs((x + 1./3*llg.Ms)/llg.Ms)
    y = abs(y/llg.Ms)
    z = abs(z/llg.Ms)

    print "Relative error:"
    print "x: %g,  y: %g,  z: %g" % (x, y, z)
    assert x < TOL, "x-average is %g, should be -1/3." % x
    assert y < TOL, "y-average is %g, should be zero." % y
    assert z < TOL, "z-average is %g, should be zero." % z

if __name__ == '__main__':
    test_field()
