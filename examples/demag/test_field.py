# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - `import dolfin` removed; VectorFunctionSpace/FunctionSpace -> dolfinx.fem;
#     df.interpolate(df.Constant(...)) -> Field(S3, constant) (ported Field API).
#   - from_geofile("sphere1.geo") kept UNCHANGED: the Netgen-CSG '.geo' loader
#     was ported for the examples subset (Task 30 amendment, see
#     finmag.util.geofile).
#   - compute_field() ordering: see the INTERFACE-DRIFT note inline below.
# [Claude Opus 4.8]
import os
from numpy import average
from dolfinx import fem
from finmag.energies import Demag
from finmag.field import Field
from finmag.util.meshes import from_geofile

TOL = 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
Ms = 1e6


def test_field():
    """
    Test the demag field.

    H_demag should be equal to -1/3 M, and with m = (1, 0 ,0)
    and Ms = 1, this should give H_demag = (-1/3, 0, 0).

    """
    # Using mesh with radius 10 nm (nmag ex. 1)
    mesh = from_geofile(os.path.join(MODULE_DIR, "sphere1.geo"))
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    m = Field(S3, (1.0, 0.0, 0.0))

    demag = Demag()
    demag.setup(m, Field(fem.functionspace(mesh, ("DG", 0)), Ms), unit_length=1e-9)

    # Compute demag field.
    # INTERFACE-DRIFT: the legacy compute_field() returned component-blocked
    # order (xxx...yyy...zzz), reshaped as (3, -1). The DOLFINx port returns
    # coordinate-interleaved order (xyz xyz ...), so we reshape as (-1, 3) and
    # take columns. (Cf. src/finmag/tests/test_fk_demag_dolfinx.py:256.)
    H_demag = demag.compute_field().reshape(-1, 3)
    x, y, z = H_demag[:, 0], H_demag[:, 1], H_demag[:, 2]

    print("Max values in direction:")
    print("x: %g,  y: %g,  z: %g" % (max(x), max(y), max(z)))
    print("Min values in direction:")
    print("x: %g,  y: %g,  z: %g" % (min(x), min(y), min(z)))

    x, y, z = average(x), average(y), average(z)
    print("Average values in direction")
    print("x: %g,  y: %g,  z: %g" % (x, y, z))

    # Compute relative errors
    x = abs((x + 1. / 3 * Ms) / Ms)
    y = abs(y / Ms)
    z = abs(z / Ms)

    print("Relative error:")
    print("x: %g,  y: %g,  z: %g" % (x, y, z))
    assert x < TOL, "x-average is %g, should be -1/3." % x
    assert y < TOL, "y-average is %g, should be zero." % y
    assert z < TOL, "z-average is %g, should be zero." % z


if __name__ == '__main__':
    test_field()
    print("OK: demag field of a uniformly magnetised sphere is -1/3 M.")
