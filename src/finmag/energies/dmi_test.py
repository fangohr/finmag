import numpy as np
import dolfin as df
from math import pi
from finmag.energies import DMI, Exchange
from finmag import Simulation
from finmag.util.helpers import vector_valued_function
from finmag.util.pbc2d import PeriodicBoundary2D


def test_dmi_uses_unit_length_2dmesh():
    """
    Set up a helical state in two meshes (one expressed in SI units
    the other expressed in nanometers) and compute energies and fields.

    """
    A = 8.78e-12  # J/m
    D = 1.58e-3  # J/m^2
    Ms = 3.84e5  # A/m

    energies = []

    # unit_lengths 1e-9 and 1 are common, let's through in an intermediate
    # length just to challenge the system a little:
    for unit_length in (1, 1e-4, 1e-9):
        radius = 200e-9 / unit_length
        maxh = 5e-9 / unit_length
        helical_period = (4 * pi * A / D) / unit_length
        k = 2 * pi / helical_period
        #HF 27 April 2014: The next command fails in dolfin 1.3
        #mesh = df.CircleMesh(df.Point(0, 0), radius, maxh)
        #The actual shape of the domain shouldn't matter for the test,
        #so let's use a Rectangular mesh which should work the same:

        nx = ny = int(round(radius/maxh))
        mesh = df.RectangleMesh(0, 0 ,  radius, radius, nx, ny)

        S3 = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
        m_expr = df.Expression(("0", "cos(k * x[0])", "sin(k * x[0])"), k=k)
        m = df.interpolate(m_expr, S3)
        dmi = DMI(D)
        dmi.setup(S3, m, Ms, unit_length=unit_length)
        energies.append(dmi.compute_energy())

        H = df.Function(S3)
        H.vector()[:] = dmi.compute_field()
        print H(0.0, 0.0)

        print "Mesh with radius {} and maxh {} and {} vertices.".format(radius, maxh, mesh.num_vertices())
        print "Using unit_length = {}.".format(unit_length)
        print "Helical period {}.".format(helical_period)
        print "Energy {}.".format(dmi.compute_energy())

    rel_diff_energies = abs(energies[0] - energies[1]) / abs(energies[1])
    print "Relative difference of energy {}.".format(rel_diff_energies)
    assert rel_diff_energies < 1e-13

    rel_diff_energies2 = abs(energies[0] - energies[2]) / abs(energies[2])
    print "Relative difference2 of energy {}.".format(rel_diff_energies2)
    assert rel_diff_energies2 < 1e-13


def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    dmi = DMI(1)
    assert hasattr(dmi, 'name')


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
        print p[0]
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
    D = 5e-3
    sim.add(Exchange(A))
    sim.add(DMI(D))

    sim.relax(stopping_dmdt=0.001)

    if plot:
        df.plot(sim._m)
        df.interactive()

    mx=[sim._m(x+0.5,1)[0] for x in range(20)]
    
    assert np.max(np.abs(mx)) < 1e-6


if __name__ == "__main__":
    #test_dmi_pbc2d()
    test_dmi_pbc2d_1D(plot=True)
