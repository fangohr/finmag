import os
import commands
import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.dmi import DMI
from finmag.sim.anisotropy import UniaxialAnisotropy

TOL = 1e-14

def test_exchange_energy_density():
    """
    Compare solution with nmag for now. Should derive the
    analytical solution here as well.

    Our initial magnetisation looks like this:

    ^   ~
    |  /  --->  \   |       (Hahahaha! :-D)
                 ~  v

    """
    TOL = 1e-7 # Should be lower when comparing with analytical solution

    # run nmag
    cmd = "nsim run_nmag_Eexch.py --clean"
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        print ouput
        sys.exit("Error %d: Running %s failed." % (status, cmd))
    nmag_data = np.load("nmag_exchange_energy_density.npy")

    # run finmag
    mesh = df.Interval(100, 0, 10e-9)
    llg = LLG(mesh)
    llg.Ms = 1
    llg.set_m(("cos(x[0]*pi/10e-9)", "sin(x[0]*pi/10e-9)", "0"))
    llg.setup(use_exchange=True, use_dmi=False, use_demag=False)
    finmag_data = llg.exchange.energy_density()

    print "Expecting low relative error when comparing with nmag."
    rel_err = np.abs(nmag_data - finmag_data)/np.linalg.norm(nmag_data)
    print "Max relative error:", np.max(rel_err)
    assert np.max(rel_err) < TOL, \
            "Max relative error is %g, should be zero." % np.max(rel_err)


def test_anisotropy_energy_density():
    """
    Written in sperical coordinates, the equation for the
    anisotropy energy density reads

        E/V = K*sin^2(theta),

    where theta is the angle between the magnetisation and
    the easy axis. With a magnetisation pointing 45 degrees
    between the x- and z-axis, and using the z-axis as the
    easy axis, theta becomes pi/4. sin^2(pi/4) evaluates
    to 1/2, and with K set to 1 in this simple test case,
    we expect the energy density to be 1/2 at every node.

    """
    # 5 simplices between 0 and 1 nm.
    mesh = df.Interval(5, 0, 1e-9)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)

    # Initial magnetisation 45 degress between x- and z-axis.
    m_vec = df.Constant((1/np.sqrt(2), 0, 1/np.sqrt(2)))
    m = df.interpolate(m_vec, V)

    # Easy axis in z-direction.
    a = df.Constant((0, 0, 1))

    # These are 1 just to simplify the analytical solution.
    K = 1
    Ms = 1

    anis = UniaxialAnisotropy(V, m, K, a, Ms)
    density = anis.energy_density()
    deviation = np.abs(density - 0.5)

    print "Anisotropy energy density (expect array of 0.5):"
    print density
    print "Max deviation: %g" % np.max(deviation)

    assert np.all(deviation < TOL), \
        "Max deviation %g, should be zero." % np.max(deviation)


def test_DMI_energy_density_2D():
    """
    For a vector field (x, y, z) = 0.5 * (-y, x, c),
    the curl is exactly 1.0. (HF)

    """
    mesh = df.UnitSquare(4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    M = df.interpolate(df.Expression(("-0.5*x[1]", "0.5*x[0]", "1")), V)
    Ms = 1
    D = 1
    dmi = DMI(V, M, D, Ms)
    density = dmi.energy_density()
    deviation = np.abs(density - 1.0)

    print "2D energy density (expect array of 1):"
    print density
    print "Max deviation: %g" % np.max(deviation)

    assert np.all(deviation < TOL), \
        "Max deviation %g, should be zero." % np.max(deviation)


def test_DMI_energy_density_3D():
    """Same as above, on a 3D mesh."""
    mesh = df.UnitCube(4, 4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    M = df.interpolate(df.Expression(("-0.5*x[1]", "0.5*x[0]", "1")), V)
    Ms = 1
    D = 1
    dmi = DMI(V, M, D, Ms)
    density = dmi.energy_density()
    deviation = np.abs(density - 1.0)

    print "3D energy density (expect array of 1):"
    print density
    print "Max deviation: %g" % np.max(deviation)

    assert np.all(deviation < TOL), \
        "Max deviation %g, should be zero." % np.max(deviation)


def test_demag_energy_density():
    """
    With a sphere mesh, unit magnetisation in x-direction,
    we expect the demag energy to be

        E = 1/6 * mu0 * Ms^2 * V.

    (See section about demag solvers in the documentation
    for how this is found.)

    To make it simple, we define Ms = sqrt(6/mu0).

    Energy density is then

        E/V = 1.

    """
    TOL = 5e-2

    mesh = df.UnitSphere(5)
    mu0 = 4*np.pi*1e-7
    llg = LLG(mesh)
    llg.Ms = np.sqrt(6.0/mu0)
    llg.set_m((1,0,0))
    llg.setup(use_demag=True)

    density = llg.demag.energy_density()
    deviation = np.abs(density - 1.0)

    print "Demag energy density (expect array of 1s):"
    print density
    print "Max deviation:", np.max(deviation)
    assert np.max(deviation) < TOL, \
            "Max deviation is %g, should be zero." % np.max(deviation)

if __name__ == '__main__':
    test_demag_energy_density()
    test_exchange_energy_density()
    test_anisotropy_energy_density()
    test_DMI_energy_density_2D()
    test_DMI_energy_density_3D()
