import os
import sys
import commands
import numpy as np
import dolfin as df
#from finmag.energies import UniaxialAnisotropy, Exchange, Demag, DMI
from finmag.energies import UniaxialAnisotropy, Exchange, Demag, DMI, DMI_Old
from finmag.util.consts import mu0
from finmag.util.meshes import sphere

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
    TOL = 1e-7  # Should be lower when comparing with analytical solution
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # run nmag
    cmd = "nsim %s --clean" % os.path.join(MODULE_DIR, "run_nmag_Eexch.py")
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        print output
        sys.exit("Error %d: Running %s failed." % (status, cmd))
    nmag_data = np.loadtxt(os.path.join(MODULE_DIR, "nmag_exchange_energy_density.txt"))

    # run finmag
    mesh = df.IntervalMesh(100, 0, 10e-9)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    Ms = 42
    m = df.interpolate(df.Expression(("cos(x[0]*pi/10e-9)", "sin(x[0]*pi/10e-9)", "0")), S3)

    exch = Exchange(1.3e-11)
    exch.setup(S3, m, Ms)

    finmag_data = exch.energy_density()
    rel_err = np.abs(nmag_data - finmag_data) / np.linalg.norm(nmag_data)

    print ("Nmag data   = %g" % nmag_data[0])
    print ("Finmag data = %g" % finmag_data[0])
    print "Relative error from nmag data (expect array of 0):"
    print rel_err
    print "Max relative error:", np.max(rel_err)
    assert np.max(rel_err) < TOL, \
            "Max relative error is %g, should be zero." % np.max(rel_err)


    print("Work out average energy density and energy")
    S1 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=1)
    #only approximative -- using assemble would be better
    average_energy_density = np.average(finmag_data)
    w = df.TestFunction(S1)
    vol = sum(df.assemble(df.dot(df.Constant([1]), w) * df.dx))
    #finmag 'manually' computed, based on node values of energy density:
    energy1 = average_energy_density * vol   
    #finmag computed by energy class
    energy2 = exch.compute_energy()
    #comparison with Nmag
    energy3 = np.average(nmag_data) * vol
    print energy1, energy2, energy3
    
    assert abs(energy1 - energy2) < 1e-12 # actual value is 0, but 
                                          # that must be pure luck.
    assert abs(energy1 - energy3) < 5e-8  # actual value
                                          # is 1.05e-8, 30 June 2012

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
    mesh = df.IntervalMesh(5, 0, 1e-9)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)

    # Initial magnetisation 45 degress between x- and z-axis.
    m_vec = df.Constant((1 / np.sqrt(2), 0, 1 / np.sqrt(2)))
    m = df.interpolate(m_vec, V)

    # Easy axis in z-direction.
    a = df.Constant((0, 0, 1))

    # These are 1 just to simplify the analytical solution.
    K = 1
    Ms = 1

    anis = UniaxialAnisotropy(K, a)
    anis.setup(V, m, Ms)
    density = anis.energy_density()
    deviation = np.abs(density - 0.5)

    print "Anisotropy energy density (expect array of 0.5):"
    print density
    print "Max deviation: %g" % np.max(deviation)

    assert np.all(deviation < TOL), \
        "Max deviation %g, should be zero." % np.max(deviation)


def test_DMI_Old_energy_density_2D():
    """
    For a vector field (x, y, z) = 0.5 * (-y, x, c),
    the curl is exactly 1.0. (HF)

    """
    mesh = df.UnitSquareMesh(4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    M = df.interpolate(df.Expression(("-0.5*x[1]", "0.5*x[0]", "1")), V)
    Ms = 1
    D = 1
    dmi = DMI_Old(D)
    dmi.setup(V, M, Ms)
    density = dmi.energy_density()
    deviation = np.abs(density - 1.0)

    print "2D energy density (expect array of 1):"
    print density
    print "Max deviation: %g" % np.max(deviation)

    assert np.all(deviation < TOL), \
        "Max deviation %g, should be zero." % np.max(deviation)


def test_DMI_energy_density_3D():
    """Same as above, on a 3D mesh."""
    mesh = df.UnitCubeMesh(4, 4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    M = df.interpolate(df.Expression(("-0.5*x[1]", "0.5*x[0]", "1")), V)
    Ms = 10
    D = 1 / (mu0 * Ms)
    dmi = DMI(D)
    dmi.setup(V, M, Ms)
    density = dmi.energy_density()
    deviation = np.abs(density - 1.0)

    print "3D energy density (expect array of 1):"
    print density
    print "Max deviation: %g" % np.max(deviation)

    assert np.all(deviation < TOL), \
        "Max deviation %g, should be zero." % np.max(deviation)

def test_DMI_Old_energy_density_3D():
    """Same as above, on a 3D mesh."""
    mesh = df.UnitCubeMesh(4, 4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    M = df.interpolate(df.Expression(("-0.5*x[1]", "0.5*x[0]", "1")), V)
    Ms = 10
    D = 1
    dmi = DMI_Old(D)
    dmi.setup(V, M, Ms)
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

    mesh = sphere(r = 1.0, maxh = 0.2)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    mu0 = 4 * np.pi * 1e-7

    demag = Demag()
    m = df.interpolate(df.Constant((1, 0, 0)), S3)
    Ms = np.sqrt(6.0 / mu0)
    demag.setup(S3, m, Ms, 1)

    density = demag.demag.energy_density()
    deviation = np.abs(density - 1.0)

    print "Demag energy density (expect array of 1s):"
    print density
    print "Max deviation:", np.max(deviation)
    assert np.max(deviation) < TOL, \
            "Max deviation is %g, should be zero." % np.max(deviation)

if __name__ == '__main__':
    test_exchange_energy_density()
    sys.exit(0)
    test_demag_energy_density()
    test_exchange_energy_density()
    test_anisotropy_energy_density()
    test_DMI_energy_density_2D()
    test_DMI_energy_density_3D()
