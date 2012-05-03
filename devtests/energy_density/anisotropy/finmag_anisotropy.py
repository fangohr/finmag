import numpy as np
import dolfin as df
from finmag.sim.anisotropy import UniaxialAnisotropy

TOL = 1e-15

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

if __name__ == '__main__':
    test_anisotropy_energy_density()
