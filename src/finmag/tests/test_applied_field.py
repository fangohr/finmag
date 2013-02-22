import numpy as np
import dolfin as df
from finmag import Simulation as Sim
from finmag.energies import Zeeman

Ms = 8.6e5


def test_uniform_external_field():
    TOLERANCE = 3.5e-10

    mesh = df.UnitCube(2, 2, 2)
    sim = Sim(mesh, Ms)
    sim.set_m((1, 0, 0))
    sim.add(Zeeman((0, Ms, 0)))
    sim.alpha = 1.0
    sim.run_until(1e-9)

    m = sim.m.reshape((3, -1)).mean(-1)
    expected_m = np.array([0, 1, 0])
    diff = np.abs(m - expected_m)
    assert np.max(diff) < TOLERANCE


def test_negative_uniform_external_field():
    TOLERANCE = 1e-10

    mesh = df.UnitCube(2, 2, 2)
    sim = Sim(mesh, Ms)
    sim.set_m((1, 0.1, 0))  # slightly misaligned
    sim.add(Zeeman((-1.0 * Ms, 0, 0)))
    sim.alpha = 1.0
    sim.run_until(1e-9)

    m = sim.m.reshape((3, -1)).mean(-1)
    print "Average magnetisation ({:.2g}, {:.2g}, {:.2g}).".format(*m)
    expected_m = np.array([-1, 0, 0])
    diff = np.abs(m - expected_m)
    TOLERANCE = 1e-5
    assert np.max(diff) < TOLERANCE


def test_non_uniform_external_field():
    TOLERANCE = 1e-9

    length = 10e-9
    vertices = 5
    mesh = df.Interval(vertices, 0, length)
    sim = Sim(mesh, Ms)
    sim.set_m((1, 0, 0))
    # applied field
    # (0, -H, 0) for 0 <= x <= a
    # (0, +H, 0) for a <  x <= length
    H_expr = df.Expression(("0", "H*(x[0]-a)/fabs(x[0]-a)", "0"), a=length/2, H=Ms/2)
    sim.add(Zeeman(H_expr))
    sim.alpha = 1.0
    sim.run_until(1e-9)

    m = sim.m.reshape((3, -1)).mean(-1)
    expected_m = np.array([0, 0, 0])
    diff = np.abs(m - expected_m)
    assert np.max(diff) < TOLERANCE
