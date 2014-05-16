import dolfin as df
import numpy as np
from finmag import Simulation
from finmag.physics.llg import LLG


def test_spatially_varying_alpha_using_Simulation_class():
    """
    test that I can change the value of alpha through the property sim.alpha
    and that I get an df.Function back.

    """
    length = 20
    simplices = 10
    mesh = df.IntervalMesh(simplices, 0, length)

    sim = Simulation(mesh, Ms=1, unit_length=1e-9)
    sim.alpha = 1
    expected_alpha = np.ones(simplices + 1)
    assert np.array_equal(sim.alpha.vector().array(), expected_alpha)


def test_spatially_varying_alpha_using_LLG_class():
    """
    no property magic here - llg.alpha is a df.Function at heart and can be
    set with any type using llg.set_alpha()

    """
    length = 20
    simplices = 10
    mesh = df.IntervalMesh(simplices, 0, length)

    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    llg = LLG(S1, S3)
    llg.set_alpha(1)
    expected_alpha = np.ones(simplices + 1)

    print "Got:\n", llg.alpha.vector().array()
    print "Expected:\n", expected_alpha
    assert np.array_equal(llg.alpha.vector().array(), expected_alpha)
