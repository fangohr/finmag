import pytest
import numpy as np
import dolfin as df
from equation import equation_module as eq

@pytest.fixture
def setup():
    mesh = df.UnitIntervalMesh(2)
    V = df.FunctionSpace(mesh, "CG", 1)
    alpha = df.Function(V)
    alpha.assign(df.Constant(1))
    W = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    m = df.Function(W)
    m.assign(df.Constant((1, 0, 0)))
    H = df.Function(W)
    H.assign(df.Constant((0, 1, 0)))
    dmdt = df.Function(W)
    return mesh, V, alpha, W, m, H, dmdt


def same(v, w):
    """
    Returns True if the vectors `v` and `w` have the same entries.

    """
    diff = v - w
    diff.abs()
    print "v = {}\nw = {}\ndiff = {}".format(v.array(), w.array(), diff.array())
    return diff.sum() == 0


def test_new_equation(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())


def test_new_equation_wrong_size(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    W = df.VectorFunctionSpace(mesh, "CG", 2, dim=3)  # W like Wrong
    H_W = df.Function(W)
    with pytest.raises(StandardError):
        equation = eq.Equation(m.vector(), H_W.vector(), dmdt.vector())


def test_regression_vector_wrong_state(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    equation.solve()
    # the following operation would fail with PETSc error code 73
    # saying the vector is in wrong state. An "apply" call in the C++
    # code fixes this.
    operation = dmdt.vector() - m.vector()


def test_damping(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    equation.solve()
    dmdt_expected = df.Function(W)
    dmdt_expected.assign(df.Constant((0, 0.5, 0)))
    assert same(dmdt.vector(), dmdt_expected.vector())


def test_alpha_not_set(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    assert equation.get_alpha() is None  # doesn't crash
    with pytest.raises(RuntimeError):
        equation.solve()


def test_alpha_keeps_track_of_change(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    assert same(alpha.vector(), equation.get_alpha())
    # since alpha and Equation::alpha are fundamentally the same object
    # changing one should change the other, which is what we test next
    alpha.assign(df.Constant(2))
    assert same(alpha.vector(), equation.get_alpha())
