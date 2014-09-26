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


def setup_for_debugging():
    """
    Sets up equation for greater convenience during interactive debugging.

    """
    mesh, V, alpha, W, m, H, dmdt = setup()
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    equation.solve()
    return {'mesh': mesh,
            'V': V, 'alpha': alpha,
            'W': W, 'm': m, 'H': H, 'dmdt': dmdt,
            'equation': equation}


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


def test_solve(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    equation.solve()

    dmdt_expected = df.Function(W)
    dmdt_expected.assign(df.Constant((0.0, 0.5, -0.5)))
    assert same(dmdt.vector(), dmdt_expected.vector())


def test_pinning(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)
    pins = df.Function(V)
    pins.vector()[0] = 1  # pin first node, but this could be done using an expression
    equation.set_pinned_nodes(pins.vector())
    equation.solve()
    dmdt_node0 = dmdt.vector()[0:3]
    dmdt_node_others = dmdt.vector()[3:]
    # check that first node is pinned, i.e. dmdt = 0 there
    assert np.all(dmdt_node0.array() == np.array((0, 0, 0)))
    # check that we don't accidentally set the whole dmdt array to zero
    assert not np.all(dmdt_node_others.array() == np.array((0, 0, 0, 0, 0, 0)))


def test_slonczewski(setup):
    mesh, V, alpha, W, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.set_alpha(alpha.vector())
    equation.set_gamma(1.0)

    Ms = df.Function(V)
    Ms.assign(df.Constant(1))
    J = df.Function(V)
    J.assign(df.Constant(1))
    equation.slonczewski(5e-9, 0.4, np.array((1.0, 0.0, 0.0)), 1, 0)
    assert equation.slonczewski_status() is False  # missing J, Ms
    equation.set_saturation_magnetisation(Ms.vector())
    equation.set_current_density(J.vector())
    assert equation.slonczewski_status() is True
    equation.solve()
