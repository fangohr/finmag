import pytest
import numpy as np
import dolfin as df
from equation import equation_module as eq

@pytest.fixture
def setup():
    mesh = df.UnitIntervalMesh(2)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    m = df.Function(V)
    m.assign(df.Constant((1, 0, 0)))
    H = df.Function(V)
    H.assign(df.Constant((0, 1, 0)))
    dmdt = df.Function(V)
    return mesh, V, m, H, dmdt


def same(f, g):
    """
    Returns True if the functions `f` and `g` have the same vector-entries.

    """
    diff = f.vector() - g.vector()
    diff.abs()
    return diff.sum() == 0


def test_new_equation(setup):
    mesh, V, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())


def test_new_equation_wrong_size(setup):
    mesh, V, m, H, dmdt = setup
    W = df.VectorFunctionSpace(mesh, "CG", 2, dim=3)  # W like Wrong
    H_W = df.Function(W)
    with pytest.raises(StandardError):
        equation = eq.Equation(m.vector(), H_W.vector(), dmdt.vector())


def _test_damping(setup):
    mesh, V, m, H, dmdt = setup
    equation = eq.Equation(m.vector(), H.vector(), dmdt.vector())
    equation.solve()

    dmdt_expected = df.Function(V)
    dmdt_expected.assign(df.Constant((0, 1, 0)))
    assert same(dmdt, dmdt_expected)
