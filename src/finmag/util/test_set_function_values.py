import pytest
import numpy as np
import dolfin as df
from finmag.field import Field
from set_function_values import *

EPSILON = 1e-14


@pytest.fixture
def f():
    mesh = df.UnitIntervalMesh(1)
    V = df.FunctionSpace(mesh, "CG", 1)
    f = df.Function(V)
    return f


@pytest.fixture
def vf():
    mesh = df.UnitIntervalMesh(1)
    W = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    vf = df.Function(W)
    return vf


def test_function_from_constant(f):
    value = 1
    from_constant(f, df.Constant(value))
    assert abs(f(0) - value) < EPSILON


def test_vector_function_from_constant(vf):
    value = np.array((1, 2, 3))
    from_constant(vf, df.Constant(value))
    assert np.max(np.abs((vf(0) - value))) < EPSILON


def test_function_from_expression(f):
    value = 1
    from_expression(f, df.Expression(str(value)))
    assert abs(f(0) - value) < EPSILON


def test_vector_function_from_expression(vf):
    value = np.array((1, 2, 3))
    from_expression(vf, df.Expression(map(str, value)))
    assert np.max(np.abs(vf(0) - value)) < EPSILON


def test_function_from_field(f):
    value = 1
    # why does the following line fail? it prevents us from creating
    # a Field instance from a dolfin function
    #assert isinstance(f.function_space(), df.FunctionSpace)
    #from_field(f, Field(f.function_space(), value))
    #assert abs(f(0) - value) < EPSILON

