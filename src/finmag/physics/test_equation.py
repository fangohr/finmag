import pytest
import numpy as np
import dolfin as df
from equation import equation_module


def test_new_equation():
    mesh = df.UnitIntervalMesh(2)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    m = df.Function(V)
    H = df.Function(V)
    dmdt = df.Function(V)
    equation = equation_module.Equation(m.vector(), H.vector(), dmdt.vector())


def test_new_equation_wrong_size():
    mesh = df.UnitIntervalMesh(2)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    W = df.VectorFunctionSpace(mesh, "CG", 2, dim=3)  # W like Wrong
    m = df.Function(V)
    Haha = df.Function(W)
    dmdt = df.Function(V)
    with pytest.raises(StandardError):
        equation = equation_module.Equation(m.vector(), Haha.vector(), dmdt.vector())

