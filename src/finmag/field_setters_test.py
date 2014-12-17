"""
Ensure that field values can be set using a wide range of types and make sure
we don't assign a new dolfin function, but only overwrite the values of the
existing function.

"""
import dolfin as df
import numpy as np
import pytest
from .field import Field

EPSILON = 1e-14


@pytest.fixture
def setup():
    mesh = df.UnitIntervalMesh(1)
    F = df.FunctionSpace(mesh, "CG", 1)
    F_DG = df.FunctionSpace(mesh, "DG", 0)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    return Field(F), Field(F_DG), Field(V)


def test_set_with_dolfin_constant(setup):
    scalar_field, scalar_field_dg, vector_field = setup

    v = df.Constant(1)
    for field in (scalar_field, scalar_field_dg):
        # keep track of dolfin function object to make sure we don't overwrite it
        f = field.f
        assert f.vector().array().all() == 0
        field.set(v)
        assert f.vector().array().all() == 1

    v = df.Constant((1, 2, 3))
    f = vector_field.f
    assert f.vector().array().all() == 0
    vector_field.set(v)
    assert np.allclose(f(0), (1, 2, 3))


def test_set_with_dolfin_expression(setup):
    scalar_field, scalar_field_dg, vector_field = setup

    v = df.Expression("x[0]")
    for field, expected in ((scalar_field, 1), (scalar_field_dg, 0.5)):
        f = field.f
        assert f(1) == 0
        field.set(v)
        assert abs(f(1) - expected) <= EPSILON

    v = df.Expression(("1", "2", "3 * x[0]"))
    f = vector_field.f
    assert np.allclose(f(1), (0, 0, 0))
    vector_field.set(v)
    assert np.allclose(f(1), (1, 2, 3))


def test_set_with_dolfin_function():
    mesh = df.UnitIntervalMesh(1)
    F = df.FunctionSpace(mesh, "CG", 1)
    function = df.Function(F)
    function.assign(df.Constant(1))
    field = Field(F)
    function_of_field = field.f
    assert function_of_field(0) == 0
    field.set(function)
    assert abs(function_of_field(0) - 1) <= EPSILON


def test_set_with_another_field():
    mesh = df.UnitIntervalMesh(1)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    field1 = Field(V)
    field1.set(df.Constant((1, 2, 3)))

    field2 = Field(V)
    function_of_field2 = field2.f
    assert function_of_field2.vector().array().all() == 0
    field2.set(field1)
    assert np.allclose(function_of_field2(1), (1, 2, 3))


def test_set_with_another_field_new_but_same_function_space():
    mesh = df.UnitIntervalMesh(1)
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    fieldV = Field(V)
    fieldV.set(df.Constant((1, 2, 3)))

    W = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    fieldW = Field(W)
    function_of_fieldW = fieldW.f
    assert function_of_fieldW.vector().array().all() == 0
    fieldW.set(fieldV)
    assert np.allclose(function_of_fieldW(1), (1, 2, 3))


def test_assumption_that_interpolate_better_than_project_same_vectorspace():
    mesh = df.UnitCubeMesh(2, 2, 2)

    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    f = df.Function(V)
    f.vector()[:] = np.random.rand(len(f.vector().array()))

    W = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    w1 = df.interpolate(f, W)
    w2 = df.project(f, W)

    diff_w1 = f.vector() - w1.vector()
    diff_w2 = f.vector() - w2.vector()
    diff_w1.abs()
    diff_w2.abs()
    assert diff_w1.array().max() <= diff_w2.array().max()


def test_assumption_that_interpolate_better_than_project_different_vectorspace(do_plot=False):
    mesh = df.UnitSquareMesh(5, 5)

    V = df.FunctionSpace(mesh, "DG", 0)
    # models use case of material parameter
    f = df.interpolate(df.Expression("x[0] <= 0.5 ? 0 : 1"), V)  

    W = df.FunctionSpace(mesh, "CG", 1)
    w_i = df.interpolate(f, W)
    w_p = df.project(f, W)
    w_e = df.interpolate(df.Expression("x[0] <= 0.5 ? 0 : 1"), W)

    if do_plot:  # proof by "looking at the picture" /s
        df.plot(f, title="original")
        df.plot(w_i, title="interpolate")
        df.plot(w_p, title="project")
        df.plot(w_e, title="from same expression")
        df.interactive()

    diff_w_i = w_e.vector() - w_i.vector()
    diff_w_p = w_e.vector() - w_p.vector()
    diff_w_i.abs()
    diff_w_p.abs()
    assert diff_w_i.array().max() <= diff_w_p.array().max()


def test_set_with_dolfin_generic_vector():
    mesh = df.UnitIntervalMesh(1)
    F = df.FunctionSpace(mesh, "CG", 1)
    function = df.Function(F)
    function.assign(df.Constant(1))
    field = Field(F)
    function_of_field = field.f
    assert function_of_field(0) == 0
    field.set(function.vector())
    assert abs(function_of_field(0) - 1) <= EPSILON


def test_set_with_dolfin_expression_ingredients():
    mesh = df.UnitIntervalMesh(1)
    field = Field(df.FunctionSpace(mesh, "CG", 1))
    field.set("a * x[0]", a=1)
    assert abs(field.f(1) - 1) <= EPSILON

    field = Field(df.VectorFunctionSpace(mesh, "CG", 1, dim=3))
    field.set(("1", "2", "3"))
    assert np.allclose(field.f(1), (1, 2, 3))
