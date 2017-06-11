import pytest
import numpy as np
import dolfin as df
from finmag.energies import Exchange
from finmag.field import Field
from math import sqrt, pi
from finmag.util.consts import mu0
from finmag.util.pbc2d import PeriodicBoundary2D

@pytest.fixture
def fixt():
    """
    Create an Exchange object that will be re-used during testing.

    """
    mesh = df.UnitCubeMesh(10, 10, 10)
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    Ms = Field(df.FunctionSpace(mesh, 'DG', 0), 1)
    A = 1
    m = Field(functionspace)
    exch = Exchange(A)
    exch.setup(m, Ms)
    return {"exch": exch, "m": m, "A": A, "Ms": Ms}


def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and
    has a 'name' attribute.
    """
    exch = Exchange(13e-12, name='MyExchange')
    assert hasattr(exch, 'name')


def test_there_should_be_no_exchange_for_uniform_m(fixt):
    """
    Check that exchange field and energy are 0 for uniform magnetisation.

    """
    FIELD_TOLERANCE = 6e-7
    fixt["m"].set((1, 0, 0))

    H = fixt["exch"].compute_field()
    print "Asserted zero exchange field for uniform m = (1, 0, 0), " + \
        "got H =\n{}.".format(H.reshape((3, -1)))
    print "np.max(np.abs(H)) =", np.max(np.abs(H))
    assert np.max(np.abs(H)) < FIELD_TOLERANCE

    ENERGY_TOLERANCE = 0.0
    E = fixt["exch"].compute_energy()
    print "Asserted zero exchange energy for uniform m = (1, 0, 0), " + \
        "got E = {:g}.".format(E)
    assert abs(E) <= ENERGY_TOLERANCE


def test_exchange_energy_analytical(fixt):
    """
    Compare one Exchange energy with the corresponding analytical result.

    """
    REL_TOLERANCE = 1e-7

    A = 1
    mesh = df.UnitCubeMesh(10, 10, 10)
    Ms = Field(df.FunctionSpace(mesh, 'DG', 0), 1)
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    m = Field(functionspace)
    m.set(df.Expression(("x[0]", "x[2]", "-x[1]"), degree=1))
    exch = Exchange(A)
    exch.setup(m, Ms)
    E = exch.compute_energy()
    # integrating the vector laplacian, the latter gives 3 already
    expected_E = 3

    print "With m = (0, sqrt(1-x^2), x), " + \
        "expecting E = {}. Got E = {}.".format(expected_E, E)
    assert abs(E - expected_E) / expected_E < REL_TOLERANCE


def test_exchange_energy_analytical_2():
    """
    Compare one Exchange energy with the corresponding analytical result.

    """
    REL_TOLERANCE = 5e-5
    lx = 6
    ly = 3
    lz = 2
    nx = 300
    ny = nz = 1
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(lx, ly, lz), nx, ny, nz)
    unit_length = 1e-9
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    Ms = Ms = Field(df.FunctionSpace(mesh, 'DG', 0), 8e5)
    A = 13e-12
    m = Field(functionspace)
    m.set(
        df.Expression(['0', 'sin(2*pi*x[0]/l_x)', 'cos(2*pi*x[0]/l_x)'],
                      l_x=lx, degree=1))
    exch = Exchange(A)
    exch.setup(m, Ms, unit_length=unit_length)
    E_expected = A * 4 * pi ** 2 * \
        (ly * unit_length) * (lz * unit_length) / (lx * unit_length)
    E = exch.compute_energy()
    print "expected energy: {}".format(E)
    print "computed energy: {}".format(E_expected)
    assert abs((E - E_expected) / E_expected) < REL_TOLERANCE


def test_exchange_field_supported_methods(fixt):
    """
    Check that all supported methods give the same results
    as the default method.

    """
    A = 1
    REL_TOLERANCE = 1e-12
    mesh = df.UnitCubeMesh(10, 10, 10)
    Ms = Field(df.FunctionSpace(mesh, 'DG', 0), 1)
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    m = Field(functionspace)
    m.set(df.Expression(("0", "sin(x[0])", "cos(x[0])"), degree=1))
    exch = Exchange(A)
    exch.setup(m, Ms)
    H_default = exch.compute_field()

    supported_methods = list(Exchange._supported_methods)
    # no need to compare default method with itself
    supported_methods.remove(exch.method)
    # the project method for the exchange is too bad
    supported_methods.remove("project")

    for method in supported_methods:
        exch = Exchange(A, method=method)
        exch.setup(m, Ms)
        H = exch.compute_field()
        print "With method '{}', expecting H =\n{}\n, got H =\n{}.".format(
            method, H_default.reshape((3, -1)).mean(1),
            H.reshape((3, -1)).mean(1))

        rel_diff = np.abs((H - H_default) / H_default)
        assert np.nanmax(rel_diff) < REL_TOLERANCE


def test_exchange_periodic_boundary_conditions():

    mesh1 = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 0.1), 2, 2, 1)
    mesh2 = df.UnitCubeMesh(10, 10, 10)

    print("""
    # for debugging, to make sense of output
    # testrun 0, 1 : mesh1
    # testrun 2,3 : mesh2
    # testrun 0, 2 : normal
    # testrun 1,3 : pbc
    """)
    testrun = 0

    for mesh in [mesh1, mesh2]:
        pbc = PeriodicBoundary2D(mesh)
        S3_normal = df.VectorFunctionSpace(mesh, "Lagrange", 1)
        S3_pbc = df.VectorFunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

        for S3 in [S3_normal, S3_pbc]:
            print("Running test {}".format(testrun))
            testrun += 1

            FIELD_TOLERANCE = 6e-7
            ENERGY_TOLERANCE = 0.0

            m_expr = df.Expression(("0", "0", "1"), degree=1)

            m = Field(S3, m_expr, name='m')

            exch = Exchange(1)
            exch.setup(m, Field(df.FunctionSpace(mesh, 'DG', 0), 1))
            field = exch.compute_field()
            energy = exch.compute_energy()
            print("m.shape={}".format(m.vector().array().shape))
            print("m=")
            print(m.vector().array())
            print("energy=")
            print(energy)
            print("shape=")
            print(field.shape)
            print("field=")
            print(field)

            H = field
            print "Asserted zero exchange field for uniform m = (1, 0, 0) " + \
                  "got H =\n{}.".format(H.reshape((3, -1)))
            print "np.max(np.abs(H)) =", np.max(np.abs(H))
            assert np.max(np.abs(H)) < FIELD_TOLERANCE

            E = energy
            print "Asserted zero exchange energy for uniform m = (1, 0, 0), " + \
                  "Got E = {:g}.".format(E)
            assert abs(E) <= ENERGY_TOLERANCE




if __name__ == "__main__":
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(2 * np.pi, 1, 1), 10, 1, 1)

    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    expr = df.Expression(("0", "cos(x[0])", "sin(x[0])"), degree=1)

    m = df.interpolate(expr, S3)

    exch = Exchange(1, pbc2d=True)
    exch.setup(S3, m, 1)
    print exch.compute_field()

    field = df.Function(S3)
    field.vector().set_local(exch.compute_field())

    df.plot(m)
    df.plot(field)
    df.interactive()
