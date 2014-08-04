import pytest
import numpy as np
import dolfin as df
from finmag.energies import Exchange
from finmag.field import Field
from math import sqrt, pi
from finmag.util.consts import mu0


@pytest.fixture(scope = "module")
def fixt():
    """
    Create an Exchange object that will be re-used during testing.

    """
    mesh = df.UnitCubeMesh(10, 10, 10)
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    Ms = 1
    A = 1
    m = Field(functionspace)
    exch = Exchange(A)
    exch.setup(m, Ms)
    return {"exch": exch, "m": m, "A": A, "Ms": Ms}


def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    exch = Exchange(13e-12, name='MyExchange')
    assert hasattr(exch, 'name')


def test_there_should_be_no_exchange_for_uniform_m(fixt):
    """
    Check that exchange field and energy are 0 for uniform magnetisation.

    """
    TOLERANCE = 1e-6
    fixt["m"].set((1, 0, 0))

    H = fixt["exch"].compute_field()
    print "Asserted zero exchange field for uniform m = (1, 0, 0), got H =\n{}.".format(H.reshape((3, -1)))
    assert np.max(np.abs(H)) < TOLERANCE

    E = fixt["exch"].compute_energy()
    print "Asserted zero exchange energy for uniform m = (1, 0, 0), got E = {}.".format(E)
    assert abs(E) < TOLERANCE


def test_exchange_energy_analytical(fixt):
    """
    Compare one Exchange energy with the corresponding analytical result.

    """
    REL_TOLERANCE = 1e-7

    A = 1
    Ms = 1
    mesh = df.UnitCubeMesh(10, 10, 10)
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    m = Field(functionspace)
    m.set(df.Expression(("x[0]", "x[2]", "-x[1]")))
    exch = Exchange(A)
    exch.setup(m, Ms)
    E = exch.compute_energy()
    expected_E = 3 # integrating the vector laplacian, the latter gives 3 already

    print "With m = (0, sqrt(1-x^2), x), expecting E = {}. Got E = {}.".format(expected_E, E)
    assert abs(E - expected_E)/expected_E < REL_TOLERANCE


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
    mesh = df.BoxMesh(0, 0, 0, lx, ly, lz, nx, ny, nz)
    unit_length = 1e-9
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    Ms = 8e5
    A = 13e-12
    m = Field(functionspace)
    m.set(df.Expression(['0', 'sin(2*pi*x[0]/l_x)', 'cos(2*pi*x[0]/l_x)'], l_x=lx))
    exch = Exchange(A)
    exch.setup(m, Ms, unit_length=unit_length)
    E_expected = A * 4 * pi**2 * (ly * unit_length) * (lz * unit_length) / (lx * unit_length)
    E = exch.compute_energy()
    print "expected energy: {}".format(E)
    print "computed energy: {}".format(E_expected)
    assert abs((E - E_expected) / E_expected) < REL_TOLERANCE


def test_exchange_field_supported_methods(fixt):
    """
    Check that all supported methods give the same results as the default method.

    """
    A = 1
    Ms = 1
    REL_TOLERANCE = 1e-12
    mesh = df.UnitCubeMesh(10, 10, 10)
    functionspace = df.VectorFunctionSpace(mesh, "CG", 1, 3)
    m = Field(functionspace)
    m.set(df.Expression(("0", "sin(x[0])", "cos(x[0])")))
    exch = Exchange(A)
    exch.setup(m, Ms)
    H_default = exch.compute_field()

    supported_methods = list(Exchange._supported_methods)
    supported_methods.remove(exch.method) # no need to compare default method with itself
    supported_methods.remove("project") # the project method for the exchange is too bad

    for method in supported_methods:
        exch = Exchange(A, method=method)
        exch.setup(m, Ms)
        H = exch.compute_field()
        print "With method '{}', expecting H =\n{}\n, got H =\n{}.".format(
            method, H_default.reshape((3, -1)).mean(1), H.reshape((3, -1)).mean(1))

        rel_diff = np.abs((H - H_default) / H_default)
        assert np.nanmax(rel_diff) < REL_TOLERANCE


def test_exchange_length(fixt):
    mesh = df.UnitCubeMesh(10, 10, 10)
    functionspace = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = Field(functionspace)
    Ms = 8e5
    A = 13e-12
    l_ex_expected = sqrt(2 * A / (mu0 * Ms**2))

    # Test with various options for A and Ms: pure number;
    # df.Constant, df.Expression.
    exch = Exchange(A)
    exch.setup(m, Ms)
    l_ex = exch.exchange_length()
    assert(np.allclose(l_ex, l_ex_expected, atol=0))

    exch2 = Exchange(df.Constant(A))
    exch2.setup(m, df.Constant(Ms))
    l_ex2 = exch2.exchange_length()
    assert(np.allclose(l_ex2, l_ex_expected, atol=0))

    exch3 = Exchange(df.Expression('A', A=A))
    exch3.setup(m, df.Expression('Ms', Ms=Ms))
    l_ex3 = exch3.exchange_length()
    assert(np.allclose(l_ex3, l_ex_expected, atol=0))

    # We should get an error with spatially non-uniform values of A or Ms
    exch4 = Exchange(df.Expression('A*x[0]', A=A))
    exch4.setup(m, Ms)
    with pytest.raises(ValueError):
        exch4.exchange_length()

    exch5 = Exchange(A)
    exch5.setup(m, df.Expression('Ms*x[0]', Ms=Ms))
    with pytest.raises(ValueError):
        exch5.exchange_length()


if __name__ == "__main__":
    mesh = df.BoxMesh(0,0,0,2*np.pi,1,1,10, 1, 1)
     
    S = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    expr = df.Expression(("0", "cos(x[0])", "sin(x[0])"))
    
    m = df.interpolate(expr, S3)
    
    exch = Exchange(1,pbc2d=True)
    exch.setup(S3, m, 1)
    print exch.compute_field()
    
    field=df.Function(S3)
    field.vector().set_local(exch.compute_field())
    
    df.plot(m)
    df.plot(field)
    df.interactive()
