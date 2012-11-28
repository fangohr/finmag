import pytest
import numpy as np
import dolfin as df
from finmag.energies import Exchange

@pytest.fixture(scope = "module")
def fixt():
    """
    Create an Exchange object that will be re-used during testing.

    """
    mesh = df.UnitCube(10, 10, 10)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    Ms = 1
    A = 1
    m = df.Function(S3)
    exch = Exchange(A)
    exch.setup(S3, m, Ms)
    return {"exch": exch, "m": m, "A": A, "S3": S3, "Ms": Ms}

def test_there_should_be_no_exchange_for_uniform_m(fixt):
    """
    Check that exchange field and energy are 0 for uniform magnetisation.

    """
    TOLERANCE = 1e-6
    fixt["m"].assign(df.Constant((1, 0, 0)))

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

    f = df.project(df.Expression(("x[0]", "x[2]", "-x[1]")), fixt["S3"])
    fixt["m"].vector().set_local(f.vector().array())
    E = fixt["exch"].compute_energy()
    expected_E = 3 # integrating the vector laplacian, the latter gives 3 already

    print "With m = (0, sqrt(1-x^2), x), expecting E = {}. Got E = {}.".format(expected_E, E)
    assert abs(E - expected_E)/expected_E < REL_TOLERANCE

def test_exchange_field_supported_methods(fixt):
    """
    Check that all supported methods give the same results as the default method.

    """
    REL_TOLERANCE = 1e-12

    f = df.project(df.Expression(("0", "sin(x[0])", "cos(x[0])")), fixt["S3"])
    fixt["m"].vector().set_local(f.vector().array())
    H_default = fixt["exch"].compute_field()

    supported_methods = list(Exchange._supported_methods)
    supported_methods.remove(fixt["exch"].method) # no need to compare default method with itself
    supported_methods.remove("project") # the project method for the exchange is too bad

    for method in supported_methods:
        exch = Exchange(fixt["A"], method=method)
        exch.setup(fixt["S3"], fixt["m"], fixt["Ms"])
        H = exch.compute_field()
        print "With method '{}', expecting H =\n{}\n, got H =\n{}.".format(
            method, H_default.reshape((3, -1)).mean(1), H.reshape((3, -1)).mean(1))

        rel_diff = np.abs((H - H_default) / H_default)
        assert np.nanmax(rel_diff) < REL_TOLERANCE
