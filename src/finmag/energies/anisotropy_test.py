import pytest
import textwrap
import dolfin as df
import numpy as np
from finmag.energies import UniaxialAnisotropy
from finmag.util.consts import mu0

TOLERANCE = 1e-12


@pytest.fixture(scope = "module")
def fixt():
    """
    Create an UniaxialAnisotropy object that will be re-used during testing.

    """
    mesh = df.UnitCubeMesh(1, 1, 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    K1 = 1
    Ms = 1
    a = df.Constant((0, 0, 1))
    m = df.Function(S3)
    anis = UniaxialAnisotropy(K1, a)
    anis.setup(S3, m, Ms)
    return {"anis": anis, "m": m, "a": a, "S3": S3, "Ms": Ms, "K1": K1}


def test_interaction_accepts_name(fixt):
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    K1 = 1
    a = df.Constant((0, 0, 1))

    anis = UniaxialAnisotropy(K1, a, name='MyAnisotropy')
    assert hasattr(anis, 'name')


@pytest.mark.parametrize(("m", "expected_E"), [
    ((0, 0, 1), 0), ((0, 0, -1), 0), ((0, 1, 0), 1), ((-1, 0, 0), 1)])
def test_anisotropy_energy_simple_configurations(fixt, m, expected_E):
    """
    Test some parallel and orthogonal configurations of m and a.

    """
    fixt["m"].assign(df.Constant(m))
    E = fixt["anis"].compute_energy()

    print "With m = {}, expecting E = {}. Got E = {}.".format(m, expected_E, E)
    #assert abs(E - expected_E) < TOLERANCE
    assert np.allclose(E, expected_E, atol=1e-14, rtol=TOLERANCE)


def test_anisotropy_energy_analytical(fixt):
    """
    Compare one UniaxialAnisotropy energy with the corresponding analytical result.

    The magnetisation is m = (0, sqrt(1 - x^2), x) and the easy axis still
    a = (0, 0, 1). The squared dot product in the energy integral thus gives
    dot(a, m)^2 = x^2. Integrating x^2 gives (x^3)/3 and the analytical
    result with the constants we have chosen is 1 - 1/3 = 2/3.

    """
    f = df.interpolate(df.Expression(("0", "sqrt(1 - pow(x[0], 2))", "x[0]")), fixt["S3"])
    fixt["m"].vector().set_local(f.vector().array())
    E = fixt["anis"].compute_energy()
    expected_E = float(2)/3

    print "With m = (0, sqrt(1-x^2), x), expecting E = {}. Got E = {}.".format(expected_E, E)
    #assert abs(E - expected_E) < TOLERANCE
    assert np.allclose(E, expected_E, atol=1e-14, rtol=TOLERANCE)


def test_anisotropy_field(fixt):
    """
    Compute one anisotropy field by hand and compare with the UniaxialAnisotropy result.

    """
    TOLERANCE = 1e-14

    fixt["m"].assign(df.Constant((1/np.sqrt(2), 0, 1/np.sqrt(2))))
    H = fixt["anis"].compute_field()

    v = df.TestFunction(fixt["S3"])
    g_ani = df.Constant(fixt["K1"]/(mu0 * fixt["Ms"])) * (
            2 * df.dot(fixt["a"], fixt["m"]) * df.dot(fixt["a"], v)) * df.dx
    volume = df.assemble(df.dot(v, df.Constant((1, 1, 1))) * df.dx).array()
    dE_dm = df.assemble(g_ani).array() / volume

    print(textwrap.dedent("""
              With m = (1, 0, 1)/sqrt(2),
                  expecting: H = {},
                  got:       H = {}.
              """.format(H.reshape((3, -1)).mean(axis=1),
                         dE_dm.reshape((3, -1)).mean(axis=1))))
    assert np.allclose(H, dE_dm, atol=0, rtol=TOLERANCE)


def test_anisotropy_field_supported_methods(fixt):
    """
    Check that all supported methods give the same results as the default method.

    """
    TOLERANCE = 1e-13

    fixt["m"].assign(df.Constant((1/np.sqrt(2), 0, 1/np.sqrt(2))))
    H_default = fixt["anis"].compute_field()

    supported_methods = list(UniaxialAnisotropy._supported_methods)
    # No need to compare default method with itself.
    supported_methods.remove(fixt["anis"].method)

    for method in supported_methods:
        anis = UniaxialAnisotropy(fixt["K1"], fixt["a"], method=method)
        anis.setup(fixt["S3"], fixt["m"], fixt["Ms"])
        H = anis.compute_field()
        print(textwrap.dedent("""
                  With method '{}',
                      expecting: H = {},
                      got:       H = {}.
                  """.format(method,
                             H_default.reshape((3, -1)).mean(axis=1),
                             H.reshape((3, -1)).mean(axis=1))))
        assert np.allclose(H, H_default, atol=0, rtol=TOLERANCE)
