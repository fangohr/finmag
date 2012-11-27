import pytest
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
    x = 1; xn = 1;
    mesh = df.Box(0, 0, 0, x, x, x, xn, xn, xn)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    K1 = 1
    Ms = 1
    a = df.Constant((0, 0, 1))
    m = df.Function(S3)
    anis = UniaxialAnisotropy(K1, a)
    anis.setup(S3, m, Ms)
    return {"anis": anis, "m": m, "a": a, "S3": S3, "Ms": Ms, "K1": K1}

@pytest.mark.parametrize(("m", "expected_E"), [
    ((0, 0, 1), 0), ((0, 0, -1), 0), ((0, 1, 0), 1), ((-1, 0, 0), 1)])
def test_anisotropy_energy(fixt, m, expected_E):
    """
    Test some parallel and orthogonal configurations of m and a.

    """
    fixt["m"].assign(df.Constant(m))
    E = fixt["anis"].compute_energy()

    print "With m = {}, expecting E = {}. Got E = {}.".format(m, expected_E, E)
    assert abs(E - expected_E) < TOLERANCE

def test_anisotropy_field(fixt):
    """
    Compute one anisotropy field by hand and compare with the UniaxialAnisotropy result.

    """
    TOLERANCE = 1e-9

    fixt["m"].assign(df.Constant((1/np.sqrt(2), 0, 1/np.sqrt(2))))
    H = fixt["anis"].compute_field()

    v = df.TestFunction(fixt["S3"])
    g_ani = df.Constant(fixt["K1"]/(mu0 * fixt["Ms"])) * (
            2 * df.dot(fixt["a"], fixt["m"]) * df.dot(fixt["a"], v)) * df.dx
    volume = df.assemble(df.dot(v, df.Constant((1, 1, 1))) * df.dx).array()
    dE_dm = df.assemble(g_ani).array() / volume

    print "With m = (1, 0, 1)/sqrt(2),\n\texpecting H = {},\n\tgot H = {}.".format(
            H.reshape((3, -1)).mean(1), dE_dm.reshape((3, -1)).mean(1))
    assert np.max(np.abs(H - dE_dm)) < TOLERANCE
