import pytest
import dolfin as df
from finmag.util.helpers import fnormalise
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman, Demag

Ms = 8.6e5


def pytest_funcarg__fixt(request):
    fixt = request.cached_setup(setup=setup, scope="module")
    return fixt


def setup():
    """
    Create a cuboid mesh representing a magnetic material and two
    dolfin.Functions defined defined on this mesh:

        m  -- unit magnetisation (linearly varying across the sample)

        Ms_func -- constant function representing the saturation
                   magnetisation Ms


    *Returns*

    A triple (m_space, m, Ms_func), where m_space is the
    VectorFunctionSpace (of type "continuous Lagrange") on which the
    magnetisation m is defined and m, Ms_funct are as above.
    """
    mesh = df.BoxMesh(0, 0, 0, 10e-9, 10e-9, 10e-9, 5, 5, 5)

    m_space = df.VectorFunctionSpace(mesh, "CG", 1)
    m = df.interpolate(df.Expression(("1e-9", "x[0]/10", "0")), m_space)
    m.vector().set_local(fnormalise(m.vector().array()))

    Ms_space = df.FunctionSpace(mesh, "DG", 0)
    Ms_func = df.interpolate(df.Constant(Ms), Ms_space)

    return m_space, m, Ms_func


@pytest.mark.parametrize(("EnergyClass", "init_args"), [
        (Exchange, [1.3e-11]),
        (UniaxialAnisotropy, [1e5, (0, 0, 1)]),
        (Zeeman, [(0, 0, 1e6)]),
        (Demag, []),
        ])
def test_can_create_energy_object(fixt, EnergyClass, init_args):
    """
    Create two instances of the same energy class, once with a
    constant number as Ms and once with a constant function.
    """
    S3, m, Ms_func = fixt

    E1 = EnergyClass(*init_args)
    E1.setup(S3, m, Ms)

    E2 = EnergyClass(*init_args)
    E2.setup(S3, m, Ms_func)

    assert(abs(E1.compute_energy() - E2.compute_energy()) < 1e-12)
