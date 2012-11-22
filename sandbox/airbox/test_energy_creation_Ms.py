import dolfin as df
from finmag.util.helpers import fnormalise
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman, Demag

Ms = 8.6e5
A = 1.3e-11

def pytest_funcarg__fixt(request):
    fixt = request.cached_setup(setup=setup, scope="module")
    return fixt

def setup():
    mesh = df.Box(0, 0, 0, 10e-9, 10e-9, 10e-9, 5, 5, 5)

    m_space = df.VectorFunctionSpace(mesh, "CG", 1)
    m = df.interpolate(df.Expression(("1e-9", "x[0]/10", "0")), m_space)
    m.vector().set_local(fnormalise(m.vector().array()))

    Ms_space = df.FunctionSpace(mesh, "DG", 0)
    Ms_func = df.interpolate(df.Constant(Ms), Ms_space)

    return m, m_space, Ms_func

def test_can_create_Exchange_object(fixt):
    with_Ms_number = Exchange(A)
    with_Ms_number.setup(fixt[1], fixt[0], Ms)

    with_Ms_func = Exchange(A)
    with_Ms_func.setup(fixt[1], fixt[0], fixt[2])

def test_can_create_UniaxialAnisotropy_object(fixt):
    with_Ms_number = UniaxialAnisotropy(1e5, (0, 0, 1))
    with_Ms_number.setup(fixt[1], fixt[0], Ms)

    with_Ms_func = UniaxialAnisotropy(1e5, (0, 0, 1))
    with_Ms_func.setup(fixt[1], fixt[0], fixt[2])

def test_can_create_Zeeman_object(fixt):
    with_Ms_number = Zeeman((0, 0, Ms))
    with_Ms_number.setup(fixt[1], fixt[0], Ms)

    with_Ms_func = Zeeman((0, 0, Ms))
    with_Ms_func.setup(fixt[1], fixt[0], fixt[2])

def test_can_create_Demag_object(fixt):
    with_Ms_number = Demag()
    with_Ms_number.setup(fixt[1], fixt[0], Ms)

    with_Ms_func = Demag()
    with_Ms_func.setup(fixt[1], fixt[0], fixt[2])




