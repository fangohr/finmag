import dolfin as df
from finmag.util.helpers import fnormalise
from finmag.energies import Exchange

Ms = 8.6e5
A = 1.3e-11

def test_can_create_Exchange_object():
    mesh = df.Box(0, 0, 0, 10e-9, 10e-9, 10e-9, 5, 5, 5)
    m_space = df.VectorFunctionSpace(mesh, "CG", 1)
    Ms_space = df.FunctionSpace(mesh, "DG", 0)

    m = df.interpolate(df.Expression(("1e-9", "x[0]/10", "0")), m_space)
    m.vector().set_local(fnormalise(m.vector().array()))

    df.plot(m, interactive=True)
    Ms_func = df.interpolate(df.Constant(Ms), Ms_space)

    exchange_Ms_number = Exchange(A)
    exchange_Ms_number.setup(m_space, m, Ms)

    exchange_Ms_func = Exchange(A)
    exchange_Ms_func.setup(m_space, m, Ms_func)


