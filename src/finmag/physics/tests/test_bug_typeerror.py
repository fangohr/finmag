import numpy as np
import dolfin as df
from finmag.physics.equation import Equation

def test_sundials_serial_jtimes_typeerror():
    mesh = df.UnitCubeMesh(2, 2, 2)
    V = df.FunctionSpace(mesh, "CG", 1)
    m = df.Function(V)
    H = df.Function(V)
    dmdt = df.Function(V)
    eq = Equation(m.vector(), H.vector(), dmdt.vector())

    a = np.zeros(10)
    b = np.zeros(10)
    eq.test_np_array(a)
