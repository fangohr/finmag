import dolfin as df
import numpy as np
import sys

#M.assign(df.Expression("3 * x[0]"))
#M.assign(df.Constant(1))
def compute_b(m):
    interval = df.UnitInterval(len(m)-1)
    V = df.FunctionSpace(interval, "Lagrange", 1)
    M = df.Function(V)
    M.vector()[:] = np.array(m)
    H_ex = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.inner(H_ex, v) * df.dx
    U = df.inner(df.grad(M), df.grad(M)) * df.dx
    H_ex_form = df.derivative(U, M, v)
    V = df.assemble(v*df.dx).array()
    dU_dM = df.assemble(H_ex_form).array()
    return dU_dM/V

#for m in np.eye(5):
#    print compute_b(m)

print "\n", compute_b([0,1,2,3,4])
