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
    L = - df.inner(df.grad(M), df.grad(v)) * df.dx
    A = df.assemble(a)
    b = df.assemble(L)
    H_ex = df.Function(V)
    df.solve(A, H_ex.vector(), b)
    return H_ex.vector().array()

for m in np.eye(5):
    print compute_b(m)

print "\n", compute_b(range(5))
