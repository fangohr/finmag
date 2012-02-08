import dolfin as df
import numpy

interval = df.UnitInterval(3)
V = df.FunctionSpace(interval, "Lagrange", 1)

M    = df.Function(V)
#M.assign(df.Expression("3 * x[0]"))
#M.assign(df.Constant(1))
m = numpy.array([1., 1.1, 1., 1.])
M.vector()[:] = m

H_ex = df.TrialFunction(V)
v    = df.TestFunction(V)

a = df.inner(H_ex, v) * df.dx
L = - df.inner(df.grad(M), df.grad(v)) * df.dx

A = df.assemble(a)
b = df.assemble(L)

H_ex = df.Function(V)
df.solve(A, H_ex.vector(), b)

print "M"
print M.vector().array()
print "\nMatrix A(H_ex, v)"
print numpy.matrix(A.array())
print "\ninverse of A(H_ex, v)"
print numpy.matrix(A.array()).I
print "\nH_ex"
print H_ex.vector().array()
