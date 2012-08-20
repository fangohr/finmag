import dolfin as df
import numpy

interval = df.UnitInterval(3)
V = df.FunctionSpace(interval, "Lagrange", 1)

M    = df.Function(V)
#M.assign(df.Expression("3 * x[0]"))
#M.assign(df.Constant(1))
m = numpy.array([0., 1., 0., 0.])
M.vector()[:] = m

H_ex = df.TrialFunction(V)
v    = df.TestFunction(V)

a = df.inner(H_ex, v) * df.dx
A = df.assemble(a)

L = - df.inner(df.grad(M), df.grad(v)) * df.dx
b = df.assemble(L)

H_ex = df.Function(V)
df.solve(A, H_ex.vector(), b)

print "M"
print M.vector().array()
print "\nMatrix b(v)"
print numpy.matrix(b.array())
print "\ninverse of b(v)"
print numpy.matrix(b.array()).I
print "\nH_ex"
print H_ex.vector().array()
