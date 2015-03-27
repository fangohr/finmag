import time
import dolfin as df

mesh = df.UnitIntervalMesh(5)
V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
u = df.TrialFunction(V)
v = df.TestFunction(V)

f = df.Function(V)
g = df.Function(V)
fxg = df.Function(V)

f.assign(df.Constant((1, 0, 0)))
g.assign(df.Constant((0, 1, 0)))

# FINITE ELEMENT METHOD
# but dP instead of dx (both work)

a = df.dot(u, v) * df.dP
L = df.dot(df.cross(f, g), v) * df.dP

u = df.Function(V)
start = time.time()
for _ in xrange(1000):
    df.solve(a == L, u)
stop = time.time()
print "fem, delta = {} s.".format(stop - start)
print u.vector().array()

# LINEAR ALGEBRA FORMULATION

A = df.assemble(a)
b = df.Function(V)

u = df.Function(V)
start = time.time()
for _ in xrange(1000):
    df.assemble(L, tensor=b.vector())
    df.solve(A, u.vector(), b.vector())
stop = time.time()
print "la, delta = {} s.".format(stop - start)
print u.vector().array()

# JUST ASSEMBLING THE THING IN THE FIRST PLACE

u = df.Function(V)
start = time.time()
for _ in xrange(1000):
    df.assemble(L, tensor=u.vector())
stop = time.time()
print "just assembling, delta = {} s.".format(stop - start)
print u.vector().array()
