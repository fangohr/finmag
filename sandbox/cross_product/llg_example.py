import dolfin as df
import time

mesh = df.UnitSquareMesh(200, 200)
print mesh
S1 = df.FunctionSpace(mesh, 'CG', 1)
S3 = df.VectorFunctionSpace(mesh, 'CG', 1, 3)
u = df.TrialFunction(S3)
v = df.TestFunction(S3)

m = df.Function(S3)  # magnetisation
Heff = df.Function(S3)  # effective field
Ms = df.Function(S1)  # saturation magnetisation
alpha = df.Function(S1)  # damping
gamma = df.Constant(1)  # gyromagnetic ratio

m.assign(df.Constant((1, 0, 0)))
Heff.assign(df.Constant((0, 0, 1)))
alpha.assign(df.Constant(1))
Ms.assign(df.Constant(1))

# just assembling it
LLG = -gamma/(1+alpha*alpha)*df.cross(m, Heff) - alpha*gamma/(1+alpha*alpha)*df.cross(m, df.cross(m, Heff))
L = df.dot(LLG, df.TestFunction(S3)) * df.dP

dmdt = df.Function(S3)
start = time.time()
for i in xrange(1000):
    df.assemble(L, tensor=dmdt.vector())
stop = time.time()
print "delta = ", stop - start
print dmdt.vector().array()

# more linear algebra, same problem... still need to assemble the cross product
# we're doing even more work than before
a = df.dot(u, v) * df.dP
A = df.assemble(a)
b = df.Function(S3)

dmdt = df.Function(S3)
start = time.time()
for i in xrange(1000):
    df.assemble(L, tensor=b.vector())  # this is what should go out of the loop
    df.solve(A, dmdt.vector(), b.vector())  # some variation of this could stay in
stop = time.time()
print "delta = ", stop - start
print dmdt.vector().array()
