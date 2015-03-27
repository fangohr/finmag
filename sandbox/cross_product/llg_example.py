import dolfin as df
import time

mesh = df.UnitSquareMesh(100, 100)
print mesh
S1 = df.FunctionSpace(mesh, 'CG', 1)
S3 = df.VectorFunctionSpace(mesh, 'CG', 1, 3)

m = df.Function(S3)  # magnetisation
Heff = df.Function(S3)  # effective field
Ms = df.Function(S1)  # saturation magnetisation
alpha = df.Function(S1)  # damping
gamma = df.Constant(1)  # gyromagnetic ratio

m.assign(df.Constant((1, 0, 0)))
Heff.assign(df.Constant((0, 0, 1)))
alpha.assign(df.Constant(1))
Ms.assign(df.Constant(1))

dmdt_expression = -gamma/(1+alpha*alpha)*df.cross(m, Heff) - alpha*gamma/(1+alpha*alpha)*df.cross(m, df.cross(m, Heff))
dmdt_function = df.Function(S3)
dmdt = dmdt_function.vector()

start = time.time()
for i in xrange(10000):
    df.assemble(df.dot(dmdt_expression, df.TestFunction(S3)) * df.dP, tensor=dmdt)
    if i % 100 == 0:
        print i
stop = time.time()
print "delta = ", stop - start

# An expected result is unit z vector and the type is dolfin vector.
print dmdt.array()
print type(dmdt)
