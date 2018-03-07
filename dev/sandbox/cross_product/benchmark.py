"""
An example code that computes the cross product of two dolfin functions.
This should be replaced in the parts of the code where dmdt is computed.
"""
import dolfin as df
import time

mesh = df.IntervalMesh(1000, 0, 1)
S3 = df.VectorFunctionSpace(mesh, 'CG', 1, 3)

a = df.Function(S3)
b = df.Function(S3)
a.assign(df.Constant((1, 0, 0)))  # unit x vector
b.assign(df.Constant((0, 1, 0)))  # unit y vector

alpha = 01
gamma = 2.11e5
m = df.Function(S3)
m.assign(df.Constant((0, 0, 1)))
Heff = df.Function(S3)
Heff.assign(df.Constant((0, 0.3, 0.2)))
dmdt = df.Function(S3)

start = time.time()
for i in range(1000):
    L = df.dot(-gamma/(1+alpha*alpha)*df.cross(m, Heff) - alpha*gamma/(1+alpha*alpha)*df.cross(m, df.cross(m, Heff)), df.TestFunction(S3)) * df.dP
    df.assemble(L, tensor=dmdt.vector())
stop = time.time()
print 'Time:', stop-start

