from dolfin import *

mesh = UnitCubeMesh(30,30,30)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
F = Function(V)

b = v*dx
L = assemble(b)

print \
"""
In this test case, F is a dolfin.Function, and L is computed as

\tL = assemble(v*dx),

where v is a dolfin.TestFunction. Say we want to divide F by L.
It we simply write

\tA1 = F.vector().array()/L.

This has proved to be much slower than

\tA2 = F.vector().array()/L.array().

Timings that computes A1 and A2 on a 30 x 30 x 30 unit cube
20 times each, show the issue:
"""

tic()
for i in range(20):
    A1 = F.vector().array()/L
print "A1: %.4f sec." % toc()

tic()
for i in range(20):
    A2 = F.vector().array()/L.array()
print "A2: %.4f sec." % toc()
