import time
import math
import dolfin as df
import numpy as np

"""
The goal is to compute the values of f(r, t) = sin(x) * sin(t) on a mesh for
different points in time. The cost of updating f a couple of times is measured.

"""

L = math.pi / 2; n = 1;
mesh = df.Box(0, 0, 0, L, L, L, n, n, n)
#mesh = df.BoxMesh(0, 0, 0, L, L, L, n, n, n)
ts = np.linspace(0, math.pi / 2, 100)

# dolfin expression code

S = df.FunctionSpace(mesh, "CG", 1)
expr = df.Expression("sin(x[0]) * sin(t)", t = 0.0)

start = time.time()
for t in ts:
    expr.t = t
    f_dolfin = df.interpolate(expr, S)
t_dolfin = time.time() - start
print "Time needed for dolfin expression: {:.2g} s.".format(t_dolfin)

# explicit loop code

f_loop = np.empty(mesh.num_vertices())

start = time.time()
for t in ts:
    for i, (x, y, z) in enumerate(mesh.coordinates()):
        f_loop[i] = math.sin(x) * math.sin(t)
t_loop = time.time() - start
print "Time needed for loop: {:.2g} s.".format(t_loop)

# compute the ratio

ratio = t_dolfin / t_loop
if ratio >= 1:
    print "Looping over numpy array is {:.2g} times faster than interpolating dolfin expression.".format(ratio)
else:
    print "Interpolating the dolfin expression is {:.2g} times faster than looping over numpy array.".format(ratio)

# check correctness

print "\nMesh coordinates:\n", mesh.coordinates()
print "Function was sin(x) * sin(t)."
print "Dolfin vector at t=pi/2:\n\t", f_dolfin.vector().array()
print "Numpy vector at t=pi/2:\n\t", f_loop

max_diff = np.max(np.abs(f_dolfin.vector().array() - f_loop))
print "Maximum difference is d = {:.2g}.".format(max_diff)
assert max_diff < 1e-14
