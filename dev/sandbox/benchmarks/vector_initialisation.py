#!/usr/bin/python

import numpy as np
import dolfin as df
import timeit

df.parameters.reorder_dofs_serial = False # for dolfin 1.1.0

# Define a rectangular mesh and a vector-function space on it

n1 = 100
n2 = 40
n3 = 25

print "Initialising mesh and function space..."
mesh = df.UnitCubeMesh(n1, n2, n3)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

print "Number of mesh nodes: {}".format(mesh.num_vertices())


# Define two different methods to create a constant vector-valued
# function on the space:

def constant_function_v1(v, S3):
    val = np.empty((S3.mesh().num_vertices(), 3))
    val[:] = v # we're using broadcasting here
    fun = df.Function(S3)
    fun.vector()[:] = val.transpose().reshape((-1,))
    return fun

def constant_function_v2(v, S3):
    val = df.Constant(v)
    fun = df.interpolate(val, S3)
    return fun


# Now create the 'same' function using each of these methods and time how long it takes:

v = [2,4,3]  # some arbitrary vector

f1 = constant_function_v1(v, S3)
f2 = constant_function_v2(v, S3)

print "Performing timing measurements (this may take a while for larger meshes) ...\n"
t1 = min(timeit.Timer('constant_function_v1(v, S3)', 'from __main__ import constant_function_v1, v, S3').repeat(3, number=10))
t2 = min(timeit.Timer('constant_function_v2(v, S3)', 'from __main__ import constant_function_v2, v, S3').repeat(3, number=10))

print "Method 1 took {:g} seconds  (using a numpy array to set the function vector directly) (best of 3 runs)".format(t1)
print "Method 2 took {:g} seconds  (using df.interpolate) (best of 3 runs)".format(t2)

# Just for safety, check that the two functions have the same values on the nodes
print "\nPerforming safety check that the two functions are actually the same on all nodes ...",
assert(all(f1.vector() == f2.vector()))
print "passed!"
