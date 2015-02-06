"""
Script to investigate the ordering process for Dolphin
meshes with periodic boundary conditions.

Want to assert that a local process contains the full
vector for each node and that they are not distributed.
"""

print "Hello World"
import dolfin as df
import numpy as np

# Explicit - do not reorder
df.parameters.reorder_dofs_serial = False

# Generate mesh and boundary conditions
nx = ny = 10
mesh = df.IntervalMesh(9,0,9)
class PeriodicBoundary(df.SubDomain):

    def inside(self, x, on_boundary):
        return bool(x[0] < 1+df.DOLFIN_EPS and x[0] > 1-df.DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 4.0

pbc = PeriodicBoundary()

# Vector space is populated with numbers representing coordinates
V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3, constrained_domain=pbc)
expression = df.Expression(['x[0]+0.1', 'x[0]+0.2', 'x[0]+0.3'])
zero_v = df.interpolate(expression, V)

# What does the vector look like?
print("My vector is of shape %s." % zero_v.vector().array().shape)
print zero_v.vector().array()

# Store vector and assert that the full vector is present
my_vec = zero_v.vector().array()
my_vec_floor = np.floor(my_vec)
N = len(my_vec_floor)
for num in np.unique(my_vec_floor):
    assert((N - np.count_nonzero(my_vec_floor - num)) == 3)

