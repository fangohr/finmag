"""
Script to investigate the ordering process for Dolphin
meshes with periodic boundary conditions.

Want to assert that a local process contains the full
vector for each node and that they are not distributed.
"""

print "Hello World"
import dolfin as df
import numpy as np

#df.parameters.reorder_dofs_serial = False
df.parameters.reorder_dofs_serial = True

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

# what does the vertex to dof map look like?
d2v = df.dof_to_vertex_map(V)
print("dof_to_vertex_map: {}".format(d2v))
v2d = df.vertex_to_dof_map(V)
print("vertex_to_dof_map: {}".format(v2d))


# Store vector and assert that the full vector is present
my_vec = zero_v.vector().array()
my_vec_floor = np.floor(my_vec)
N = len(my_vec_floor)
for num in np.unique(my_vec_floor):
    assert((N - np.count_nonzero(my_vec_floor - num)) == 3)






""" 

2015/02/06 HF, MAB:

fangohr@osiris:~/hg/finmag/sandbox/basics-dolfin/parallel/local-arrays$ mpirun -n 1 python parallel-with-coord-and-map.py 
Hello World
My vector is of shape 30.
[ 0.1  0.2  0.3  1.1  1.2  1.3  2.1  2.2  2.3  3.1  3.2  3.3  4.1  4.2  4.3
  5.1  5.2  5.3  6.1  6.2  6.3  7.1  7.2  7.3  8.1  8.2  8.3  9.1  9.2  9.3]
dof_to_vertex_map: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29]
vertex_to_dof_map: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29]
fangohr@osiris:~/hg/finmag/sandbox/basics-dolfin/parallel/local-arrays$ mpirun -n 2 python parallel-with-coord-and-map.py 
Hello World
Hello World
Building mesh (dist 0a)
Number of global vertices: 10
Number of global cells: 9
Building mesh (dist 1a)
My vector is of shape 15.
My vector is of shape 15.
[ 5.1  5.2  5.3  6.1  6.2  6.3  7.1  7.2  7.3  8.1  8.2  8.3  9.1  9.2  9.3]
[ 0.1  0.2  0.3  1.1  1.2  1.3  2.1  2.2  2.3  3.1  3.2  3.3  4.1  4.2  4.3]
dof_to_vertex_map: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
dof_to_vertex_map: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
vertex_to_dof_map: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
vertex_to_dof_map: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14 -15 -14 -13]


Observation: dof_to_vertex_map seems to provide local indices (that's good).

We don't know what the additional negative numbers are (last line above): could be a periodic point, or
a hint that this node is on a different process. Will need to look further into this.

"""
