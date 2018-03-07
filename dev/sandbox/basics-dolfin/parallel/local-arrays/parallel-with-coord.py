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


"""
Observations: 6 Feb 2015 HF:

fangohr@osiris:~/hg/finmag/sandbox/basics-dolfin/parallel/local-arrays$ dir
parallel.py	    parallel.py-n2.out	parallel-with-coord.py	run_parallel.sh
parallel.py-n1.out  parallel.py-n3.out	parallel-with-rank.py
fangohr@osiris:~/hg/finmag/sandbox/basics-dolfin/parallel/local-arrays$ cat parallel.py-n1.out 
My vector is of shape 30.
[ 0.1  1.1  2.1  3.1  4.1  5.1  6.1  7.1  8.1  9.1  0.2  1.2  2.2  3.2  4.2
  5.2  6.2  7.2  8.2  9.2  0.3  1.3  2.3  3.3  4.3  5.3  6.3  7.3  8.3  9.3]
fangohr@osiris:~/hg/finmag/sandbox/basics-dolfin/parallel/local-arrays$ cat parallel.py-n2.out 
My vector is of shape 15.
My vector is of shape 15.
[ 0.1  0.2  0.3  1.1  1.2  1.3  2.1  2.2  2.3  3.1  3.2  3.3  4.1  4.2  4.3]
[ 5.1  5.2  5.3  6.1  6.2  6.3  7.1  7.2  7.3  8.1  8.2  8.3  9.1  9.2  9.3]
fangohr@osiris:~/hg/finmag/sandbox/basics-dolfin/parallel/local-arrays$ cat parallel.py-n3.out 
My vector is of shape 9.
My vector is of shape 12.
My vector is of shape 9.
[ 3.1  3.2  3.3  4.1  4.2  4.3  5.1  5.2  5.3  6.1  6.2  6.3]
[ 0.1  0.2  0.3  1.1  1.2  1.3  2.1  2.2  2.3]
[ 7.1  7.2  7.3  8.1  8.2  8.3  9.1  9.2  9.3]
fangohr@osiris:~/hg/finmag/sandbox/basics-dolfin/parallel/local-arrays$ 

Interestingly, this uses the 'old' order xxxxx, yyyyy, zzzzz for the serial run, but
switches to xyz, xyz, xyz, xyz, xyz  when run with mpi -n N where N >= 2 .

In other words: the df.parameters.reorder_dofs_serial = False 
is ignored for parallel runs.


"""
