#!/usr/bin/env python

import dolfin as df

# Define a non-constant function with unique entries.
mesh = df.IntervalMesh(100, 0, 10)
V = df.FunctionSpace(mesh, 'CG', 1)
f = df.interpolate(df.Expression('x[0]'), V, degree=1)

rank = df.cpp.common.MPI.rank(df.mpi_comm_world())
size = df.cpp.common.MPI.size(df.mpi_comm_world())

print("[{}/{}] Local vector: {}".format(rank, size, f.vector().array()))
max_f = f.vector().max();   # get the local max (might be different on each node)
#max_f = df.cpp.common.MPI.max(df.mpi_comm_world(), max_f);  # get the global max across all nodes

print("[{}/{}] max_f={}".format(rank, size, max_f))
