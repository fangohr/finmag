#!/usr/bin/env python

# This is a slightly adapted example from the FEniCS Q&A forum [1].
# It creates a 2D vector field 'v' on a 2D mesh (e.g. representing
# velocity of a flow) and then extracts these values out again and
# displays them in the form
#
#    (x_i, y_i) -->  (vx_i, vy_i)
#
# It also plots the field values using matplotlib and dolfin's own
# plotting functionality.
#
# This example is a good illustration of how to use the dofmap to
# access function data on a mesh. It is instructive to run this
# example in parallel, e.g.:
#
#    mpirun -n 4 python foobar.py
#
# This will show that each process only knows about the coordinates
# of the part of the mesh that it owns. The plotting will also only
# plot the vector field on the corresponding part of the mesh.
#
# Note that in parallel the dofmap and mesh on each process also
# use 'ghost nodes'. These are mesh nodes which do not belong to
# the current process but are neighbours of nodes that do. These
# are dealt with in the 'try ... except IndexError' statements
# below.
#
# (Max, 15.5.2014)
#
# [1] http://fenicsproject.org/qa/1460/numpy-arrays-from-fenics-data

import matplotlib
matplotlib.use('wx')
from dolfin import *
import numpy as np
import pylab as pl

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#mesh = RectangleMesh(0, 0, 20, 50, 4, 10)
mesh = UnitSquareMesh(10, 20)
V = VectorFunctionSpace(mesh, "CG", 1)
u = Function(V)

# Vertex based data
vertex_vector_values = np.zeros(mesh.num_vertices()*2)
vertex_vector_values[::2] = mesh.coordinates().sum(1)
vertex_vector_values[1::2] = 2-mesh.coordinates().sum(1)
dof_to_vertex_map = dof_to_vertex_map(V)

u.vector().set_local(vertex_vector_values[dof_to_vertex_map])

####### NOW GO THE OTHER WAY #####

arr = u.vector().array()
coor = mesh.coordinates()
vtd = vertex_to_dof_map(V)

values = list()
for i, dum in enumerate(coor):
    try:
        values.append([arr[vtd[2*i]],arr[vtd[2*i+1]]])
    except IndexError:
        print("[Process {}/{}] IndexError for i={}, dum={}".format(rank, size, i, dum))
values = np.array(values)

x = list()
y = list()
vx = list()
vy = list()
for i, dum in enumerate(coor):
    try:
        print '(%f,%f) -> (%f,%f)' %(coor[i][0], coor[i][1], values[i][0], values[i][1])
        x.append(coor[i][0])
        y.append(coor[i][1])
        vx.append(values[i][0])
        vy.append(values[i][1])
    except IndexError:
        print("[Process {}/{}] IndexError for i={}, dum={}".format(rank, size, i, dum))

pl.quiver(x,y,vx,vy)
pl.axis([0, 1.3, 0, 1.3])
pl.show()
plot(u, axes=True, interactive=True)
