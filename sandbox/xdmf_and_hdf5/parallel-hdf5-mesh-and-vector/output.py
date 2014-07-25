from dolfin import *

mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, 'CG', 1)
u = interpolate(Expression('x[0]'), V)

f = HDF5File(mesh.mpi_comm(), 'u.h5', 'w')    # works

f.write(mesh, 'mesh')
print "mesh we have written: {}".format(mesh.coordinates().shape)
f.write(u, 'u')
print "vector we have written: {}".format(u.vector().array().shape)
del f


