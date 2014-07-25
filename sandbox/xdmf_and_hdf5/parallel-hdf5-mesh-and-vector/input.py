from dolfin import *

mesh2 = Mesh()
f2 = HDF5File(mesh2.mpi_comm(), 'u.h5', 'r')  
f2.read(mesh2, 'mesh', True)
print("Mesh we have read: {}".format(mesh2.coordinates().shape))
V2 = FunctionSpace(mesh2, 'CG', 1)
u2 = Function(V2)
f2.read(u2, 'u')
print "vector we have read: {}".format(u2.vector().array().shape)
del f2

