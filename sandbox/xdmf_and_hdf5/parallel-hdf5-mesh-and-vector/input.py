from dolfin import *

mesh2 = Mesh()
f2 = HDF5File(mesh2.mpi_comm(), 'u.h5', 'r')  

# The 3rd parameter in df.HDF5File.read is use_partition_from_file.
# When dolfin runs in parallel the mesh is divided/partitioned and
# each process has a one partition. When a mesh is saved in parallel
# the details of how the mesh is partitioned is also saved. If the
# data is then read in again this data is then available, but is 
# naturally only relevant if the same number of processes are being
# used to read as were used to save the data. 
f2.read(mesh2, 'mesh', False)

print("Mesh we have read: {}".format(mesh2.coordinates().shape))
V2 = FunctionSpace(mesh2, 'CG', 1)
u2 = Function(V2)
f2.read(u2, 'u')
print "vector we have read: {}".format(u2.vector().array().shape)
del f2

