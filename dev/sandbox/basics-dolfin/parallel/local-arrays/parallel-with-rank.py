

print "Hello World"
import dolfin as df
mpi_world = df.mpi_comm_world()
rank = df.MPI.rank(mpi_world)
size = df.MPI.size(mpi_world)
me_str = "{}/{}".format(rank, size)
print("I am " + me_str)
#df.parameters.reorder_dofs_serial = False

nx = ny = 1
mesh = df.RectangleMesh(0, 0, 1, 1, nx, ny)
V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
zero = df.Constant(list([0,0,1]))
zero_v = df.interpolate(zero, V)
print("{}: My vector is of shape {}.".format(me_str, zero_v.vector().array().shape))
#print zero_v.vector().array()
