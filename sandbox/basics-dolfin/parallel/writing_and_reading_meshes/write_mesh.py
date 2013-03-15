from dolfin import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print "This is process {}/{}".format(rank, size)

from common import mesh
F = HDF5File('meshfile_{:02d}.h5'.format(size), 'w')
F.write(mesh, 'mymesh')

print "Process {}/{} has written the mesh.".format(rank, size)
