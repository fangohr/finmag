# Example motivated by dolfin mailing list (questions 221862): 
# http://answers.launchpad.net/dolfin/+question/221862 -- see last reply from Hake

# The entry on the list is not actually very
# informative, but the snippet below seems to show the key idea.
#
# I am gathering this here, to push forward the whole dofmap question
# that was introduced with dolfin 1.1 (HF, Feb 2013)

# Addendum: The main (whole?) point of the dofmap seems to be
# for bookkeeping purposes when dolfin is running in parallel.
# Thus we're now also printing the process number and cell id
# in each process. It is instructive to look at the output of
# a command like this:
#
#    mpirun -n 6 python example1.py | grep "cell #" | sort
#
# So it seems that cell id's are local to each process, and
# that V.dofmap().cell_dofs() gives a mapping from the local
# degrees of freedom of the cell to the global degrees of freedom.
#
# (Max, 15.3.2013)


import dolfin as df
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("2d Example")

mesh = df.UnitSquareMesh(4, 4)

V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
x = V.dofmap()

for cell in df.cells(mesh):
    print "Process {}/{}, cell #{}:  {}, {}".format(rank, size, cell.index(), cell, V.dofmap().cell_dofs(cell.index()))

#For a 1d example, uncomment the code below and run
# mpirun -np 2 python dofmap_cell_dofs.py  | grep "topological" | sort
#
#if rank == 0:
#    print("1d Example")
#
#mesh = df.IntervalMesh(20, -1, 1)
#V = df.FunctionSpace(mesh, 'CG', 1)
#x = V.dofmap()
#for cell in df.cells(mesh):
#    print("Process {}/{}, cell #{}:  {}, {}".format(rank, size, cell.index(), cell, V.dofmap().cell_dofs(cell.index())))
#
#
