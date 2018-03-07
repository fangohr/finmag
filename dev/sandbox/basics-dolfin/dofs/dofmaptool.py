from dolfin import *
from fenicstools import DofMapPlotter
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, 'CG', 1)
if rank == 0:
    dmp = DofMapPlotter(V)
    print("""Pressing C, E, v with mouse hovering over cells 0 and 1 and d with mouse over cell 40 and 41 will light up cell and edge indices for the entire mesh and vertex and dof indices in cells 0, 1 and 40, 41 respectively. Thus we see that first mesh vertex is located at [0, 0] while the first dof is located at [0, 1]. The assignement u.vector[0] thus modifies the function's value at [0, 1].""")
    dmp.plot()
    dmp.show()
