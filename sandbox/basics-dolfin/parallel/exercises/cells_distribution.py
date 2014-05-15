#!/usr/bin/env python

"""
A short example to show how the nodes are distributed on processes
without the knowledge of the data details. To run this script,

    mpirun -n 8 python cells_distribution.py
    
"""

import dolfin as df
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def fun(x):
    return 1.0*rank/size

"""
A wrapper class that could be called by dolfin in parallel 
for a normal python function. In this way, we don't have to
know the details of the data.
"""
class HelperExpression(df.Expression):
    def __init__(self,value):
        super(HelperExpression, self).__init__()
        self.fun = value

    def eval(self, value, x):
        value[0] = self.fun(x)



if __name__=="__main__":

    mesh = df.RectangleMesh(0, 0, 20, 20, 100, 100)
    V = df.FunctionSpace(mesh, 'DG', 0)
    hexp = HelperExpression(fun)
    u = df.interpolate(hexp, V)
    
    file = df.File('color_map.pvd')
    file << u