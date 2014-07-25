#!/usr/bin/env python

"""
A short example to show how the nodes are distributed on processes
without the knowledge of the data details. To run this script,

    mpirun -n 8 python cells_distribution.py
    
"""

import dolfin as df

mpi_world = df.mpi_comm_world()
rank = df.MPI.rank(mpi_world)
size = df.MPI.size(mpi_world)

def fun(x):
    return rank 


## David, Becky, Hans 25 July 14: the next few lines are potentially
## useful, but not needed here.
## """
## A wrapper class that could be called by dolfin in parallel 
## for a normal python function. In this way, we don't have to
## know the details of the data.
## """
## class HelperExpression(df.Expression):
##     def __init__(self,value):
##         super(HelperExpression, self).__init__()
##         self.fun = value
## 
##     def eval(self, value, x):
##         value[0] = self.fun(x)
## 


if __name__=="__main__":

    mesh = df.RectangleMesh(0, 0, 20, 20, 100, 100)
    V = df.FunctionSpace(mesh, 'DG', 0)
    #hexp = HelperExpression(fun)       # if we want to use the HelperExpression
    hexp = df.Expression("alpha", alpha=rank)
    u = df.interpolate(hexp, V)
    
    file = df.File('color_map.pvd')
    file << u
