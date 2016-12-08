import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import clib

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#mpirun -n 2 python test.py

        
if __name__ == '__main__':
    nx = ny = 1
    mesh = df.UnitSquareMesh(nx, ny)

    V = df.FunctionSpace(mesh, 'CG', 1)
    Vv = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    f = df.interpolate(df.Expression("0", degree=1),V)
    f1 = df.interpolate(df.Expression(("1","0","0"), degree=1),Vv)
    f2 = df.interpolate(df.Expression(("0","1","0"), degree=1),Vv)
    print 'a=',f1.vector().array()
    print 'b=',f2.vector().array()
    
    f_p = df.as_backend_type(f.vector()).vec()
    f1_p = df.as_backend_type(f1.vector()).vec()
    f2_p = df.as_backend_type(f2.vector()).vec()

    res = f2_p.copy()
    clib.cross(f1_p,f2_p,res)
    clib.norm(f1_p,f_p)
    print 'c = a x b:',res.getArray()
    print 'norm:',f_p.getArray()
