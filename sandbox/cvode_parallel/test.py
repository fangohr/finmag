import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import cvode2

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#mpirun -n 2 python test.py

"""
    Solve equation du/dt = sin(t) based on dolfin vector and sundials/cvode in parallel.
"""
class Test(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.V = df.FunctionSpace(mesh, 'CG', 1)
        zero = df.Expression('0.3')
        self.u = df.interpolate(zero, self.V)
        self.spin = self.u.vector().array()
        self.t = 0
        
        self.m_petsc = df.as_backend_type(self.u.vector()).vec()
        

    def set_up_solver(self, rtol=1e-8, atol=1e-8):
        
        self.ode = cvode2.CvodeSolver(self.sundials_rhs, 0, self.m_petsc, rtol, atol)

        #self.ode.test_petsc(self.m_petsc)
        
    def sundials_rhs(self, t, y, ydot):
        
        print 'rank=%d t=%g local=%d size=%d'%(rank, t, y.getLocalSize(), ydot.getSize())

        ydot.setArray(np.sin(t))
 
        return 0

    def run_until(self,t):
        
        if t <= self.t:
            return
        
        ode = self.ode
        
        flag = ode.run_until(t)
        
        if flag < 0:
            raise Exception("Run cython run_until failed!!!")
        
        self.u.vector().set_local(ode.y_np)
        self.spin = self.u.vector().array()
        #print self.spin[0]
        #self.spin[:] = ode.y[:]
    

def plot_m(ts, m):
    plt.plot(ts, m, ".", label="m", color='DarkGreen')
    plt.xlabel('Time')
    plt.ylabel('m')
    plt.savefig('m_rank_%d.pdf'%rank)


if __name__ == '__main__':
    nx = ny = 10
    mesh = df.UnitSquareMesh(nx, ny)
    sim = Test(mesh)
    sim.set_up_solver()
    
    ts = np.linspace(0, 5, 101)
    
    
    us = []
    
    file = df.File('u_time.pvd')
    
    for t in ts:
        sim.run_until(t)
        #file << sim.u
        us.append(sim.spin[0])
        
    plot_m(ts,us)
