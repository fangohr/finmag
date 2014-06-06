import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import cvode2
from finmag.physics.llg import LLG
from finmag.energies import Exchange, Zeeman

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
        self.S1 = df.FunctionSpace(mesh, 'CG', 1)
        self.S3 = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
        #zero = df.Expression('0.3')
        m_init = df.Constant([1, 0, 0])
        self.m = df.interpolate(m_init, self.S3)
        self.field = df.interpolate(m_init, self.S3)
        self.spin = self.m.vector().array()
        self.t = 0

        self.m_petsc = df.as_backend_type(self.m.vector()).vec()
        
        self.llg = LLG(self.S1, self.S3)
        self.llg.set_m(m_init)
        self.exchange = Exchange(13e-12)
        self.zeeman = Zeeman([0, 0, 1e5])
        self.llg.effective_field.add(self.exchange)
        self.llg.effective_field.add(self.zeeman)
        #self.llg.alpha = 0.1

    def set_up_solver(self, rtol=1e-8, atol=1e-8):
        
        self.ode = cvode2.CvodeSolver(self.sundials_rhs, 0, self.m_petsc, rtol, atol)

        #self.ode.test_petsc(self.m_petsc)
        
    def sundials_rhs(self, t, y, ydot):
        y_np = y.getArray()

        #self.llg._m.vector().set_local(y_np)
        ydot_np = self.llg.solve_for(y_np, t)
        ydot.setArray(ydot_np)
        
        #print 'rank=%d t=%g local=%d size=%d'%(rank, t, y.getLocalSize(), ydot.getSize())
        #print ydot.getArray()[0]

        #ydot.setArray(np.sin(t+np.pi/4*rank))
        sim.field.vector().set_local(ydot_np)
        return 0

    def run_until(self,t):
        
        if t <= self.t:
            ode = self.ode
            print ode.y_np[0]
            return
        
        ode = self.ode
        print ode.y_np[0]
        
        flag = ode.run_until(t)
        
        if flag < 0:
            raise Exception("Run cython run_until failed!!!")
        
        self.m.vector().set_local(ode.y_np)
        
        self.spin = self.m.vector().array()
        #print self.spin[0]
        #self.spin[:] = ode.y[:]
    

def plot_m(ts, m):
    plt.plot(ts, m, "-.", label="m", color='DarkGreen')
    plt.xlabel('Time')
    plt.ylabel('m')
    plt.savefig('m_rank_%d.pdf'%rank)


if __name__ == '__main__':
    nx = ny = 10
    mesh = df.UnitSquareMesh(nx, ny)
    sim = Test(mesh)
    sim.set_up_solver()
    
    ts = np.linspace(0, 2e-9, 101)
    
    
    us = []
    
    file = df.File('u_time.pvd')
    
    
    for t in ts:
        sim.run_until(t)
        #sim.field.vector().set_local(sim.llg.effective_field.H_eff)
        file << sim.field
        us.append(sim.spin[0])
        
    plot_m(ts,us)
