import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import finmag.native.cvode_petsc as cvode

#from mpi4py import MPI

#comm = MPI.COMM_WORLD
#rank1 = comm.Get_rank()
#size1 = comm.Get_size()

mpi_world = df.mpi_comm_world()
rank = df.MPI.rank(mpi_world)
size = df.MPI.size(mpi_world)

#mpirun -n 2 python test.py

"""
    Solve equation du/dt = sin(t) based on dolfin vector and sundials/cvode in parallel.
"""
class Test(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.V = df.FunctionSpace(mesh, 'CG', 1)
        zero = df.Expression('0')
        self.u = df.interpolate(zero, self.V)
        self.spin = self.u.vector().get_local()
        self.spin[:] = 0.1*rank
        self.u.vector().set_local(self.spin)
        #print rank, len(self.spin),self.spin
        #print rank, len(self.u.vector().array()), self.u.vector().array()
        self.t = 0
        
        self.m_petsc = df.as_backend_type(self.u.vector()).vec()
        

    def set_up_solver(self, rtol=1e-8, atol=1e-8):
        
        self.ode = cvode.CvodeSolver(self.sundials_rhs, 0, self.m_petsc, rtol, atol)

        #self.ode.test_petsc(self.m_petsc)
        
    def sundials_rhs(self, t, y, ydot):
        
        print 'rank=%d t=%g local=%d size=%d'%(rank, t, y.getLocalSize(), ydot.getSize())
        
        ydot.setArray(np.cos((rank+1)/2.0*t))
 
        return 0

    def run_until(self,t):
        
        if t <= self.t:
            return
        
        ode = self.ode
        
        flag = ode.run_until(t)
        
        if flag < 0:
            raise Exception("Run cython run_until failed!!!")
        
        self.u.vector().set_local(ode.y_np)
        self.spin = self.u.vector().get_local()
        #print self.spin[0]
        #self.spin[:] = ode.y[:]
    

def plot_m(ts, m):
    fig = plt.figure()
    plt.plot(ts, m, ".", label="sim", color='DarkGreen')
    m2 = 2.0/(rank+1.0)*np.sin((rank+1)/2.0*ts)+0.1*rank
    plt.plot(ts, m2, label="exact", color='red')
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.legend()
    fig.savefig('m_rank_%d.pdf'%rank)


if __name__ == '__main__':
    nx = ny = 10
    mesh = df.UnitSquareMesh(nx, ny)
    sim = Test(mesh)
    sim.set_up_solver()
    
    ts = np.linspace(0, 5, 101)
    
    us = []
    
    file = df.File('u_time.pvd')
    file << sim.u
    
    for t in ts:
        sim.run_until(t)
        #file << sim.u
        us.append(sim.spin[0])
        
    plot_m(ts,us)
