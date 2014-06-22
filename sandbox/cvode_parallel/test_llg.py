import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import cvode2
import llg_petsc
from finmag.physics.llg import LLG
from finmag.energies import Exchange, Zeeman, Demag

df.parameters.reorder_dofs_serial = True
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
        m_init = df.Constant([1, 1, 1.0])
        self.m = df.interpolate(m_init, self.S3)
        self.field = df.interpolate(m_init, self.S3)
        self.spin = self.m.vector().array()
        self.t = 0
        
        #It seems it's not safe to specify rank in df.interpolate???
        self._alpha = df.interpolate(df.Constant("0.001"), self.S1)
        self._alpha.vector().set_local(self._alpha.vector().array())
        print 'dolfin',self._alpha.vector().array()
        
        self.llg = LLG(self.S1, self.S3, unit_length=1e-9)
        #normalise doesn't work due to the wrong order
        self.llg.set_m(m_init, normalise=True)
        
        parameters = {
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-10,
            'maximum_iterations': int(1e5)
            }
    
        demag = Demag()
    
        demag.parameters['phi_1'] = parameters
        demag.parameters['phi_2'] = parameters
        
        self.exchange = Exchange(13e-12)
        self.zeeman = Zeeman([0, 0, 1e5])
        
        self.llg.effective_field.add(self.exchange)
        #self.llg.effective_field.add(demag)
        self.llg.effective_field.add(self.zeeman)
        
        self.m_petsc = df.as_backend_type(self.llg._m.vector()).vec()
        self.h_petsc = df.as_backend_type(self.field.vector()).vec()
        self.alpha_petsc = df.as_backend_type(self._alpha.vector()).vec()
        

    def set_up_solver(self, rtol=1e-8, atol=1e-8):
        
        self.ode = cvode2.CvodeSolver(self.sundials_rhs, 0, self.m_petsc, rtol, atol)

        
    def sundials_rhs(self, t, y, ydot):
        
        self.llg.effective_field.update(t)
        self.field.vector().set_local(self.llg.effective_field.H_eff)

        llg_petsc.compute_dm_dt(y,
                                self.h_petsc,
                                ydot,
                                self.alpha_petsc,
                                self.llg.gamma,
                                self.llg.do_precession,
                                self.llg.c)

        return 0

    def run_until(self,t):
        
        if t <= self.t:
            ode = self.ode
            return
        
        ode = self.ode
        
        flag = ode.run_until(t)
        
        if flag < 0:
            raise Exception("Run cython run_until failed!!!")
        
        self.m.vector().set_local(ode.y_np)
        
        self.spin = self.m.vector().array()


        #print self.spin[0]
        #self.spin[:] = ode.y[:]
    

def plot_m(ts, m):
    plt.plot(ts, m, "-", label="m", color='DarkGreen')
    plt.xlabel('Time')
    plt.ylabel('energy')
    plt.savefig('m_rank_%d.pdf'%rank)



if __name__ == '__main__':
    mesh = df.RectangleMesh(0,0,100.0,20.0,25,5)
    mesh = df.BoxMesh(0, 0, 0, 30, 30, 100, 6, 6, 20)
    #mesh = df.IntervalMesh(10,0,10)
    sim = Test(mesh)
    sim.set_up_solver()
    
    ts = np.linspace(0, 2e-10, 101)
        
    energy = []
    
    #file = df.File('u_time.pvd')
    file = df.HDF5File(sim.m.vector().mpi_comm(),'test.h5','w')
    
    for t in ts:
        sim.run_until(t)
        print t, sim.spin[0]
        #sim.field.vector().set_local(sim.llg.effective_field.H_eff)
        file.write(sim.m,'/m_%g'%t)
        energy.append(sim.llg.effective_field.total_energy())
        
    plot_m(ts,energy)
