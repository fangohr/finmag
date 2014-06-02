import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import cvode2

"""
    Solve equation du/dt = sin(t) based on dolfin vector and sundials/cvode in parallel.
"""
class Test(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.V = df.FunctionSpace(mesh, 'CG', 1)
        zero = df.Expression('0')
        self.u = df.interpolate(zero, self.V)
        self.spin = self.u.vector().array()
        self.t = 0
                

    def set_up_solver(self, rtol=1e-8, atol=1e-8):
        
        self.ode = cvode2.CvodeSolver(self.spin,
                                      rtol,atol,
                                      self.sundials_rhs)
        
        self.ode.set_initial_value(self.spin, self.t)
        

    def sundials_rhs(self, t, y, ydot):
        
        ydot[:] = np.sin(t)
 
        return 0

    def run_until(self,t):
        
        if t <= self.t:
            return
        
        ode = self.ode
        
        flag = ode.run_until(t)
        
        if flag < 0:
            raise Exception("Run cython run_until failed!!!")
        
        self.spin[:] = ode.y[:]
    

def plot_m(ts, m):
    plt.plot(ts, m, ".", label="m", color='DarkGreen')
    plt.xlabel('Time')
    plt.ylabel('m')
    plt.savefig('m_t.pdf')


if __name__ == '__main__':
    nx = ny = 10
    mesh = df.UnitSquareMesh(nx, ny)
    sim = Test(mesh)
    sim.set_up_solver()
    
    ts = np.linspace(0, 5, 101)
    us = []
    for t in ts:
        sim.run_until(t)
        us.append(sim.spin[0])
        
    plot_m(ts,us)
        
    


