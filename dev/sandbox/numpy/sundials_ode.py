import os
import numpy as np
from finmag.native import sundials


def call_back(t, y):
    
    return y**2-y**3
    
class Test_Sundials(object):
    
    def __init__(self, call_back, x0):
        
        self.sim = call_back
        self.x0 = x0
        self.x = x0.copy()
        self.ode_count = 0
        
        self.create_integrator()
    
    def create_integrator(self, reltol=1e-2, abstol=1e-2, nsteps=10000):

        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.x0)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)
        
        self.integrator = integrator
    
    
    def sundials_rhs(self, t, y, ydot):

        self.ode_count+=1
                
        #The following line is very important!!!
        ydot[:] = 0
        ydot[:] = self.sim(t, y)
        
        return 0
    
    def run_step(self, steps, min_dt=1e-10):
        
        self.t = 0
        self.ts=[]
        self.ys=[]

        for i in range(steps):
            cvode_dt = self.integrator.get_current_step()
            if cvode_dt < min_dt:
                cvode_dt = min_dt
            self.t += cvode_dt
            self.integrator.advance_time(self.t, self.x)
            self.ts.append(self.t)
            self.ys.append(self.x[0])
            #print i
            
    
    
    def print_info(self):
        print 'sim t=%0.15g'%self.t
        print 'x=',self.x
        print 'rhs=%d'%self.ode_count


    
