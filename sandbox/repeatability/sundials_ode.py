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
    
    def create_integrator(self, reltol=1e-6, abstol=1e-6, nsteps=10000):

        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.x0)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)
        
        self.integrator = integrator
        self.t = 0
    
    
    def sundials_rhs(self, t, y, ydot):

        self.ode_count+=1
                
        #The following line is very important!!!
        ydot[:] = 0
        ydot[:] = self.sim(t, y)
        
        return 0
    
    def advance_time(self, t):
        
        if t > self.t:
            self.integrator.advance_time(t, self.x)
            self.t = t
            
    
    def print_info(self):
        print 'sim t=%0.15g'%self.t
        print 'x=',self.x
        print 'rhs=%d'%self.ode_count


    
