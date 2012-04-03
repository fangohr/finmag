import numpy as np
import dolfin as df
import pylab 
import sys, os, commands
import GCRruntime as Grt
from finmag.sim.llg import LLG
from finmag.demag.demag_solver import Demag
from finmag.util.convert_mesh import convert_mesh
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.problems.prob_fembem_testcases import MagSphere30

class DemagXtra(Demag):
    #Demag class with more detailed timings
    def __init__(self, V, m, Ms, method="GCR"):
        super(DemagXtra,self).__init__(V, m, Ms, method="GCR")    
        #Change the Demag Solver        
        self.solver = Grt.GCRtimings(self.problem)
            
class LLGXtra(LLG):
    #llg class with more detailed demag timings
    def setup(self, use_exchange=True, use_dmi=False, use_demag=False,
              exchange_method="box-matrix-petsc",
              dmi_method="box-matrix-petsc",
              demag_method="GCR"):
        super(LLGXtra,self).setup(use_exchange=use_exchange, use_dmi=use_dmi, use_demag=use_demag,
              exchange_method="box-matrix-petsc",
              dmi_method="box-matrix-petsc",
              demag_method="GCR")
        #Change the Demag solver
        self.demag = DemagXtra(self.V, self._m, self.Ms, method=demag_method)
        #Add an extra timing object for the method
        
##Note maxh=3.0
mesh = MagSphere30().mesh
print "Using mesh with %g vertices" % mesh.num_vertices()

from scipy.integrate import ode

#Add the extra timings
llg=LLGXtra(mesh)
llg.set_m0(df.Constant((1, 0, 0)))

tfinal = 0.3*1e-9
dt = 0.001e-9

llg.setup(use_demag=True)
#llg.setup(use_demag=False)
rhswrap = lambda t,y: llg.solve_for(y,t)
r = ode(rhswrap).set_integrator('vode', method='bdf', 
                                with_jacobian=False, max_step=dt/2.,
                                nsteps=1)#enforce only one step per call

#to gather data for later analysis
mlist = []
tlist = []
hext = []

y0 = llg.m
t0 = 0
r.set_initial_value(y0, t0)

steps = 0
while r.t < tfinal-dt:
    steps +=1
    r.integrate(r.t + dt)
    print "Integrating time: %g" % r.t
    mlist.append(llg.m_average) #not used here, just interested in timings
    tlist.append(r.t)           #not used here, just interested in timings

    if steps==3:
        print "Done 3 steps, exiting now"
        break

    #only plotting and data analysis from here on

llg.timings(30)
print Grt.qtime.report_str(10)


