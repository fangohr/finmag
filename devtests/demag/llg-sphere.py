import numpy as np
import dolfin as df
from finmag.util.convert_mesh import convert_mesh
from finmag.demag.solver_gcr import FemBemGCRSolver
import pylab 
import sys, os, commands

from finmag.sim.llg import LLG




maxh=3.
# Create geofile
geo = """
algebraic3d

solid main = sphere (0, 0, 0; 10)-maxh=%s ;

tlo main;""" % str(maxh)

absname = "sphere_maxh_%s" % str(maxh)
geofilename = os.path.join('.', absname)
geofile = geofilename + '.geo'
f = open(geofile, "w")
f.write(geo)
f.close()

# Finmag data
mesh = df.Mesh(convert_mesh(geofile))
print "Using mesh with %g vertices" % mesh.num_vertices()


from scipy.integrate import ode

llg=LLG(mesh)
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

llg.timings()


