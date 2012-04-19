import os, sys
from finmag.sim.llg import LLG
from scipy.integrate import ode
from finmag.util.timings import timings
from finmag.util.convert_mesh import convert_mesh
import pytest
import pylab as p
import numpy as np
import dolfin as df
import progressbar as pb
import finmag.sim.helpers as h
from finmag.sim.integrator import LLGIntegrator

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
METHOD = "weiwei"
METHOD = "FK"
TOL = 1e-2
#TOL = 5e-2

ref_data = np.array(h.read_float_data(MODULE_DIR + "/../../../examples/exchange_demag/averages_ref.txt"))
#mesh = df.Mesh(convert_mesh(MODULE_DIR + "/../../../examples/exchange_demag/bar30_30_100.geo"))
mesh = df.Box(0,0,0,30,30,100, 6,6,20)

# Set up LLG
llg = LLG(mesh, mesh_units=1e-9)
llg.Ms = 0.86e6
llg.A = 13.0e-12
llg.alpha = 0.5
llg.set_m((1,0,1))
llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method=METHOD)
integrator = LLGIntegrator(llg, llg.m, backend="scipy")

# Set up scipy.ode time integrator
llg_wrap = lambda t, y: llg.solve_for(y, t)
r = ode(llg_wrap)#.set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
r.set_initial_value(llg.m, 0)

dt = 5.0e-12
T1 = 60*dt

xlist, ylist, zlist, tlist = [], [], [], []
bar = pb.ProgressBar(maxval=60, \
                widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
counter = 0
print "Time integration (this may take some time..)"
max_diff = 0
while r.successful() and r.t <= T1:
    bar.update(counter)
    counter += 1
    mx, my, mz = llg.m_average

    computed = np.array([mx, my, mz])
    ref = ref_data[counter-1, 1:]
    diff = np.abs(ref - computed)

    if np.max(diff) > max_diff:
        max_diff = np.max(diff)
        #print "Aborting scipy.ode at time t=%g, with error=%s." % (r.t, np.max(diff))
        #sys.exit(1)
    r.integrate(r.t + dt)
print "scipy ode", max_diff

max_diff = 0
# Do the same with Dmitri's time integrator
for i in range(61):
    t = i*dt
    integrator.run_until(t)

    m = integrator.m.copy()
    m.shape = (3, -1)
    mx, my, mz = m
    mx, my, mz = np.average(mx), np.average(my), np.average(mz)
    computed = np.array([mx, my, mz])

    ref = ref_data[i, 1:]
    diff = np.abs(ref - computed)

    if np.max(diff) > max_diff:
        max_diff = np.max(diff)

print "Dmitri:", max_diff

