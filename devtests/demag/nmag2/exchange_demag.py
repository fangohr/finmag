import os
from dolfin import *
from finmag.sim.llg import LLG
from finmag.util.convert_mesh import convert_mesh
from scipy.integrate import ode
import pylab as p

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

mesh = Mesh(convert_mesh(MODULE_DIR + "/bar30_30_100.geo"))

# Set up LLG
llg = LLG(mesh)
llg.Ms = 0.86e6
llg.A = 13.0e-12
llg.alpha = 0.5
llg.set_m((1,0,1))
llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="weiwei")

# Set up time integrator
llg_wrap = lambda t, y: llg.solve_for(y, t)
r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
r.set_initial_value(llg.m, 0)

dt = 1.0e-12
T1 = 50*dt

fh = open(MODULE_DIR + "/averages.txt", "w")
xlist, ylist, zlist, tlist = [], [], [], []
time_series = TimeSeries(MODULE_DIR + "/simulation_data/sim")

while r.successful() and r.t <= T1:
    mx, my, mz = llg.m_average
    xlist.append(mx)
    ylist.append(my)
    zlist.append(mz)
    tlist.append(r.t)

    fh.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")
    
    time_series.store(llg._m.vector(), r.t)
    print "integration time", r.t + dt
    r.integrate(r.t + dt)
    plot(llg._m)


fh.close()
p.plot(tlist, xlist, tlist, ylist, tlist, zlist)
p.show()
interactive()

