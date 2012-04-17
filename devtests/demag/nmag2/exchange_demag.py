import os
from dolfin import *
from finmag.sim.llg import LLG
from finmag.util.convert_mesh import convert_mesh
from scipy.integrate import ode
import pylab as p
import numpy as np
import progressbar as pb
import finmag.sim.helpers as h

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
mesh = Mesh(convert_mesh(MODULE_DIR + "/bar30_30_100.geo"))
#mesh.coordinates()[:] *= 1e-9

# Set up LLG
llg = LLG(mesh, mesh_units=1e-9)
llg.Ms = 0.86e6
llg.A = 13.0e-12
llg.alpha = 0.5
llg.set_m((1,0,1))
llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="weiwei")

# Set up time integrator
llg_wrap = lambda t, y: llg.solve_for(y, t)
r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
r.set_initial_value(llg.m, 0)

dt = 5.0e-12
T1 = 60*dt

fh = open(MODULE_DIR + "/averages.txt", "w")
xlist, ylist, zlist, tlist = [], [], [], []

bar = pb.ProgressBar(maxval=60, \
                widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
counter = 0
print "Time integration (this may take some time..)"
while r.successful() and r.t <= T1:
    bar.update(counter)
    counter += 1

    mx, my, mz = llg.m_average
    xlist.append(mx)
    ylist.append(my)
    zlist.append(mz)
    tlist.append(r.t)

    fh.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")
    r.integrate(r.t + dt)
fh.close()

# Add data points from nmag to plot
if os.path.isfile(MODULE_DIR + "/averages_ref.txt"):
    ref = np.array(h.read_float_data(MODULE_DIR + "/averages_ref.txt"))
    dt = ref[:,0] - np.array(tlist)
    assert np.max(dt) < 1e-15, "Compare timesteps."
    t = list(ref[:,0])*3
    y = list(ref[:,1]) + list(ref[:,2]) + list(ref[:,3])
    p.plot(t, y, 'o', mfc='w', label='nmag')

# Plot finmag data
p.plot(tlist, xlist, 'k', label='$\mathsf{m_x}$')
p.plot(tlist, ylist, 'r', label='$\mathsf{m_y}$')
p.plot(tlist, zlist, 'b', label='$\mathsf{m_z}$')
p.axis([0, T1, -0.2, 1.1])
p.title("Finmag vs Nmag")
p.legend(loc='center right')
p.savefig("exchange_demag.png")
