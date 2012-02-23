import sys
import dolfin as df
import numpy as np
import finmag.sim.helpers as h

from scipy.integrate import ode
from finmag.sim.llg import LLG

fh = open("data_M_dolfin.txt", "w")

mesh = df.Mesh("bar.xml")
mesh.coordinates()[:] = 1e-9 * mesh.coordinates() # from (implied) nm to m
llg = LLG(mesh)

llg.alpha = 0.1
llg.Ms = 0.86e6 # A/m
llg.C = 1.3e-11 # J/m
#llg.H_app = (0.43e6, 0, 0) # A/m
llg.set_m0(("2*x[0]/L - 1","2*x[1]/W - 1","1"), L=3e-8, H=1e-8, W=1e-8)
llg.setup(exchange_flag=True)

llg_wrap = lambda t, y: llg.solve_for(y, t)
t0 = 0; dt = 1e-11; tmax = 1e-9 # s
r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-3)
r.set_initial_value(llg.m, t0)

while r.successful() and r.t <= tmax:
    Mx, My, Mz = np.mean(h.components(llg.M), axis=1)
    print r.t
    fh.write(str(r.t) + " " + str(Mx) + " " + str(My) + " " + str(Mz) + "\n")
    r.integrate(r.t + dt)

fh.close()
