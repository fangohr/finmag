import sys
import dolfin as df
import numpy as np
import finmag.sim.helpers as h

from scipy.integrate import ode, odeint
from finmag.sim.llg import LLG

fh = open("data_M_dolfin.txt", "w")

mesh = df.Mesh("bar.xml")
mesh.coordinates()[:] = 1e-9 * mesh.coordinates() # from (implied) nm to m
llg = LLG(mesh)

llg.alpha = 0.1
llg.MS = 0.86e6 # A/m
llg.C = 1.3e-11 # J/m
#llg.H_app = (0.43e6, 0, 0) # A/m
llg.initial_M_expr(("2*x[0]/L - 1","2*x[1]/W - 1","1"), L=3e-8, H=1e-8, W=1e-8)
llg.M = h.for_dolfin(h.normalise(h.vectors(llg.M), llg.MS))
llg.setup(exchange_flag=True)

ts = np.arange(0, 1.5e-9, 1e-11)
ys = odeint(llg.solve_for, llg.M, ts, atol=10)

for i in range(len(ys)):
    t = ts[i]
    Mx, My, Mz = np.mean(h.components(ys[i]), axis=1)
    fh.write(str(t) + " " + str(Mx) + " " + str(My) + " " + str(Mz) + "\n")
fh.close()
