import dolfin as df
import numpy as np

from scipy.integrate import ode
from finmag.sim.llg import LLG

mesh = df.Mesh("bar.xml")
llg = LLG(mesh)

llg.alpha = 0.1
llg.MS = 0.86e6 # A/m
llg.C = 1.3e-11 # J/m
llg.H_app = (0.43e6, 0, 0) # A/m
llg.initial_M_expr(("MS*sqrt(2)", "MS*sqrt(2)", "0"), MS=llg.MS)
llg.setup(exchange_flag=False)

t0 = 0; dt = 1e-11; tmax = 1e-9 # s
llg_wrap = lambda t, y: llg.solve_for(y, t)
r = ode(llg_wrap).set_integrator("vode", method="bdf", with_jacobian=False)
r.set_initial_value(llg.M, t0)
print llg.average_M()
while r.successful() and r.t <= tmax:
    r.integrate(r.t+dt)
    print r.t, llg.average_M(),"\n\n"

