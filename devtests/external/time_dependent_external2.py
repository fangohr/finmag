from dolfin import *
from scipy.integrate import ode
import numpy as np
import pylab
from finmag.sim.llg import LLG

set_log_level(21)


simplices = 5
L = 10e-9
mesh = Interval(simplices, 0, L)

# External field.

llg=LLG(mesh)

llg.set_m0(Constant((1, 0, 0)))

H = Expression(("0.0", "H0*sin(omega*t)", "0.0"), H0=1e5, omega=50*DOLFIN_PI/1e-9, t=0.0)

def update_H_ext(llg):
    print "update_H_ext being called"
    llg._H_app.t = llg.t

llg._pre_rhs_callables.append(update_H_ext)
llg.setup()

rhswrap = lambda t,y: llg.solve_for(y,t)
r = ode(rhswrap).set_integrator('vode', method='bdf', with_jacobian=False)

y0 = llg.m
t0 = 0

r.set_initial_value(y0, t0)

ps = 1e-12
t1 = 300*ps
dt = 5*ps

mlist = []
tlist = []

while r.successful() and r.t < t1-dt:
    r.integrate(r.t + dt)
    print "Integrating time: %g" % r.t
    mlist.append(llg.m_average)
    tlist.append(r.t)

mx = [tmp[0] for tmp in mlist]
print r.y
pylab.plot(tlist,mx)
pylab.show()
