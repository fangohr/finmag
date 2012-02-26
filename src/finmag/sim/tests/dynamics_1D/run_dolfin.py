import dolfin
import numpy
import finmag.sim.helpers as h
from finmag.sim.llg import LLG
from scipy.integrate import odeint, ode

TOLERANCE = 1e-7

# define the mesh
length = 20e-9 # m
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

# initial configuration of the magnetisation
m0_x = '2*x[0]/L - 1'
m0_y = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'
m0_z = '0'

llg = LLG(mesh)
llg.Ms = 0.86e6
llg.C = 1.3e-11
llg.alpha = 0.2
llg.set_m0((m0_x, m0_y, m0_z), L=length)
llg.setup(exchange_flag=True)
llg.H_app = (0, 0, llg.Ms/2)
llg.pins = [0, 10]

# ode takes the parameters in the order t, y whereas odeint and we use y, t.
llg_wrap = lambda t, y: llg.solve_for(y, t)

t0 = 0; t1 = 3.10e-9; dt = 5e-12; # s
r = ode(llg_wrap).set_integrator("vode", method="bdf")
r.set_initial_value(llg.m, t0)

log1 = open("averages.txt", "w")
log2 = open("third_node.txt", "w")
while r.successful() and r.t <= t1:
    mx, my, mz = llg.m_average
    log1.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

    mx, my, mz = h.components(llg.m)
    m2x, m2y, m2z = mx[2], my[2], mz[2]
    log2.write(str(r.t) + " " + str(m2x) + " " + str(m2y) + " " + str(m2z) + "\n")

    r.integrate(r.t + dt)
log1.close()
log2.close()
