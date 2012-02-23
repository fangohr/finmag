import sys
import numpy
import dolfin
import sys

from scipy.integrate import odeint, ode

from finmag.sim.llg import LLG
from finmag.sim.helpers import vectors,angle

#
# TODO: test with and without pinning, build comparison with
#       dynamics of nmag and then remove the file test_exchange_static
#

TOLERANCE = 1e-7

# define the mesh
length = 20e-9 #m
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

# initial configuration of the magnetisation
m0_x = '2*x[0]/L - 1'
m0_y = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'
m0_z = '0'

def test_exchange_with_odeint():
    llg = LLG(mesh)
    llg.set_m0((m0_x, m0_y, m0_z), L=length)
    llg.setup(exchange_flag=True)
    #llg.pins = [0, 10]

    ts = numpy.linspace(0, 1e-9, 100)
    ys, infodict = odeint(llg.solve_for, llg.m, ts, full_output=True)

    m = vectors(ys[-1])
    angles = numpy.array([angle(m[j], m[j+1]) for j in xrange(len(m)-1)])
    assert abs(angles.max() - angles.min()) < TOLERANCE

def test_exchange_with_ode():
    llg = LLG(mesh)
    llg.set_m0((m0_x, m0_y, m0_z), L=length)
    llg.setup(exchange_flag=True)
    #llg.pins = [0, 10]

    llg_wrap = lambda t, y: llg.solve_for(y, t) # for ode
    t0 = 0; dt = 1e-12; tmax = 1e-9 # s
    r = ode(llg_wrap).set_integrator("vode", method="bdf")
    r.set_initial_value(llg.m, t0)

    while r.successful() and r.t <= tmax:
        r.integrate(r.t + dt)
    m = vectors(llg.m)
    angles = numpy.array([angle(m[j], m[j+1]) for j in xrange(len(m)-1)])
    assert abs(angles.max() - angles.min()) < TOLERANCE
