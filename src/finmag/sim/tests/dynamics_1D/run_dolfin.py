import sys
import numpy
import dolfin
import sys

from scipy.integrate import odeint, ode

from finmag.sim.llg import LLG
from finmag.sim.helpers import vectors,angle

TOLERANCE = 2e-9

# define the mesh
length = 20e-9 #m
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

# initial configuration of the magnetisation
M0_x = 'MS * (2*x[0]/L - 1)'
M0_y = 'sqrt(MS*MS - MS*MS*(2*x[0]/L - 1)*(2*x[0]/L - 1))'
M0_z = '0'

llg = LLG(mesh)
llg.initial_M_expr((M0_x, M0_y, M0_z), L=length, MS=llg.MS)
llg.setup(exchange_flag=True)
print llg.exchange.compute_field()
llg_wrap = lambda t, y: llg.solve_for(y, t) # for ode
#llg.pins = [0, 10]

#sys.exit()

if True: # odeint, this works
    ts = numpy.linspace(0, 1e-13, 1e3)
    ys, infodict = odeint(llg.solve_for, llg.M, ts, atol=10, full_output=True)

    for i in range(len(ts)):
        M = vectors(ys[i])
        angles = numpy.array([angle(M[j], M[j+1]) for j in xrange(len(M)-1)])
        print "time ", ts[i], angles
        if abs(angles.max() - angles.min()) < TOLERANCE:
            break

    print "System converged."
else: # ode, this doesn't
    tolerances_to_try = [0] + [10**e for e in range(-10,10)]
    # having 0 in either atol or rtol leads to pure relative/absolute local error control.
    # compare fortran code http://www.netlib.org/ode/vode.f
    
    for rtol in tolerances_to_try:
        for atol in tolerances_to_try:
            print "rtol:{0}, atol:{1}.".format(rtol, atol)

            t0 = 0; dt = 1e-12; tmax = 1e-9 # s
            llg.reset()
            r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=rtol, atol=atol)
            r.set_initial_value(llg.M, t0)

            while r.successful() and r.t <= tmax:
                print "time: ", r.t
                if r.t > 0:
                    print "This is it!"
                    sys.exit()
                r.integrate(r.t + dt)
            del(r)

