import numpy
import dolfin
from scipy.integrate import odeint

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
if False:
    llg.pins = [0, 10]

ts = numpy.linspace(0, 1e-9, 1e5)
ys, infodict = odeint(llg.solve_for, llg.M, ts, atol=10, full_output=True)

for i in range(len(ts)):
    M = vectors(ys[i])
    angles = numpy.array([angle(M[j], M[j+1]) for j in xrange(len(M)-1)])
    print "time ", ts[i], angles
    if abs(angles.max() - angles.min()) < TOLERANCE:
        break

print "System converged."
