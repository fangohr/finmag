import dolfin
import numpy
from scipy.integrate import odeint
from finmag.sim.llg import LLG

"""
Compute the behaviour of a one-dimensional strip of magnetic material,
with exchange interaction.

"""

length = 20e-9 # in meters
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

llg = LLG(mesh)
llg.set_m0((
        '2*x[0]/L - 1',
        'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))',
        '0'), L=length)
llg.setup()
llg.pins = [0, 10]

print "Solving problem..."

ts = numpy.linspace(0, 1e-9, 10)
ys, infodict = odeint(llg.solve_for, llg.m, ts, full_output=True)

print "Used", infodict["nfe"][-1], "function evaluations."
print "Saving data..."

numpy.savetxt("1d_times.txt", ts)
numpy.savetxt("1d_M.txt", ys)
numpy.savetxt("1d_coord.txt", llg.mesh.coordinates().flatten())

print "Done."
