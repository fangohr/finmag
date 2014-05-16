import dolfin
import numpy
from scipy.integrate import odeint
from finmag.physics.llg import LLG
from finmag.energies import Exchange

"""
Compute the behaviour of a one-dimensional strip of magnetic material,
with exchange interaction.

"""

A = 1.3e-11
Ms = 8.6e5

length = 20e-9 # in meters
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)
S1 = dolfin.FunctionSpace(mesh, "Lagrange", 1)
S3 = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

llg = LLG(S1, S3)
llg.set_m((
        '2*x[0]/L - 1',
        'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))',
        '0'), L=length)
llg.pins = [0, 10]
exchange = Exchange(A)
llg.effective_field.add(exchange)

print "Solving problem..."

ts = numpy.linspace(0, 1e-9, 10)
ys, infodict = odeint(llg.solve_for, llg.m, ts, full_output=True)

print "Used", infodict["nfe"][-1], "function evaluations."
print "Saving data..."

numpy.savetxt("1d_times.txt", ts)
numpy.savetxt("1d_M.txt", ys)
numpy.savetxt("1d_coord.txt", mesh.coordinates().flatten())

print "Done."
