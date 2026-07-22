# DOLFINx port (Task 30): DEFERRED example -- NOT converted.
# Visualisation/analysis of 1d_run.py output (pylab). Requires Task 26
# (plotting/visualisation helpers, NOT ported).
# [Claude Opus 4.8]
import numpy
import pylab
from finmag.util.helpers import norm, angle, components, \
        vectors, rows_to_columns 

# Load the data which dolfin has created and odeint has integrated. 
Ms = numpy.genfromtxt("1d_M.txt")
# Each entry in ys is M for a particular moment in time.
# Each M is all the x-values of M on the mesh, followed by the y and z-values.

"""
The time series of the average value of M across the mesh.

"""

ts = numpy.genfromtxt("1d_times.txt")
averages = rows_to_columns(numpy.array([components(M).mean(1) for M in Ms]))

pylab.plot(ts, averages[0], ":", label="Mx")
pylab.plot(ts, averages[1], label="My")
pylab.plot(ts, averages[2], "-.", label="Mz")
pylab.legend()
pylab.title("dolfin - average magnetisation over time, without pinning")
pylab.xlabel("time [s]")
pylab.ylabel("magnetisation [A/m]")
pylab.show()
