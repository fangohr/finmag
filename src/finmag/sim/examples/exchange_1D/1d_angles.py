import numpy
import pylab
from finmag.sim.helpers import vectors, norm, angle

# Load the data which dolfin has created and odeint has integrated. 
Ms = numpy.genfromtxt("1d_M.txt")
# Each entry in ys is M for a particular moment in time.
# Each M is all the x-values of M on the mesh, followed by the y and z-values.

"""
Norm of M at each node, and the angles between M at
the nodes for the first and last moment of the simulation.

"""

M0 = vectors(Ms[0])
M1 = vectors(Ms[-1])

norms0 = [norm(M) for M in M0]
norms1 = [norm(M) for M in M1]

angles0 = [angle(M0[i], M0[i+1]) for i in xrange(len(M0)-1)]
angles1 = [angle(M1[i], M1[i+1]) for i in xrange(len(M1)-1)]

print "Initial configuration."
print M0, "\n", norms0
print "Final configuration."
print M1, "\n", norms1
print "Angles in the initial configuration."
print angles0
print "Angles at the end of the simulation."
print angles1

pylab.plot(angles0, label="beginning")
pylab.plot(angles1, label="end")
pylab.legend()
pylab.show()
