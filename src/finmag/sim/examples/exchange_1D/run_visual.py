""" Attempt to visualise some dynimacs using python's visual module.

Visualisation probably okay, but some convergence problem in the time 
integration means this does not look smooth at all, and is very slow.

ODEINT gives up after 500 iterations (before the solution is correct
- note the magnitude of the vectors changing from vector to vector;
this should not happen).

HF 16 Feb 2012

"""

import dolfin
import numpy
from scipy.integrate import odeint
from finmag.sim.llg import LLG

"""
Compute the behaviour of a one-dimensional strip of magnetic material,
with exchange interaction.

"""

length = 40e-9 # in meters
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

llg = LLG(mesh)
llg.alpha=0.1
llg.H_app=(0,0,llg.MS)
#llg.initial_M_expr((
#        'MS * (2*x[0]/L - 1)',
#        'sqrt(MS*MS - MS*MS*(2*x[0]/L - 1)*(2*x[0]/L - 1))',
#        '0'), L=length*10, MS=llg.MS)
llg.initial_M_expr((
        'MS',
        '0',
        '0'), MS=llg.MS)
llg.setup()
#llg.pins = [0, 10]


#import sys
#sys.exit(0)


print "Solving problem..."

import visual
y = llg.M[:]
y.shape=(3,len(llg.M)/3)
arrows = []
coordinates = (mesh.coordinates()/length-0.5)*len(mesh.coordinates())*0.4
for i in range(y.shape[1]):
    pos = list(coordinates[i])
    thisM = y[:,i]/llg.MS
    while len(pos) < 3: #visual python needs 3d vector
        pos.append(0.0) 
    
    arrows.append(visual.arrow(pos=pos,axis=tuple(thisM)))

ts = numpy.linspace(0, 1e-10, 200)
tol = 100
for i in range(len(ts)-1):
    ys,infodict = odeint(llg.solve_for, llg.M, [ts[i],ts[i+1]], full_output=True,printmessg=True,rtol=tol,atol=tol)
    y = ys[-1,:]/llg.MS
    y.shape=(3,len(llg.M)/3)
    for j in range(y.shape[1]):
        arrows[j].axis=tuple(y[:,j])
    print("i=%d/%d, t=%s" % (i,len(ts),ts[i])),
    print "nfe=%d, nje=%d" % (infodict['nfe'],infodict['nje'])

print "Done"
