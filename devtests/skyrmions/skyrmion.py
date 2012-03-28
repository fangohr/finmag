""" Attempt to visualise some dynimacs using python's visual module.

Visualisation probably okay, but some convergence problem in the time 
integration means this does not look smooth at all, and is very slow.

"""

import dolfin
import dolfin as df
import numpy
from scipy.integrate import odeint
from finmag.sim.llg import LLG

"""
Compute the behaviour of a one-dimensional strip of magnetic material,
with exchange interaction.

"""

nm=1e-9
xdim = 30
ydim = 30
zdim = 3
mesh = dolfin.Box(0,0,0,xdim*nm,ydim*nm, zdim*nm, xdim/3, ydim/3, zdim/3)

llg = LLG(mesh)
llg.alpha=1.0
llg.H_app=(0,0,0)

llg.C =1.3e-11
llg.D = 4e-3

#llg.set_m0((
 #       'MS * (2*x[0]/L - 1)',
  #      'sqrt(MS*MS - MS*MS*(2*x[0]/L - 1)*(2*x[0]/L - 1))',
   #     '0'), L=length, MS=llg.Ms)
llg.set_m0((
        'MS',
        '0',
        '0'), MS=llg.Ms)
llg.setup(use_dmi=True,use_exchange=True)
llg.pins = []
print "point 0:",mesh.coordinates()[0]

print "Solving problem..."

ts = numpy.linspace(0, 1e-9, 50)
tol = 1e-4
for i in range(len(ts)-1):
    print "step=%4d, time=%12.6g " % (i,ts[i]),
    #ys,infodict = odeint(llg.solve_for, llg.m, [ts[i],ts[i+1]], full_output=True,printmessg=True,rtol=tol,atol=tol)
    ys,infodict = odeint(llg.solve_for, llg.m, [ts[i],ts[i+1]], full_output=True,printmessg=False)
    print "NFE=%4d, NJE=%4d" % (infodict['nfe'],infodict['nje'])

    df.plot(llg._m)
df.interactive()
print "Done"
