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

length = 50e-9 # in meters
simplexes = 10
#mesh = dolfin.Interval(simplexes, 0, length)
#mesh = dolfin.Rectangle(0,0,length,length, simplexes, simplexes)
mesh = dolfin.Box(0,0,0,length,length, length, simplexes, simplexes, simplexes)


llg = LLG(mesh)
llg.alpha=0.1
llg.H_app=(0,0,500)
#llg.set_m0((
 #       'MS * (2*x[0]/L - 1)',
  #      'sqrt(MS*MS - MS*MS*(2*x[0]/L - 1)*(2*x[0]/L - 1))',
   #     '0'), L=length, MS=llg.Ms)
llg.set_m0((
        'MS',
        '0',
        '0'), MS=llg.Ms)
llg.setup(use_dmi=True,exchange_flag=True)
#llg.setup(use_dmi=False)
#llg.pins = [0, 10]
llg.pins = [0]
print "point 0:",mesh.coordinates()[0]

print "Solving problem..."

y = llg.m[:]
y.shape=(3,len(llg.m)/3)
ts = numpy.linspace(0, 5e-10, 1000)
tol = 1e-4
for i in range(len(ts)-1):
    print i
    ys,infodict = odeint(llg.solve_for, llg.m, [ts[i],ts[i+1]], full_output=True,printmessg=True,rtol=tol,atol=tol)
    df.plot(llg._m)
df.interactive()
print "Done"
