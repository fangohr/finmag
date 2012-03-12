from dolfin import *
from llg2 import LLG2 as LLG
import numpy as np
from scipy.integrate import ode
import pylab

def Mz_exact(x):
    """Analytical solution."""
    return 1400e3 * np.cos(np.pi/2 + np.arctan(np.sinh((x - 252e-9)/np.sqrt(30e-12/520e3))))

def M0(r):
    """Return initial magnetisation."""
    offset = 2e-9
    length = 500e-9
    x = r[0]
    
    rel = -2*(x - offset)/length + 1
    mz = min(1.0, max(-1.0, rel))
    return 0.0, np.sqrt(1.0 - mz*mz), mz

simplices = 1000
L = 504e-9
dim = 3
mesh = Interval(simplices, 0, L)
V = VectorFunctionSpace(mesh, "CG", 1, dim=dim)

m0 = Function(V)
coor = mesh.coordinates()
n = len(m0.vector().array())

print "Double check that the length of the vectors are equal: %g and %g" \
        % (n, len(coor))
assert n == len(coor)*dim

xstart = 0
ystart = n//3
zstart = 2*n//3

# FIXME: Make this vectorized
for i in range(len(coor)):
    x, y, z = M0(coor[i])
    m0.vector()[xstart + i] = x
    m0.vector()[ystart + i] = y
    m0.vector()[zstart + i] = z

# Setup LLG
llg = LLG(mesh)
llg.Ms = 1400e3

# TODO: Find out how one are supposed to pin.
# llg.pins = [0,1,-2,-1] # This is not so good

llg.c = 30e12
llg.set_m0(m0)
llg.setup(exchange_flag=True, anisotropy_flag=True)
llg_wrap = lambda t, y: llg.solve_for(y, t)

# Time integration
t0 = 0; dt = 0.05e-12; t1 = 10e-12
r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
r.set_initial_value(llg.m, t0)

while r.successful() and r.t <= t1:
    r.integrate(r.t + dt)
    print "Integrating time: %g" % r.t

# Plot magnetisation in z-direction
mz = llg.m[zstart:]
x = np.linspace(0, L, simplices+1)
# Comment out analytical solution for now, as it has overflow problems.
#pylab.plot(x,mz,x,Mz_exact(x))
pylab.plot(x, mz)
pylab.show()
