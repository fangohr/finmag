import dolfin as df
from finmag.sim.llg import LLG
import numpy as np
from scipy.integrate import ode
import pylab

K1=520e3  #J/m^3
A=30e-12  #J/m
x0=252e-9 #m
Ms=1400e3 #A/m

def Mz_exact(x,x0=x0,A=A,Ms=Ms):
    """Analytical solution.
    
    """
    #return np.cos(np.pi/2 + np.arctan(np.sinh((x - 252e-9)/np.sqrt(30e-12/520.e3))))
    return Ms*np.cos(np.pi/2 + np.arctan(np.sinh((x - x0)/np.sqrt(A/K1))))

def M0(r):
    """Return initial magnetisation (vectorized)."""
    offset = 2e-9
    length = 500e-9
    x = r[:,0]
    relative_position = -2*(x - offset)/length + 1
    
    # The following two lines are the vectorized version of this:
    # mz = min(1.0, max(-1.0, relative_position))
    max1r = np.where(relative_position < -1.0, -1.0, relative_position)
    mz = np.where(max1r > 1.0, 1.0, max1r)
    return 0*mz, np.sqrt(1.0 - mz*mz), mz

simplices = 202
L = 504e-9
dim = 3
mesh = df.Interval(simplices, 0, L)
V = df.VectorFunctionSpace(mesh, "CG", 1, dim=dim)

m0 = df.Function(V)        
coor = mesh.coordinates()
n = len(m0.vector().array())

print "Double check that the length of the vectors are equal: %g and %g" \
        % (n, len(coor)*dim)
assert n == len(coor)*dim

# Setup LLG
llg = LLG(mesh)

# TODO: Find out how one are supposed to pin.
# llg.pins = [0,1,-2,-1] # This is not so good

llg.Ms = Ms                 # saturation magnetization
llg.C = A                   # exchange coupling
a = df.Constant((0, 0, 1)) # easy axis


# set initial magnetization
x, y, z = M0(coor)
m0.vector()[:] = np.array([x, y, z]).reshape(n)
llg.set_m0(np.array([x, y, z]).reshape(n))

llg.setup(exchange_flag=True)
llg.add_uniaxial_anisotropy(K1,a)
llg_wrap = lambda t, y: llg.solve_for(y, t)

# Time integration
t0 = 0; dt = 10e-12; t1 = 2e-09
r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
r.set_initial_value(llg.m, t0)

while r.successful() and r.t < t1-dt:
    r.integrate(r.t + dt)
    print "Integrating time: %g" % r.t

# Plot magnetisation in z-direction
#zstart = 2*n//3
#mz = llg.m[zstart:]*Ms
mz = []
x = np.linspace(0, L, simplices+1)
for xpos in x:
    mz.append(llg._m(xpos)[2])
mz = np.array(mz)*Ms
pylab.plot(x,mz,x,Mz_exact(x))
#pylab.axis([0, L, -1.1, 1.1])
pylab.legend(("Finmag", "Analytical"))
pylab.title("Domain wall example - Finmag vs analytical solution")
pylab.xlabel("Length")
pylab.ylabel("M.z")

try:
    import scipy.optimize
except ImportError:
    pass
else:
    popt,pcov = scipy.optimize.curve_fit(Mz_exact,x,mz,p0=(x0*1.1,A*1.1,Ms*1.1))
    print "popt=",popt
    
    fittedx0,fittedA,fittedMs=popt
    print "Error in fitted x0: %9.7f%%" % ((fittedx0-x0)/x0*100)
    print "Error in fitted Ms: %9.7f%%" % ((fittedMs-Ms)/Ms*100)
    print "Error in fitted A : %9.7f%%" % ((fittedA-A)/A*100)

diff = abs(mz - Mz_exact(x))
print max(diff)
pylab.show()
