from dolfin import *
from scipy.integrate import ode
import numpy as np
import pylab

set_log_level(21)


simplices = 5
L = 10e-9
mesh = Interval(simplices, 0, L)

# Functions
V = VectorFunctionSpace(mesh, "CG", 1, dim=3)
u = TrialFunction(V)
v = TestFunction(V)

# Parameters
alpha = 0.5
gamma = 2.211e5
c = 1e8
p = Constant(gamma/(1 + alpha**2))

# External field.
H = Expression(("0.0", "H0*sin(omega*t)", "0.0"), H0=1e5, omega=50*DOLFIN_PI/1e-9, t=0.0)

# Initial direction of the magnetic field.
M = Function(V)
M.assign(Constant((1, 1, 1)))
#Ms = Constant(1.0/(4.0e-7*np.pi))
Ms = Constant(1)

# Variational forms
a = inner(u, v)*dx
L = inner((-p*cross(M,H)
           -p*alpha/Ms*cross(M,cross(M,H))
           -c*(inner(M,M) - Ms**2)*M/Ms**2), v)*dx

dM = Function(V)

def f(t, y):
    M.vector()[:] = y
    solve(a==L, dM)
    return dM.vector().array()

# Time integration
y0 = M.vector().array()
t0 = 0
r = ode(f).set_integrator('vode', method='bdf', with_jacobian=False)
r.set_initial_value(y0, t0)

ps = 1e-12
t1 = 300*ps
dt = 0.1*ps
xlist = []

while r.successful() and r.t < t1-dt:
    r.integrate(r.t + dt)
    H.t = r.t
    # TODO: Find out what is supposed to be plotted
    xlist.append(sum(r.y[6:12])/8.5)
print r.y
pylab.plot(xlist)
pylab.show()
