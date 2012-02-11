from dolfin import *
from scipy.integrate import odeint
from numpy import linspace

set_log_level(21)

counter = 0
m = 1e-5
mesh = Box(0,m,0,m,0,m,1,1,1)

# Functions
V = VectorFunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Parameters
alpha = 0.5
gamma = 2.211e5
p = Constant(gamma/(1 + alpha**2))
c = Constant(1e10)

# Applied field.
H = Constant((0, 1e5, 0))

# Initial direction of the magnetic field.
Ms = 8e5
M0 = Constant((Ms, 0, 0))
M = Function(V)
M.assign(M0)

# Variational forms
a = inner(u, v)*dx
L = inner((-p*cross(M,H)
           -p*alpha/Ms*cross(M,cross(M,H))
           -c*(inner(M,M) - Ms**2)*M/Ms**2), v)*dx

# Time derivative of the magnetic field.
dM = Function(V)

def f(y, t):
    global counter
    counter += 1
    M.vector()[:] = y
    solve(a==L, dM)
    return dM.vector().array()

ts = linspace(0, 1e-9, 100)
y0 = M.vector().array()
ys, infodict = odeint(f, y0, ts, Dfun=None, full_output=True)
print ys[-1]
print counter
