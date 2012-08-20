from dolfin import *
from scipy.integrate import odeint
from numpy import linspace
from values import c, M0

set_log_level(21)

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
c = c()

# Applied field.
H = Constant((0, 1e5, 0))

# Initial direction of the magnetic field.
Ms, M0 = M0()
M = Function(V)
M.assign(M0)

# Variational forms
a = inner(u, v)*dx
L = inner((-p*cross(M,H)
           -p*alpha/Ms*cross(M,cross(M,H))
           -c*(inner(M,M) - Ms**2)*M/Ms**2), v)*dx

# Time derivative of the magnetic field.
dM = Function(V)
J = derivative(L, M)

counter = 0
def f(y, t):
    global counter
    counter += 1
    M.vector()[:] = y
    solve(a==L, dM)
    return dM.vector().array()

def j(y, t): 
    Jac = assemble(J)
    return Jac.array() 

ts = linspace(0, 1e-9, 100)
y0 = M.vector().array()
ys, infodict = odeint(f, y0, ts, Dfun=j, full_output=True)
print ys[-1]
print counter
