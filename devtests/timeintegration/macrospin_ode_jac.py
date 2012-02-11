from dolfin import *
from scipy.integrate import ode
from numpy import linspace

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

counter = 0

def f(t, y):
    global counter
    counter += 1
    M.vector()[:] = y
    solve(a==L, dM)
    return dM.vector().array()

def j(t, y):
    J = derivative(L, M)
    J = assemble(J)
    print J.array()[0,0]
    return J.array() 

y0 = M.vector().array()
t0 = 0
r = ode(f, j).set_integrator('vode', method='bdf', with_jacobian=True)
r.set_initial_value(y0, t0)

t1 = 1e-9
dt = 1e-11

while r.successful() and r.t < t1-dt:
    r.integrate(r.t + dt)
print r.y
print counter
