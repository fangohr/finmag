from dolfin import *
from scipy.integrate import odeint
import numpy as np

set_log_level(21)

# Mesh and functions
m = 1e-5
mesh = Box(0,m,0,m,0,m,1,1,1)
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
Ms = 8e5

# Initial magnetic field.
M = project(Constant((Ms, 0, 0)), V)

# Variational forms
a = inner(u, v)*dx
L = inner((-p*cross(M,H)
           -p*alpha/Ms*cross(M,cross(M,H))
           -c*(inner(M,M) - Ms**2)*M/Ms**2), v)*dx

# Ufl jacobian
J = derivative(L, M)
J = assemble(J).array()


dM1 = Function(V) # f(x)
dM2 = Function(V) # f(x+h)

# Compute f(x)
solve(a == L, dM1)

# Our jacobian
n = len(dM1.vector())
J2 = np.zeros((n,n))

h = 1e-10
for i in range(n):
    M.vector()[i] += h

    # Compute f(x + h)
    solve(a == L, dM2)

    # Forward differences
    J2[i,:] = (dM2.vector() - dM1.vector())/h
    #M.vector()[i] -= h

print J[0]
print ''
print J2[0]

