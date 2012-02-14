from dolfin import *
from numpy import linspace, sqrt
from scipy.integrate import odeint
import numpy as np

# Mesh
m = 1e-5
mesh = Box(0,m,0,m,0,m,1,1,1)

# Functionspace and functions
V   = VectorFunctionSpace(mesh, "CG", 1)
Hex = TrialFunction(V)
v   = TestFunction(V)

# Parameters
alpha = 0.5
gamma = 2.211e5
p     = Constant(gamma/(1 + alpha**2))
c     = 1e10
K1    = Constant(48e15) # 48e3for Fe (J/m^3) 48e15 is interesting

# Initial direction of the magnetic field.
Ms = 8e5
M0 = Constant((Ms,0,0))
M  = Function(V)
M.assign(M0)
#M.vector()[:] = M.vector()/sqrt(sum(M.vector()*M.vector()))
#print norm(M.vector())


# Easy axes
a = Constant((0,0,1))

# Anisotropy energy
E_ani = K1*(1 - (dot(a, M))**2)*dx

# Gradient of anisotropy energy
g_ani = derivative(E_ani, M)
g_ani = assemble(g_ani)
H = Function(V)
H.vector()[:] = g_ani.array()

Heff = project(Constant((0,1e5,0)), V)
H.vector()[:] += Heff.vector()[:]

# Variational forms of LLG
a = inner(Hex, v)*dx
L = inner((-p*cross(M,H)
           -p*alpha/Ms*cross(M,cross(M,H))
           -c*(inner(M,M) - Ms**2)*M/Ms**2), v)*dx

# Time derivative
dM = Function(V)

def f(y, t):  
    M.vector()[:] = y#/sqrt(np.dot(y,y))
    #print norm(M.vector()[:])
    g_ani = derivative(E_ani, M)
    g_ani = assemble(g_ani)
    H.vector()[:] = g_ani.array() + Heff.vector()
    print g_ani.array()
    #print M.vector().array()
    #print H.vector().array()
    #print '\n'

    solve(a==L, dM)
    plot(M)
    return dM.vector().array()

# Solve
ts = linspace(0, 1e-9, 1000)
y0 = M.vector().array()
ys = odeint(f, y0, ts)
interactive()
