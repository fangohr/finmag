from dolfin import *
from numpy import linspace, sqrt
from scipy.integrate import odeint
import numpy as np

set_log_level(21)

# Mesh
m = 1e-5
mesh = Box(0,m,0,m,0,m,1,1,1)

# Functionspace and functions
V   = VectorFunctionSpace(mesh, "CG", 1)
u   = TrialFunction(V)
v   = TestFunction(V)

# Parameters
alpha = 0.5
gamma = 2.211e5
p     = Constant(gamma/(1 + alpha**2))
c     = 1e10
K1    = Constant(48e3) # 48e3for Fe (J/m^3) 48e15 is interesting

# Initial direction of the magnetic field.
Ms = 1# 17e5
#M0 = Constant((0,0.9,sqrt(0.19)))
#M0 = Constant((1,0,0))
M0 = Constant((1, 0, 0))

M  = Function(V)
M.assign(M0)

# Easy axes
a = Constant((0, 0, 1))

# Anisotropy energy
E_ani = K1*(1 - (dot(a, M))**2)*dx
g_ani_form = derivative(E_ani, M)

# Manually derivative
W = TestFunction(V)
g_ani_form = K1*(2*dot(a, M)*dot(a, W))*dx

H_eff = Function(V)
H_ani = assemble(g_ani_form)
H_app = project(Constant((0,1,0)), V)
H_eff.vector()[:] = H_ani.array() + H_app.vector()

# Variational forms of LLG
a = inner(u, v)*dx
L = inner((-p*cross(M,H_eff)
           -p*alpha/Ms*cross(M,cross(M,H_eff))
           -c*(inner(M,M) - Ms**2)*M/Ms**2), v)*dx

# Time derivative
dM = Function(V)

def f(y, t):  
    M.vector()[:] = y
    H_ani = assemble(g_ani_form)
    print H_ani.array()
    H_eff.vector()[:] = H_ani.array() + H_app.vector()
    
    solve(a==L, dM)
    plot(M)
    return dM.vector().array()

# Solve
ts = linspace(0, 1e-9, 1000)
y0 = M.vector().array()
ys = odeint(f, y0, ts)
#interactive()
