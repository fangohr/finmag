from dolfin import *
import instant
from finmag.energies import Exchange
from scipy.integrate import ode
import finmag.util.helpers as h
import finmag.util.consts as consts
import os
import numpy as np

set_log_level(21)

def m_average(y, V, vol):
    m = Function(V)
    m.vector()[:] = y

    mx = assemble(dot(m, Constant([1,0,0])) * dx)
    my = assemble(dot(m, Constant([0,1,0])) * dx)
    mz = assemble(dot(m, Constant([0,0,1])) * dx)
    return np.array([mx, my, mz]) / vol

def dolfinsolve(A, dm, L):
    b = assemble(L)
    solve(A, dm.vector(), b)
    return 0, dm.vector().array()

x0 = 0; x1 = 15e-9; nx = 30;
y0 = -4.5e-9; y1 = 4.5e-9; ny = 18;
z0 = -0.1e-9; z1 = 0.1e-9; nz = 1;
mesh = BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz) 
nb_nodes = len(mesh.coordinates())
V = VectorFunctionSpace(mesh, 'Lagrange', 1, dim=3)
Volume = assemble(Constant(1)*dx(mesh))

# Defaults from LLG
alpha = 0.5
gamma =  consts.gamma
c = 1e11 # 1/s numerical scaling correction
              # 0.1e12 1/s is the value used by default in nmag 0.2
C = 1.3e-11 # J/m exchange constant
Ms = 8.6e5 # A/m saturation magnetisation
t = 0 # s
H_app = (0, 0, 0)
H_app = interpolate(Constant(H_app), V)
pins = []

# Defaults overwrite from spinwave program
Ms = 1e6
C = 1.3e-11
#llg.c = 1e11
alpha = 0.02

m0_tuple = (("1",
             "5 * pow(cos(pi * (x[0] * pow(10, 9) - 11) / 6), 3) \
                * pow(cos(pi * x[1] * pow(10, 9) / 6), 3)",
             "0"))

M = interpolate(Expression(m0_tuple), V)
M.vector()[:] = h.fnormalise(M.vector().array())

m_arr = M.vector().array()
for i in xrange(nb_nodes):
    x, y, z = mesh.coordinates()[i]
    mx = 1; my = 0; mz = 0;
    if 8e-9 < x < 14e-9 and -3e-9 < y < 3e-9:
        pass
    else:
        m_arr[i] = mx; m_arr[i+nb_nodes] = my; m_arr[i+2*nb_nodes] = mz;

M.vector()[:] = m_arr

# LLG setup
exchange = Exchange(C)
exchange.setup(V, M, Ms)

H_eff = Function(V)
H_eff.vector()[:] = exchange.compute_field()

p = gamma / (1 + alpha*alpha)
q = alpha * p
u = TrialFunction(V)
v = TestFunction(V)

a = inner(u, v)*dx
L = inner(-p * cross(M, H_eff)
          -q * cross(M, cross(M, H_eff)) 
          -c * (inner(M,M) - 1) * M , v) * dx

dm = Function(V)
A = assemble(a)

# LLG solve_for
def solve_for(t, y):
    M.vector()[:] = y
    H_ex = exchange.compute_field()
    H_eff.vector()[:] = H_ex + H_app.vector().array()

    status, dMdt = dolfinsolve(A, dm, L)
    if status == 0:
        return dMdt

t0 = 0; dt = 0.05e-12; t1 = 10e-12
r = ode(solve_for).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
r.set_initial_value(M.vector().array(), t0)

fh = open("averages_dolfin.txt", "w")
while r.successful() and r.t <= t1:
    print "Integrating time = %gs" % (r.t)
    mx, my, mz = m_average(r.y, V, Volume)
    print str(r.t), " ", str(mx), " ",  str(my), " ", str(mz)
    fh.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")
    r.integrate(r.t + dt)
    plot(M)
fh.close()
print "Done"
interactive()
