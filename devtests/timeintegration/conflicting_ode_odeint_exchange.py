from dolfin import *
from finmag.sim.exchange import Exchange
from numpy import linspace
from scipy.integrate import odeint, ode

# FIXME: Figure out of this extreme inconsistency between ode and odeint
# FIXME: Make odeint convert when adding applied field

set_log_level(21)

# Parameters
alpha = 0.5
gamma = 2.211e5
c = Constant(1e12)
C = 1.3e-11
p = Constant(gamma/(1 + alpha**2))
Ms = 8.6e5
length = 20e-9
simplexes = 10

# Mesh and functions
mesh = Interval(simplexes, 0, length)
V = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
u = TrialFunction(V)
v = TestFunction(V)

# Orientations
left_right = 'MS * (2*x[0]/L - 1)'
up_down = 'sqrt(MS*MS - MS*MS*(2*x[0]/L - 1)*(2*x[0]/L - 1))'

# Initial
M0 = Expression((left_right, up_down, '0'), L=length, MS=Ms)
M = interpolate(M0, V)

# Exchange
H_exch1 = Exchange(V, M, C, Ms)

# Applied
H_app = interpolate(Constant((0, 0, 1e5)), V)

# Effective
H_eff = Function(V)
H_eff.vector()[:] = H_exch1.compute()# + H_app.vector()

Ms = Constant(Ms)

# Variational forms
a = inner(u, v)*dx
L = inner((-p*cross(M, H_eff)
           -p*alpha/Ms*cross(M, cross(M, H_eff))
           -c*(inner(M, M) - Ms**2)*M/Ms**2), v)*dx

# Time derivative of the magnetic field.
dM = Function(V)

# Jacobian
J = derivative(L, M)

def f(y, t):
    # Update M and H_eff
    M.vector()[:] = y
    H_eff.vector()[:] = H_exch1.compute()# + H_app.vector()
    solve(a==L, dM)
    return dM.vector().array()

def j(t, y):
    return assemble(J).array()


# Using odeint
ts = linspace(0, 1e-9, 100)
y0 = M.vector().array()
ys, infodict = odeint(f, y0, ts, rtol=10, full_output=True)
#ys, infodict = odeint(f, y0, ts, rtol=10, Dfun=j, full_output=True)
print ys[-1]
#print infodict


"""
# Using ode
y0 = M.vector().array()
t0, t1, dt = 0, 1e-9, 1e-11
r = ode(f).set_integrator('vode', method='bdf', with_jacobian=False)
#r = ode(f, j).set_integrator('vode', method='bdf', with_jacobian=True)
r.set_initial_value(y0, t0)
while r.successful() and r.t < t1-dt:
    r.integrate(r.t + dt)
print r.y
"""
