from dolfin import *
import numpy as np

# Mesh and functionspace
x0 = 0; x1 = 15e-9; nx = 30;
y0 = -4.5e-9; y1 = 4.5e-9; ny = 18;
z0 = -0.1e-9; z1 = 0.1e-9; nz = 1;
mesh = Box(x0, y0, z0, x1, y1, z1, nx, ny, nz) 
V = VectorFunctionSpace(mesh, 'Lagrange', 1, dim=3)

# Parameters
Ms = 1e6
alpha = 0.02
gamma =  2.210173e5 # m/(As)
c = 1e11 # 1/s numerical scaling correction
C = 1.3e-11 # J/m exchange constant
K = Constant(520e3) # Anisotropy constant
a = Constant((np.sqrt(0.5), np.sqrt(0.5), 0)) # Easy axis

# Applied field
Happ = interpolate(Constant((Ms,0,0)), V)

# Initial magnetisation
m0_tuple = (("1",
             "5 * pow(cos(pi * (x[0] * pow(10, 9) - 11) / 6), 3) \
                * pow(cos(pi * x[1] * pow(10, 9) / 6), 3)",
             "0"))
M = interpolate(Expression(m0_tuple), V)

# Exchange field
Eex = inner(grad(M), grad(M))*dx
Hex = derivative(Eex, M)

# Anisotropy field
Eani = K*(Constant(1) - (dot(a, M))**2)*dx
Hani = assemble(derivative(Eani, M)) # Try with assemble to see if that changes anything

print type(Happ), type(Hex), type(Hani)
exit()

# Effective field
Heff = Hex + Hani + Happ

# LLG equation
p = gamma / (1 + alpha*alpha)
q = alpha * p
u = TrialFunction(V)
v = TestFunction(V)

a = inner(u, v)*dx
L = inner(-p * cross(M, Heff)
          -q * cross(M, cross(M, Heff)) 
          -c * (inner(M, M) - 1) * M , v) * dx

# Jacobian 
J = derivative(L, M)
