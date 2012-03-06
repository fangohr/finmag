from dolfin import *

# Mesh and functionspace
L = 10e-9 # 10 nm
mesh = Box(0,0,0,L,L,L,4,4,4) 
V = VectorFunctionSpace(mesh, 'CG', 1)

# Parameters
alpha = 0.02
gamma = 2.210173e5 # m/(As)

# Applied field
Happ = interpolate(Constant((5e3, 0, 0)), V)

# Initial magnetisation
m0_tuple = (("1",
             "5 * pow(cos(pi * (x[0] * pow(10, 9) - 11) / 6.), 3) \
                * pow(cos(pi * x[1] * pow(10, 9) / 6.), 3)",
             "0"))
M = interpolate(Expression(m0_tuple), V)

# Exchange field
Eex = inner(grad(M), grad(M))*dx
Hex = derivative(Eex, M)

# Effective field (sum of applied and exchange field)
# TODO: Figure out a way to add these without using the scheme

# Heff = Function(V)
# Heff.vector()[:] = Happ.vector() + assemble(Hex)

# because then we (probably) loose the information that Heff depends on M,
# which is useful later when computing the jacobian.
# The following line fails because Hex is a ufl.form.Form
Heff = Hex + Happ

# LLG equation
p = gamma / (1 + alpha*alpha)
q = alpha * p
u = TrialFunction(V)
v = TestFunction(V)

a = inner(u, v)*dx
L = inner(-p * cross(M, Heff)
          -q * cross(M, cross(M, Heff)), v) * dx

# Jacobian 
J = derivative(L, M)
