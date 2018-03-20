from dolfin import *
import finmag.util.consts as consts

# Mesh and functionspace
L = 10e-9 # 10 nm
mesh = Box(0,0,0,L,L,L,4,4,4) 
V = VectorFunctionSpace(mesh, 'CG', 1)

# Parameters
alpha = 0.02
gamma = consts.gamma

# Initial magnetisation
m0_tuple = (("1",
             "5 * pow(cos(pi * (x[0] * pow(10, 9) - 11) / 6.), 3) \
                * pow(cos(pi * x[1] * pow(10, 9) / 6.), 3)",
             "0"))
M = interpolate(Expression(m0_tuple), V)

# Exchange field
Eex = inner(grad(M), grad(M))*dx
Hex = derivative(Eex, M)

# TODO: Figure out a way to do this without using this scheme
H = Function(V)
H.vector()[:] = assemble(Hex)
# because then we (probably) loose the information that H depends on M,
# which is useful later when computing the jacobian.

# LLG equation
p = gamma / (1 + alpha*alpha)
q = alpha * p
u = TrialFunction(V)
v = TestFunction(V)

# L should contain Hex instead of the "manually" assigned H, but this fails.
a = inner(u, v)*dx
L = inner(-p * cross(M, H)
          -q * cross(M, cross(M, H)), v) * dx

# Jacobian 
J = derivative(L, M)
