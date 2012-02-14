from dolfin import *

# Mesh
m = 1e-5
mesh = Box(0,m,0,m,0,m,1,1,1)

# K1 for Fe = 48e3 J/m^3
K1 = 48e3

# Exact solution to the anisotropy energy.
# K1 x volume of mesh
vol  = assemble(Constant(1)*dx, mesh=mesh)
dofs = mesh.num_vertices()
E_exact = K1*vol

# Functionspace
V = VectorFunctionSpace(mesh, "CG", 1)
K1 = Constant(K1)

# Initial direction of the magnetic field.
M = project(Constant((0.5,0,0.5)), V)

# Easy axes
a = Constant((0,0,1))

# Anisotropy energy
E_ani = K1*(1 - (dot(a, M))**2)*dx

# Print value of E_ani
E = assemble(E_ani)
print 'Anisotropy energy (should be equal to %g) =' % E_exact, E

# Gradient of anisotropy energy
g_ani = derivative(E_ani, M)

# Print the gradient
H1 = assemble(g_ani)

# FIXME: This becomes zeros... Probably related to the compiler warning 
# "Summation index does not appear exactly twice: ?"
# Not sure what this means. Get same warning and result when just replacing
# exchange energy with anisotropy energy in exchange_3d_test.py.
print 'Gradient of the anisotropy energy (should NOT be zeros) =', H1.array()

