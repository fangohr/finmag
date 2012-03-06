from dolfin import *

# Compare the project and interpolate methods for 
# a small mesh size. Would assume that both
# methods give the same result in this 
# simple case.

# Mesh and functionspace
L = 10e-9    #10 nm
n = 5
mesh = Box(0,0,0,L,L,L,n,n,n)
V  = VectorFunctionSpace(mesh, "CG", 1)

# Initial magnetisation 
M0  = Constant((0,0,1))

# Using project
Mp = project(M0, V)
print "This should be an array of 0 on the first 2/3 and 1 on the last 1/3:"
print Mp.vector().array()
print "... but it isn't. Don't know the reason why project gives this result yet."

# Using interpolate
Mi = interpolate(M0, V)
print "Interpolate gives the result we assumed:"
print Mi.vector().array()

