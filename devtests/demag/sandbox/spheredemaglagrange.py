"The demagnetisation field for a unit cicle is computed given a Mag field"
"(1,0,0) and an approximation of the infinite domain"

#The boundary of the magnetic core seems to be calculated fine. The number of facets found is the same
#as the number of facets on the boundary of the core submesh. The volume of the calculated boundary
#is also close to the exact surface area of a sphere with the same radius.
#The jump boundary condition is implemented as a Lagrange Multiplier

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from interiorboundary import *
import math
import numpy as np
np.set_printoptions(edgeitems= 2000)

mesh = UnitCircle(20)

V = FunctionSpace(mesh,"CG",2)
L = FunctionSpace(mesh,"CG",2)
M = MixedFunctionSpace((V,L))
VV = VectorFunctionSpace(mesh,"CG",2)
u,l = TestFunctions(M)
v,q = TrialFunctions(M)
sol = Function(M) 
phi = Function(V)

#Define the magnetisation
M = interpolate(Expression(("1","0")),VV)

#Define the Magnetic domain
r = 0.2 #Radius of magnetic Core
class MagCore(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]*x[0] + x[1]*x[1] < r*r + DOLFIN_EPS
    
corefunc = MeshFunction("uint", mesh, 2)
corefunc.set_all(0)
coredom = MagCore()  
coredom.mark(corefunc,1)
#Measure for integration over the core
dxC = dx(1)

###################################
#Define the magnetic boundary layer
###################################
intbound = InteriorBoundary(mesh)
coremesh = SubMesh(mesh,corefunc,1)
intbound.create_boundary(coremesh)
intfacet = intbound.boundaries[0]
##coreboundmeshfacets = create_intbound(mesh,MagCore())
N = FacetNormal(coremesh)
dSC = dS(2)

coreboundmesh = BoundaryMesh(coremesh)
print "Number of facets in submesh", coreboundmesh.num_cells()

#Forms for Poisson Equation
a = dot(grad(u),grad(v))*dx
f = (-div(M)*v)*dxC  #Source term in core
f += q('-')*(dot(M,N)('-'))*dSC
a += q('-')*jump(grad(u),N)*dSC #Jump condition on core boundary
a += (l*v)('-')*dSC

#Dirichlet BC at our approximate boundary
dbc = DirichletBC(V,0.0,"on_boundary")

A = assemble(a,cell_domains = corefunc, interior_facet_domains = intfacet)
F = assemble(f,cell_domains = corefunc, interior_facet_domains = intfacet)

dbc.apply(A)
dbc.apply(F)
A.ident_zeros()
solve(A, sol.vector(),F)
solphi,sollag = sol.split()
phi.assign(solphi)
print phi.vector().array()
print phi
#demag = project(grad(phi),VV)

#plot(phi)
#interactive()


###############################
#Give the Volume of the surface of the magnetic core
###############################
one = interpolate(Constant(1),V)
volform = one('-')*dSC
vol = assemble(volform,interior_facet_domains = intfacet)
print "Volume of magnetic core using generated boundary", vol

##W = FunctionSpace(coreboundmesh,"CG",1)
##one = interpolate(Constant(1),W)
##volform = one*dx
##vol = assemble(volform)
##print "Volume of magnetic core usig boundary of submesh", vol
print "Exact surface area of a sphere with same radius as magnetic core", 2*math.pi*r

###############################
#Post process
##############################
##demagfile = File("demag.pvd")
##demagfile << sol
##print "value of Demag field on the boundary point (1,0,0)"
##print "x", sol[0]((1,0,0))
##print "y", sol[1]((1,0,0))
##print "z", sol[2]((1,0,0))
##print "somewhere else", sol[0]((0.1,0,0))

