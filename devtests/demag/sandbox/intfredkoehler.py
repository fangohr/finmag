"The demagnetisation field for a unit cicle is computed given a Mag field"
"(1,0,0) and an approximation of the infinite domain"

#The boundary of the magnetic core seems to be calculated fine. The number of facets found is the same
#as the number of facets on the boundary of the core submesh. The volume of the calculated boundary
#is also close to the exact surface area of a sphere with the same radius.
#The Fredkin Koehler approach is used to solve the problem.

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from interiorboundary import *
import math
import numpy as np
import pylab as pl 
np.set_printoptions(edgeitems= 2000)

mesh = UnitInterval(40)
degree = 1
#mesh = refine(mesh)

V = FunctionSpace(mesh,"CG",degree)
phi1 = Function(V)
phi2 = Function(V)

#Define the magnetisation
M = interpolate(Expression("1"),V)

#Define the Magnetic domain
r = 0.3 #Radius of magnetic Core
class MagCore(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]< 0.5 + r + DOLFIN_EPS and x[0] > 0.5 - r - DOLFIN_EPS
    
corefunc = MeshFunction("uint", mesh, 1)
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
print "Coremesh", coremesh.coordinates()
print "Boundary of Core", coreboundmesh.coordinates()


########################################
#Solve for phi1 with a Lagrange multiplier formulation
########################################
print "***************"
print "solving for phi1"
print "***************"


#A boundary point used to specify the pure neumann problem
class BoundPoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.5 - r)

dbc1 = DirichletBC(V, 0.0, BoundPoint())

#Forms for Neumann Poisson Equation for phi1

u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u),grad(v))*dxC
f = (div(M)*v)*dxC  #Source term in core
f += (dot(M,N)*v)('+')*dSC   #Neumann Conditions on edge of core

A = assemble(a,cell_domains = corefunc, interior_facet_domains = intfacet)
F = assemble(f,cell_domains = corefunc, interior_facet_domains = intfacet)

#dbc1.apply(A,F)
print F.array()
A.ident_zeros()
print A.array()
solve(A,phi1.vector(),F)
#print phi1.vector().array()
##pl.plot(phi1.array(),mesh.coordinates())
plot(phi1, title = "phi1")
##interactive()

########################################
#Solve for phi2 with a Lagrange multiplier formulation
########################################
print "***************"
print "solving for phi2"
print "***************"

L = FunctionSpace(mesh,"CG",degree)
VD = FunctionSpace(mesh,"DG",degree)
W = MixedFunctionSpace((V,L))
u,l = TrialFunctions(W)
v,q = TestFunctions(W)
sol = Function(W)

#Forms for phi2
a = dot(grad(u),grad(v))*dx
f = q('-')*phi1('-')*dSC
a += q('-')*jump(u)*dSC #Jump in solution on core boundary
a += (l*v)('-')*dSC

#Dirichlet BC at our approximate boundary
dbc = DirichletBC(W.sub(0),0.0,"on_boundary")

A = assemble(a,cell_domains = corefunc, interior_facet_domains = intfacet)
F = assemble(f,cell_domains = corefunc, interior_facet_domains = intfacet)

dbc.apply(A)
dbc.apply(F)
A.ident_zeros()
print "Matrix", A.array()
print
print "Vector", F.array()
solve(A, sol.vector(),F)
solphi,sollag = sol.split()
phi2.assign(solphi)
print phi2.vector().array()
plot(phi2, title = "phi2")

phitot = Function(V)
phitot.vector()[:] = phi1.vector() + phi2.vector()
phitot2=project(phi1+phi2,V)
plot(phitot, title = "Sum of phi1 and phi2")
#plot(phitot2, title = "Sum of phi1 and phi2 a different way")
interactive()


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
#print "Exact surface area of a sphere with same radius as magnetic core", 2*math.pi*r
##
#################################
###Post process
################################
####demagfile = File("demag.pvd")
####demagfile << sol
####print "value of Demag field on the boundary point (1,0,0)"
####print "x", sol[0]((1,0,0))
####print "y", sol[1]((1,0,0))
####print "z", sol[2]((1,0,0))
####print "somewhere else", sol[0]((0.1,0,0))
##
