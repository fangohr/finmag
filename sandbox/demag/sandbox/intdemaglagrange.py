"The demagnetisation field for a unit interval is computed given a Mag field"
"(1,0,0) and an approximation of the infinite domain"

#The jump interior boundary condition is implemented as a Lagrange Multiplier

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from interiorboundary import *
import math
import numpy as np
import pylab as pl
import matrixdoctor as md
np.set_printoptions(edgeitems= 2000)

mesh = UnitInterval(5)

V = FunctionSpace(mesh,"CG",1)
L = FunctionSpace(mesh,"CG",1)
W = MixedFunctionSpace((V,L))
##VV = VectorFunctionSpace(mesh,"CG",2)
u,l = TestFunctions(W)
v,q = TrialFunctions(W)
sol = Function(W) 
phi = Function(V)

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
print "Mesh coordinates", mesh.coordinates()
print "Submesh coordinates", coremesh.coordinates()
##print "Number of facets in submesh", coreboundmesh.num_cells()

#Forms for Poisson Equation
a = dot(grad(u),grad(v))*dx
f = (div(M)*v)*dxC  #Source term in core
f += q('-')*(dot(M,N)('-'))*dSC
a += q('-')*jump(grad(u))*N('-')*dSC #Jump in derivative on core boundary
#a += q('-')*jump(u)*dSC #Jump in solution on core boundary
a += (l*v)('-')*dSC

#Dirichlet BC at our approximate boundary
dbc = DirichletBC(W.sub(0),0.0,"on_boundary")

A = assemble(a,cell_domains = corefunc, interior_facet_domains = intfacet)
F = assemble(f,cell_domains = corefunc, interior_facet_domains = intfacet)

dbc.apply(A)
dbc.apply(F)
A.ident_zeros()
print "Matrix"
print A.array()
print "Vector f"
print F.array()
md.diagnose(A.array(),"Jacobian")
solve(A, sol.vector(),F)
solphi,sollag = sol.split()
phi.assign(sollag)
print "Lagrange mult DOFS", phi.vector().array()
#demag = project(grad(phi),VV)
phi.assign(solphi)
print "Solution DOFS", phi.vector().array()

#plot(phi)
#interactive()


###############################
#Give the Volume of the surface of the magnetic core
###############################
#one = interpolate(Constant(1),V)
#volform = one('-')*dSC
#vol = assemble(volform,interior_facet_domains = intfacet)
#print "Volume of magnetic core using generated boundary", vol

##W = FunctionSpace(coreboundmesh,"CG",1)
##one = interpolate(Constant(1),W)
##volform = one*dx
##vol = assemble(volform)
##print "Volume of magnetic core usig boundary of submesh", vol
#print "Exact surface area of a sphere with same radius as magnetic core", 2

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

