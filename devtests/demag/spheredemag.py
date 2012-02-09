"The demagnetisation field for a unit sphere is computed given a Mag field"
"(1,0,0) and an approximation of the infinite domain"

#The boundary of the magnetic core seems to be calculated fine. The number of facets found is the same
#as the number of facets on the boundary of the core submesh. The volume of the calculated boundary
#is also close to the exact surface area of a sphere with the same radius.

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from interiorboundary import *
import math
mesh = UnitSphere(10)

V = FunctionSpace(mesh,"CG",2)
VV = VectorFunctionSpace(mesh,"CG",2)
u = TestFunction(V)
v = TrialFunction(V)
phi = Function(V)

#Define the magnetisation
M = interpolate(Expression(("1","0","0")),VV)

#Define the Magnetic domain
r = 0.2 #Radius of magnetic Core
class MagCore(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] < 0.04 + DOLFIN_EPS
    
corefunc = MeshFunction("uint", mesh, 3)
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
##coreboundfacets = create_intbound(mesh,MagCore())
N = FacetNormal(coremesh)
dSC = dS(2)

corebound = BoundaryMesh(coremesh)
print "Number of facets in submesh", corebound.num_cells()
##plot(corebound)
##interactive()

#Forms for Poisson Equation
a = dot(grad(u),grad(v))*dx
f = (-div(M)*v)*dxC  #Source term in core
f += (-dot(M,N)*v)('-')*dSC       #Neumann condition on core boundary

#Dirichlet BC at our approximate boundary
dbc = DirichletBC(V,0.0,"on_boundary")

A = assemble(a)
F = assemble(f,cell_domains = corefunc, interior_facet_domains = intfacet)

dbc.apply(A)
dbc.apply(F)

solve(A, phi.vector(),F)
sol = project(grad(phi),VV)

###############################
#Give the Volume of the surface of the magnetic core
###############################
one = interpolate(Constant(1),V)
volform = one('-')*dSC
vol = assemble(volform,interior_facet_domains = intfacet)
print "Volume of magnetic core using generated boundary", vol

##W = FunctionSpace(corebound,"CG",1)
##one = interpolate(Constant(1),W)
##volform = one*dx
##vol = assemble(volform)
##print "Volume of magnetic core usig boundary of submesh", vol
print "Exact surface area of a sphere with same radius as magnetic core", 4*math.pi*r*r
                  


###############################
#Post process
##############################
##demagfile = File("demag.pvd")
##demagfile << sol
##print "Value of Demag field in x direction at origin", sol[0]((0,0,0))
##print "Value of Demag field in y direction at origin", sol[1]((0,0,0))
##print "Value of Demag field in z direction at origin", sol[2]((0,0,0))
##print "somewhere else", sol[0]((0.1,0,0))

##plot(sol)
##interactive()


