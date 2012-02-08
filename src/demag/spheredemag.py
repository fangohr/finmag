"The demagnetisation field for a unit sphere is computed given a Mag field"
"(1,0,0) and an approximation of the infinite domain"

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from interiorboundary import *
mesh = UnitSphere(10)

V = FunctionSpace(mesh,"CG",2)
VV = VectorFunctionSpace(mesh,"CG",2)
u = TestFunction(V)
v = TrialFunction(V)
phi = Function(V)

#Define the magnetisation
M = interpolate(Expression(("1","0","0")),VV)

#Define the Magnetic domain
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
intbound.create_boundary(MagCore())
intfacet = intbound.boundaries[0]
##coreboundfacets = create_intbound(mesh,MagCore())
coremesh = SubMesh(mesh,corefunc,1)
N = FacetNormal(coremesh)
dSC = dS(2)

corebound = SubMesh(mesh,intfacet,2)
plot(coremesh)
interactive()

#Forms for Poisson Equation
a = dot(grad(u),grad(v))*dx
f = (-div(M)*v)*dxC  #Source term in core
f += (-dot(M,N)*v)('-')*dSC       #Neumann condition on core boundary

#Dirichlet BC at our approximate boundary
dbc = DirichletBC(V,0.0,"on_boundary")

A = assemble(a)
F = assemble(f,cell_domains = corefunc, interior_facet_domains = coreboundfacets)

dbc.apply(A)
dbc.apply(F)

solve(A, phi.vector(),F)
sol = project(grad(phi),VV)

###############################
#Post process
##############################
##demagfile = File("demag.pvd")
##demagfile << sol
print "Value of Demag field in x direction at origin", sol[0]((0,0,0))
print "Value of Demag field in y direction at origin", sol[1]((0,0,0))
print "Value of Demag field in z direction at origin", sol[2]((0,0,0))
print "somewhere else", sol[0]((0.1,0,0))

##plot(sol)
##interactive()


