"The demagnetisation field for a unit sphere is computed given a Mag field"
"(1,0,0) and an approximation of the infinite domain"

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from interiorboundary import *
mesh = UnitSphere(10)

V = FunctionSpace(mesh,"CG",1)
VV = VectorFunctionSpace(mesh,"CG",1)
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
coreboundfacets = create_intbound(mesh,MagCore())
coremesh = SubMesh(mesh,corefunc,1)
N = FacetNormal(coremesh)
dSC = dS(2)

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
sol = grad(phi)

###############################
#Post process
##############################
print sol
demagfile = File("demag.pvd")
solstore = Function(VV)
solstore.vector()[:] = sol.vector()
demagfile << solstore
##plot(sol)
##interactive()


