#Test of 1d poisson equation with a discontinous source term

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

mesh = UnitInterval(100)
V = FunctionSpace(mesh,"CG",2)
u = TrialFunction(V)
v = TestFunction(V)
f = interpolate(Constant(1),V)
sol = Function(V)

#Define Magnetic Core
class MagCore(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 2.0/3.0 and x[0] > 1.0/3.0
    
corefunc = MeshFunction("uint", mesh, 1)
corefunc.set_all(0)
coredom = MagCore()  
coredom.mark(corefunc,1)
#Measure for integration over the core
dxC = dx(1)

#Define Core Boundary
intbound = InteriorBoundary(mesh)
intbound.create_boundary(MagCore())
coreboundfacets = intbound.boundaries[0]
##coreboundfacets = create_intbound(mesh,MagCore())
coremesh = SubMesh(mesh,corefunc,1) 
#Test to see if a point is in the submesh
p = Point(0.4)
print coremesh.intersected_cell(p)

N = FacetNormal(coremesh)
dSC = dS(2)

corebound = SubMesh(mesh,intbound.boundaries[0],2)
##plot(corebound)
##interactive()
##coreboundfacets = create_intbound(mesh,MagCore())
##coremesh = SubMesh(mesh,corefunc,1)
##N = FacetNormal(coremesh)
##dSC = dS(2)

dbc = DirichletBC(V,0.0,"on_boundary")

a = grad(u)*grad(v)*dx
l = f*v*dxC
#l+= (-N('-')*v('+'))*dSC

A = assemble(a)
L = assemble(l, cell_domains = corefunc,interior_facet_domains = coreboundfacets )
dbc.apply(A)
dbc.apply(L)
solve(A,sol.vector(),L)
##Check the length of the SubDomain
##f = interpolate(Constant(1),V)
##a = f*dxC
##A = assemble(a, cell_domains = corefunc)
##print A
plot(sol)
interactive()
