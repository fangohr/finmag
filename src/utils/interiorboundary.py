##Test of assembling and subdomains
##Gabriel Balaban

#TODO Expand the class InteriorBoundary so it can handle multiple Boundaries 
from dolfin import *

class TestProblem():
    def __init__(self):
        self.mesh = Rectangle(0.0,0.0,4,4,4,4)
        #plot(self.mesh)
        self.cell_domains = MeshFunction("uint", self.mesh, 2)
        class Structure(SubDomain):
            def inside(self, x, on_boundary):
                return x[1] < 2 + DOLFIN_EPS
        
        #Create Submesh
        self.Structure = Structure
        self.cell_domains.set_all(0)
        subdomain0 = Structure()
        subdomain0.mark(self.cell_domains,1)
        self.submesh = SubMesh(self.mesh,self.cell_domains,1)
        #plot(submesh)


class InteriorBoundary():
    def __init__(self,mesh):
        self.D = mesh.topology().dim()
        self.mesh = mesh
        self.orientation = [] 
        self.boundaries = []

#TODO add the possibility of several subdomains        
    def create_boundary(self,Subdomain):
        #Compute FSI boundary and orientation markers on Omega
        newboundary = FacetFunction("uint",self.mesh,self.D)
        neworientation= self.mesh.data().create_mesh_function("facet_orientation", self.D - 1)
        neworientation.set_all(0)
        newboundary.set_all(0)
        self.mesh.init(self.D - 1, self.D)
        for facet in facets(self.mesh):
            # Skip facets on the boundary
            cells = facet.entities(self.D)
            if len(cells) == 1:
                continue
            elif len(cells) != 2:
                error("Strange, expecting one or two cells that share a boundary!")

            # Create the two cells
            c0, c1 = cells
            cell0 = Cell(self.mesh, c0)
            cell1 = Cell(self.mesh, c1)

            # Get the two midpoints
            p0 = cell0.midpoint()
            p1 = cell1.midpoint()

            # Check if the points are inside
            p0_inside = Subdomain.inside(p0, False)
            p1_inside = Subdomain.inside(p1, False)
            # Just set c0, will be set only for facets below
            neworientation[facet.index()] = c0

            # Markers:
            # 0 = Not Subdomain
            # 1 = Subdomain
            # 2 = Boundary

            # Look for points where exactly one is inside the Subdomain
            facet_index = facet.index()
            if p0_inside and not p1_inside:
                newboundary[facet_index] = 2
                neworientation[facet_index] = c1
            elif p1_inside and not p0_inside:
                newboundary[facet_index] = 2
                neworientation[facet_index] = c0
            elif p0_inside and p1_inside:
                newboundary[facet_index] = 1
            else:
                newboundary[facet_index] = 0
        self.boundaries += [newboundary]
        self.orientation += [neworientation]

def create_intbound(mesh,subdomain):
    #Automatically generates the boundary and return the facet function
    # '2' is the interior boundary
    intbound = InteriorBoundary(mesh)
    intbound.create_boundary(subdomain)
    intfacet = intbound.boundaries[0]
    return intfacet

#Test Case
if __name__ == "__main__":
    problem = TestProblem()
    intbound = InteriorBoundary(problem.mesh)
    intbound.create_boundary(problem.Structure())

    N_S = FacetNormal(problem.submesh)
    #Create a test Laplace Equation with a Neumann boundary on the interior
    V = FunctionSpace(problem.mesh,"CG",1)
    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)
    A = inner(grad(u),grad(v))*dx(1)
    vec = Constant((3,1))
    F = (dot(vec,N_S)*v)('-')*dS(2)
    A = assemble(A,cell_domains=problem.cell_domains,interior_facet_domains=intbound.boundaries[0])
    F = assemble(F,cell_domains=problem.cell_domains,interior_facet_domains=intbound.boundaries[0])
    #Dirichlet BC
    left = "near(x[0],0.0)"
    right = "near(x[0],4.0)"
    bcleft = DirichletBC(V,Constant(0.0),left)
    bcright = DirichletBC(V,Constant(1.0),right)
    bcleft.apply(A,F)
    bcright.apply(A,F)
    A.ident_zeros()
    solve(A,sol.vector(),F,"lu")
    plot(sol)
    interactive()

    
