##Test of assembling and subdomains
##Gabriel Balaban

#Note when restricting forms ('+') will be on the outside of the submesh,
#and ('-') will be on the inside
#TODO Expand the class InteriorBoundary so it can handle multiple Boundaries 
from dolfin import *

class InteriorBoundary():
    def __init__(self,mesh):
        self.D = mesh.topology().dim()
        self.mesh = mesh
        self.orientation = [] 
        self.boundaries = []

#TODO add the possibility of several subdomains        
    def create_boundary(self,submesh):
        #Compute FSI boundary and orientation markers on Omega
        self.mesh.init()
        newboundfunc = MeshFunction("uint",self.mesh,self.D-1)
        neworientation= self.mesh.data().create_mesh_function("facet_orientation", self.D - 1)
        neworientation.set_all(0)
        newboundfunc.set_all(0)
        #self.mesh.init(self.D - 1, self.D)
        c = 0
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
            p0_inside = submesh.intersected_cell(p0)
            p1_inside = submesh.intersected_cell(p1)
            p0_inside = translate(p0_inside)
            p1_inside = translate(p1_inside)
            
##            p0_inside = subdomain.inside(p0, False)
##            p1_inside = subdomain.inside(p1, False)
            # Just set c0, will be set only for facets below
            neworientation[facet.index()] = c0

            # Markers:
            # 0 = Not Subdomain
            # 1 = Subdomain
            # 2 = Boundary

            # Look for points where exactly one is inside the Subdomain
            facet_index = facet.index()
            if p0_inside and not p1_inside:
                newboundfunc[facet_index] = 2
                neworientation[facet_index] = c1
                c += 1
            elif p1_inside and not p0_inside:
                newboundfunc[facet_index] = 2
                neworientation[facet_index] = c0
                c += 1
            elif p0_inside and p1_inside:
                newboundfunc[facet_index] = 1
            else:
                newboundfunc[facet_index] = 0
        print "Number of boundary facets found manually", c
        boundmesh = BoundaryMesh(submesh)
        M = boundmesh.num_cells()
##This test should be moved to an external test suite as it only makes sense if the submesh is completely contained in the rest of the mesh
##        assert c == M, "Internal Error in Interiorboundary. c=%d != M=%d" % (c,M)
        self.boundaries += [newboundfunc]
        self.orientation += [neworientation]

#Used to check that points were inside or outside a sphere
def length(x,y,z):
    return sqrt(x*x + y*y + z*z)

def create_intbound(mesh,subdomain):
    #Automatically generates the boundary and return the facet function
    # '2' is the interior boundary
    intbound = InteriorBoundary(mesh)
    intbound.create_boundary(subdomain)
    intfacet = intbound.boundaries[0]
    return intfacet

def translate(p):
    #Translate the result of the Mesh.intersected_cells method into a true or false statement
    if p == -1:
        p = False
    else:
        p = True
    return p
#######################################################################
#Module Tests
#######################################################################
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

class TestProblemCube():
    def __init__(self):
        self.mesh = UnitCube(4,4,4)
        #plot(self.mesh)
        self.cell_domains = MeshFunction("uint", self.mesh, 3)
        class Structure(SubDomain):
            def inside(self, x, on_boundary):
                return x[2] < 0.75 + DOLFIN_EPS
        
        #Create Submesh
        self.Structure = Structure
        self.cell_domains.set_all(0)
        subdomain = Structure()
        subdomain.mark(self.cell_domains,1)
        self.submesh = SubMesh(self.mesh,self.cell_domains,1)

#Test Case for the module
#Solve a test Laplace Equation with a Neumann boundary on the interior
#Changing the F should change the solution
if __name__ == "__main__":
    problem = TestProblem()
    intbound = InteriorBoundary(problem.mesh)
    intbound.create_boundary(problem.submesh)
    intfacet = intbound.boundaries[0]
    
    N_S = FacetNormal(problem.submesh)
    V = FunctionSpace(problem.mesh,"CG",1)
    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)
    A = inner(grad(u),grad(v))*dx(1)
    vec = Constant((3,1))
    F = (dot(vec,N_S)*v)('-')*dS(2)
   ## F = sol*v*dx  #If sol is initial this form is 0 when assembled 
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
