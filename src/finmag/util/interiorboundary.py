"""A Utility Module for generating interior boundaries in Fenics"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

#Note when restricting forms ('+') will be on the outside of the submesh,
#and ('-') will be on the inside
#TODO Expand the class InteriorBoundary so it can handle multiple Boundaries 
from dolfin import *


class InteriorBoundary():
    """Marks interior boundary and gives you a function newboundfunc

    """
    def __init__(self,mesh):
        """Expects mesh"""
        self.D = mesh.topology().dim()
        self.mesh = mesh
        self.orientation = [] 
        self.boundaries = []

#TODO add the possibility of several subdomains        
    def create_boundary(self,submesh):
        """
        Will find and return the boundary of the given submesh.

        This appends a FacetFunction to 
           self.boundaries.

        That FacetFunction takes the value 2 for all Facets on the bounday between submesh and the mesh.

        Boundary facets at the outer boundray of the mesh are not included.

        Also appends something (probably also a Facetfunction, or at
          least something like it) to self.orientation which contains
          a reference to the cells on the outside of the boundary for
          every facet.

        """
        #Compute FSI boundary and orientation markers on Omega
        self.mesh.init()
        newboundfunc = MeshFunction("uint",self.mesh,self.D-1) # D-1 means FacetFunction
        
        #The next line creates a 'facet_orientation' attribute(or function) in the mesh object
        neworientation= self.mesh.data().create_mesh_function("facet_orientation", self.D - 1)

        neworientation.set_all(0)
        newboundfunc.set_all(0)
        #self.mesh.init(self.D - 1, self.D)
        self.countfacets = 0
        
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
            
            # Just set c0, will be set only for facets below
            neworientation[facet.index()] = c0

            # Markers:
            # 0 = Not Subdomain
            # 1 = Subdomain
            # 2 = Boundary

            # Look for points where exactly one is inside the Subdomain
            facet_index = facet.index()      #global facet index
            if p0_inside and not p1_inside:
                newboundfunc[facet_index] = 2
                neworientation[facet_index] = c1 #store cell object on the 'outside' of the facet,
                                                 #i.e. c1 is not in the given submesh.
                self.countfacets += 1
            elif p1_inside and not p0_inside:
                newboundfunc[facet_index] = 2
                neworientation[facet_index] = c0
                self.countfacets += 1
            elif p0_inside and p1_inside:
                newboundfunc[facet_index] = 1
            else:
                newboundfunc[facet_index] = 0
        self.boundaries += [newboundfunc]
        self.orientation += [neworientation]
        
class inputerror(Exception):
    def __str__(self):
        return "Can only give Lagrange Element dimension for mesh dimensions 1-3"

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
#Module Tests (For Visual inspection)
#######################################################################
class TestProblem():
    def __init__(self):
        self.mesh = RectangleMesh(0.0,0.0,4,4,4,4)
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
