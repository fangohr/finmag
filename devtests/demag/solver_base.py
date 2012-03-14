"""Base Classes for Demagnetization solvers"""
__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np

class DeMagSolver(object):
    """Base class for Demag Solvers"""
    def __init__(self,problem,degree =1):
        """problem - Object from class derived from TruncDemagProblem"""
        self.problem = problem
        self.degree = degree
        #Create the space for the potential function
        self.V = FunctionSpace(self.problem.mesh,"CG",degree)
        
        #Convert M into a function
        #HF: I think this should always be
        #Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree,3)
        #GB: ToDo fix the magnetisation of the problems so Mspace is always 3d
        #Some work on taking a lower dim Unit Normal and making it 3D is needed
        if self.problem.mesh.topology().dim() == 1:
            self.Mspace = FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            self.Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        #Define the magnetisation
        self.M = interpolate(Expression(self.problem.M),self.Mspace)
        
    def save_function(self,function,name):
        """
        The function is saved as a file name.pvd under the folder ~/results.
        It can be viewed with paraviewer or mayavi
        """
        file = File("results/"+ name + ".pvd")
        file << function

class FemBemDeMagSolver(DeMagSolver):
    """Base Class for FEM/BEM Demag Solvers"""
    def __init__(self,problem,degree = 1):
        super(FemBemDeMagSolver,self).__init__(problem,degree)
        #Paramters to use a quadrature rule that avoids the endpoints
        #of a triangle
        self.ffc_options = {"quadrature_rule":"canonical" \
                            ,"fquadrature_degree":1}
        #Change the Function space to CR
        self.V = FunctionSpace(self.problem.mesh,"CR",degree)
        #Total Function that we solve for
        self.phitot = Function(self.V)
        
    def calc_phitot(self,func1,func2):
        """Add two functions to get phitotal"""
        self.phitot.vector()[:] = func1.vector()[:] + func2.vector()[:]
        return self.phitot
            
    def get_boundary_dofs(self):
        """Gets the dofs that live on the boundary of a mesh"""
        dummyBC = DirichletBC(self.V,0,"on_boundary")
        return dummyBC.get_boundary_values()

    def get_boundary_dof_coordinate_dict(self):
        """
        Provides a dictionary with boundary DOF's
        key = DOFnumber in the functionspace defined over the entire mesh
        value = coordinates of DOF
        The order that the dofs appear in the dic becomes their col/row
        number in the BEM matrix.
        """
        #objects needed
        mesh = self.problem.mesh
        d = mesh.topology().dim()
        dm = self.V.dofmap()      
        boundarydofs = self.get_boundary_dofs()
        mesh.init()
        
        #Build a dictionary with all dofs and their coordinates
        #TODO Optimize me!
        doftionary = {}
        for facet in facets(mesh):
            cells = facet.entities(d)
            #create one cell (since we have CG)
            cell = Cell(mesh, cells[0])
            #Create the cell dofs and see if any
            #of the global numbers turn up in BoundaryDofs
            #If so update the BoundaryDic with the coordinates
            celldofcord = dm.tabulate_coordinates(cell)
            globaldofs = dm.cell_dofs(cells[0])
            globalcoord = dm.tabulate_coordinates(cell)
            for locind,dof in enumerate(globaldofs):
                doftionary[dof] = globalcoord[locind]
        #restrict the doftionary to the boundary
        for x in [x for x in doftionary if x not in boundarydofs]:
            doftionary.pop(x)
        return doftionary
    
    def restrict_to(self,bigvector,dofs):
        """Restrict a vector to the dofs in dofs (usually boundary)"""
        vector = np.zeros(len(dofs))
        for i,key in enumerate(dofs):
             vector[i] = bigvector[key]
        return vector
    
    def solve_laplace_inside(self,function):
        """Take a functions boundary data as a dirichlet BC and solve
            a laplace equation"""
        V = function.function_space()
        bc = DirichletBC(V,function, "on_boundary")
        u = TrialFunction(V)
        v = TestFunction(V)

        #Laplace forms
        a = inner(grad(u),grad(v))*dx
        A = assemble(a)

        #RHS = 0
        f = Function(V).vector()
        #Apply BC
        bc.apply(A,f)

        solve(A,function.vector(),f)
        return function
        
class TruncDeMagSolver(DeMagSolver):
    """Base Class for truncated Demag Solvers"""
    def __init__(self,problem,degree = 1):
            super(TruncDeMagSolver,self).__init__(problem,degree)
            
    def restrictfunc(self,function,submesh):
        """
        Restricts a function in a P1 space to a submesh
        using the fact that the verticies and DOFS are
        numbered the same way. A function defined on a P1
        space is returned. This will ONLY work for P1 elements
        For other elements the dolfin FunctionSpace.restrict()
        should be used when it is implemented.
        """
        wholemesh = function.function_space().mesh()

        #Since compute_vertex_map only accepts a SubMesh object we need to
        #create a trivial "SubMesh" of the whole mesh 
        dummymeshfunc = MeshFunction("uint",wholemesh,wholemesh.topology().dim())
        dummymeshfunc.set_all(1)

        #This is actually the whole mesh, but compute_vertex_map, only accepts a SubMesh
        wholesubmesh = SubMesh(wholemesh,dummymeshfunc,1) #THIS CRASHES!!!!BLAH
        #Mapping from the wholesubmesh to the wholemesh
        map_to_mesh = wholesubmesh.data().mesh_function("parent_vertex_indices")

        #This is a dictionary mapping the matching DOFS from a Submesh to a SubMesh
        vm = compute_vertex_map(submesh,wholesubmesh) 
        #Now we want to "restrict" the function to the restricted space
        restrictedspace = FunctionSpace(submesh,"CG",1)
        restrictedfunction = Function(restrictedspace)
        for index,dof in enumerate(restrictedfunction.vector()):
            restrictedfunction.vector()[index] = function.vector()[map_to_mesh[vm[index]]]
        return restrictedfunction


#Not used now but may be useful later
##def unit_vector_functions(self,mesh):
##    """Builds Unit Vector functions defined over the whole mesh"""
##    ##uvecspace = VectorFunctionSpace(mesh,"DG",0)
##    d = mesh.topology().dim()
##    #Create a zero vector"        
##    zerovec = [0 for i in range(d)]
##    #Initialize unit vector list
##    elist = [zerovec[:] for i in range(d)]
##    #Change an entry to get a unit vector
##    for i in range(d):          
##        elist[i][i] = 1
##    #Generate constants
##    elist = [Constant(tuple(elist[i])) for i in range(len(elist))]
##    print elist
##    return elist

