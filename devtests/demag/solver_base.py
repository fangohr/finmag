"""Base Classes for Demagnitization solvers"""
__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *

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
