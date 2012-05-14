"""Base Classes for Demagnetization solvers"""
__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import dolfin as df
import numpy as np
import abc
from finmag.util.timings import timings

class FemBemDeMagSolver(object):
    """Base Class for FEM/BEM Demag Solvers containing shared methods
        for a top level demag solver interface see
        class Demag in finmag/energies/demag
        
        *Arguments*
        problem
            An object of type DemagProblem
        degree
            polynomial degree of the function space
        element
            finite element type, default is "CG" or Lagrange polynomial.
        unit_length
            the scale of the mesh, defaults to 1.
        project_method
            method to calculate the demag field from the potential
            possible methods are
                * 'magpar'
                * 'project'
    """
                
    def __init__(self, problem, degree=1, element="CG",project_method = 'magpar',
                 unit_length = 1):

        #Problem objects and parameters
        self.problem = problem
        self.mesh = problem.mesh
        self.unit_length = unit_length
        
        #Mesh Facet Normal
        self.n = df.FacetNormal(self.mesh)
        
        #Unit Magentisation field
        self.m = problem.M
        self.Ms = problem.Ms
        
        #Spaces and functions for the Demag Potential
        self.V = df.FunctionSpace(self.problem.mesh,element,degree)
        self.v = df.TestFunction(self.V)
        self.u = df.TrialFunction(self.V)
        self.phi = df.Function(self.V)

        #Space and functions for the Demag Field
        self.W = df.VectorFunctionSpace(self.mesh, element, degree, dim=3)
        self.w = df.TrialFunction(self.W)
        self.vv = df.TestFunction(self.W)
        self.H_demag = df.Function(self.W)

        # Initilize the boundary element matrix variable
        self.bem = None

        #Objects that are needed frequently for linear solves.
        self.poisson_matrix = self.build_poisson_matrix()
        #2nd FEM.
        self.laplace_zeros = df.Function(self.V).vector()
        self.laplace_solver = df.KrylovSolver()
        self.laplace_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True

        #Objects needed for energy density computation
        self.nodal_vol = df.assemble(self.v*df.dx, mesh=self.mesh).array()
        self.ED = df.Function(self.V)

        #Method to calculate the Demag field from the potential
        self.project_method = project_method
        if self.project_method == 'magpar':
            self.__setup_field_magpar()
            self.__compute_field = self.__compute_field_magpar
        elif self.project_method == 'project':
            self.__compute_field = self.__compute_field_project
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'magpar',
                                    * 'project'""")
    @abc.abstractmethod
    def solve():
        return

    def compute_field(self):
        """
        Compute the demag field.

        .. note::
            The interface has to be changed to this later anyway, so
            we can just keep it this way so we don't need to change the
            examples later.

        *Returns*
            numpy.ndarray
                The demag field.

        """
        self.solve()
        return self.__compute_field()

    def scalar_potential(self):
        """Return the scalar potential."""
        return self.phi

    def compute_energy(self):
        """
        Compute the demag energy defined by

        .. math::

            E_\\mathrm{demag} = -\\frac12 \\mu_0 \\int_\\Omega
            H_\\mathrm{demag} \\cdot \\vec M \\mathrm{d}x

        *Returns*
            Float
                The demag energy.

        """
        self.H_demag.vector()[:] = self.compute_field()
        E = -0.5*self.mu0*df.dot(self.H_demag, self.m*self.Ms)*df.dx
        return df.assemble(E, mesh=self.mesh)*\
                self.unit_length**self.mesh.topology().dim()

    def energy_density(self):
        """
        Compute the demag energy density,

        .. math::

            \\frac{E_\\mathrm{demag}}{V},

        where V is the volume of each node.

        *Returns*
            numpy.ndarray
                The demag energy density.

        """
        self.H_demag.vector()[:] = self.compute_field()
        E = df.dot(-0.5*self.mu0*df.dot(self.H_demag, self.m*self.Ms), self.v)*df.dx
        nodal_E = df.assemble(E).array()
        return nodal_E/self.nodal_vol

    def energy_density_function(self):
        """
        Compute the demag energy density the same way as the
        function above, but return a Function to allow probing.

        *Returns*
            dolfin.Function
                The demag energy density.
        """
        self.ED.vector()[:] = self.energy_density()
        return self.ED
    
    def calc_phitot(self,func1,func2):
        """Add two functions to get phitotal"""
        self.phi.vector()[:] = func1.vector()[:] + func2.vector()[:]
        return self.phi

    def build_poisson_matrix(self):
        """assemble a poisson equation 'stiffness' matrix"""
        a = df.dot(df.grad(self.u),df.grad(self.v))*df.dx
        return df.assemble(a)

    def solve_laplace_inside(self, function, solverparams=None):
        """Take a functions boundary data as a dirichlet BC and solve
            a laplace equation"""
        bc = df.DirichletBC(self.V, function, df.DomainBoundary())
        A = self.poisson_matrix.copy()
        b = self.laplace_zeros.copy()
        bc.apply(A, b)
        df.solve(A,function.vector(),b)
        return function

    def __compute_field_project(self):
        """
        Dolfin method of projecting the scalar potential
        onto a dolfin.VectorFunctionSpace.
        """
        Hdemag = df.project(-df.grad(self.phi), self.W)
        return Hdemag.vector().array()

    def __setup_field_magpar(self):
        """Needed by the magpar method we may use instead of project."""
        #FIXME: Someone with a bit more insight in this method should
        # write something about it in the documentation.
        a = df.inner(df.grad(self.u), self.vv)*df.dx
        b = df.dot(self.vv, df.Constant([-1, -1, -1]))*df.dx
        self.G = df.assemble(a)
        self.L = df.assemble(b).array()

    def __compute_field_magpar(self):
        """Magpar method used by Weiwei."""
        timings.start("Compute field")
        Hd = self.G*self.phi.vector()
        Hd = Hd.array()/self.L
        timings.stop("Compute field")
        return Hd

    def get_demagfield(self,phi,use_default_function_space = True):
        """
        Returns the projection of the negative gradient of
        phi onto a DG0 space defined on the same mesh
        Note: Do not trust the viper solver to plot the DeMag field,
        it can give some wierd results, paraview is recommended instead

        use_default_function_space - If true project into self.Hdemagspace,
                                     if false project into a Vector DG0 space
                                     over the mesh of phi.
        """

        Hdemag = -df.grad(phi)
        if use_default_function_space == True:
            Hdemag = df.project(Hdemag,self.Hdemagspace)
        else:
            if self.D == 1:
                Hspace = df.FunctionSpace(phi.function_space().mesh(),"DG",0)
            else:
                Hspace = df.VectorFunctionSpace(phi.function_space().mesh(),"DG",0)
            Hdemag = df.project(Hdemag,Hspace)
        return Hdemag
    

class TruncDeMagSolver(object):
    """Base Class for truncated domain type Demag Solvers"""
    def __init__(self,problem,degree = 1):
        """problem - Object from class derived from FEMBEMDemagProblem"""
        self.problem = problem
        self.degree = degree
        #Create the space for the potential function
        self.V = df.FunctionSpace(self.problem.mesh,"CG",degree)
        #Get the dimension of the mesh
        self.D = problem.mesh.topology().dim()
        #Create the space for the Demag Field
        if self.D == 1:
            self.Hdemagspace = df.FunctionSpace(problem.mesh,"DG",0)
        else:
            self.Hdemagspace = df.VectorFunctionSpace(problem.mesh,"DG",0)

        #Convert M into a function
        #HF: I think this should always be
        #Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree,3)
        #GB: ToDo fix the magnetisation of the problems so Mspace is always 3d
        #Some work on taking a lower dim Unit Normal and making it 3D is needed
        if self.D == 1:
            self.Mspace = df.FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            self.Mspace = df.VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        
        #Define the magnetisation
        # At the moment this just accepts strings, tuple of strings
        # or a dolfin.Function
        # TODO: When the magnetisation no longer are allowed to be 
        # one-dimensional, remove " or isinstance(self.problem.M, str)"
        if isinstance(self.problem.M, tuple) or isinstance(self.problem.M, str):
            self.M = df.interpolate(df.Expression(self.problem.M),self.Mspace)
        elif 'dolfin.functions.function.Function' in str(type(self.problem.M)):
            self.M = self.problem.M
        else:
            raise NotImplementedError("%s is not implemented." \
                    % type(self.problem.M))
            
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
        dummymeshfunc = df.MeshFunction("uint",wholemesh,wholemesh.topology().dim())
        dummymeshfunc.set_all(1)

        #This is actually the whole mesh, but compute_vertex_map, only accepts a SubMesh
        wholesubmesh = df.SubMesh(wholemesh,dummymeshfunc,1)
        #Mapping from the wholesubmesh to the wholemesh
        map_to_mesh = wholesubmesh.data().mesh_function("parent_vertex_indices")

        #This is a dictionary mapping the matching DOFS from a Submesh to a SubMesh
        vm = df.compute_vertex_map(submesh,wholesubmesh) 
        #Now we want to "restrict" the function to the restricted space
        restrictedspace = df.FunctionSpace(submesh,"CG",1)
        restrictedfunction = df.Function(restrictedspace)
        for index,dof in enumerate(restrictedfunction.vector()):
            restrictedfunction.vector()[index] = function.vector()[map_to_mesh[vm[index]]]
        return restrictedfunction

    def get_demagfield(self,phi,use_default_function_space = True):
        """
        Returns the projection of the negative gradient of
        phi onto a DG0 space defined on the same mesh
        Note: Do not trust the viper solver to plot the DeMag field,
        it can give some wierd results, paraview is recommended instead

        use_default_function_space - If true project into self.Hdemagspace,
                                     if false project into a Vector DG0 space
                                     over the mesh of phi.
        """

        Hdemag = -df.grad(phi)
        if use_default_function_space == True:
            Hdemag = df.project(Hdemag,self.Hdemagspace)
        else:
            if self.D == 1:
                Hspace = df.FunctionSpace(phi.function_space().mesh(),"DG",0)
            else:
                Hspace = df.VectorFunctionSpace(phi.function_space().mesh(),"DG",0)
            Hdemag = df.project(Hdemag,Hspace)
        return Hdemag
        
    def save_function(self,function,name):
        """
        The function is saved as a file name.pvd under the folder ~/results.
        It can be viewed with paraviewer or mayavi
        """
        file = File("results/"+ name + ".pvd")
        file << function
