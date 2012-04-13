"""Base Classes for Demagnetization solvers"""
__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np
import instant

class DeMagSolver(object):
    """Base class for Demag Solvers"""
    def __init__(self,problem,degree =1):
        """problem - Object from class derived from TruncDemagProblem"""
        self.problem = problem
        self.degree = degree
        #Create the space for the potential function
        self.V = FunctionSpace(self.problem.mesh,"CG",degree)
        #Get the dimension of the mesh
        self.D = problem.mesh.topology().dim()
        #Create the space for the Demag Field
        if self.D == 1:
            self.Hdemagspace = FunctionSpace(problem.mesh,"DG",0)
        else:
            self.Hdemagspace = VectorFunctionSpace(problem.mesh,"DG",0)

        #Convert M into a function
        #HF: I think this should always be
        #Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree,3)
        #GB: ToDo fix the magnetisation of the problems so Mspace is always 3d
        #Some work on taking a lower dim Unit Normal and making it 3D is needed
        if self.D == 1:
            self.Mspace = FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            self.Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        
        #Define the magnetisation
        # At the moment this just accepts strings, tuple of strings
        # or a dolfin.Function
        # TODO: When the magnetisation no longer are allowed to be 
        # one-dimensional, remove " or isinstance(self.problem.M, str)"
        if isinstance(self.problem.M, tuple) or isinstance(self.problem.M, str):
            self.M = interpolate(Expression(self.problem.M),self.Mspace)
        elif 'dolfin.functions.function.Function' in str(type(self.problem.M)):
            self.M = self.problem.M
        else:
            raise NotImplementedError("%s is not implemented." \
                    % type(self.problem.M))

        
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


        Hdemag = -grad(phi)
        if use_default_function_space == True:
            Hdemag = project(Hdemag,self.Hdemagspace)
        else:
            if self.D == 1:
                Hspace = FunctionSpace(phi.function_space().mesh(),"DG",0)
            else:
                Hspace = VectorFunctionSpace(phi.function_space().mesh(),"DG",0)
            Hdemag = project(Hdemag,Hspace)
        return Hdemag
        
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
        #Parameters to use a quadrature rule that avoids the endpoints
        #of a triangle #GB NOT WORKING with CG1 elements
        self.ffc_options = {"quadrature_rule":"canonical"}
        
        #Change the Function space to CR
        self.V = FunctionSpace(self.problem.mesh,"CR",degree)
        #Create and store a Trial and Test Function
        self.v = TestFunction(self.V)
        self.u= TrialFunction(self.V)
        #Total Function that we solve for
        self.phi = Function(self.V)

        # Store the computed boundary element matrix
        self.bem = None

        #build the boundary data (normals and coordinates)
        self.build_boundary_data()
        self.build_crestrict_to()
        
    def build_crestrict_to(self):
        #Create the c++ function for restrict_to
        c_code_restrict_to = """
        void restrict_to(int bigvec_n, double *bigvec, int resvec_n, double *resvec, int dofs_n, unsigned int *dofs) {
            for ( int i=0; i<resvec_n; i++ )
                { resvec[i] = bigvec[int(dofs[i])]; }
        }
        """

        args = [["bigvec_n", "bigvec"],["resvec_n", "resvec"],["dofs_n","dofs","unsigned int"]]
        self.crestrict_to = instant.inline_with_numpy(c_code_restrict_to, arrays=args)
            
    def calc_phitot(self,func1,func2):
        """Add two functions to get phitotal"""
        self.phi.vector()[:] = func1.vector()[:] + func2.vector()[:]
        return self.phi
            
    def get_boundary_dofs(self,V):
        """Gets the dofs that live on the boundary of the mesh
            of function space V"""
        dummyBC = DirichletBC(V,0,"on_boundary")
        return dummyBC.get_boundary_values()

    def get_dof_normal_dict_avg(self,normtionary):
        """
        Provides a dictionary with all of the boundary DOF's as keys
        and an average of facet normal components associated to the DOF as values
        V = FunctionSpace
        """
        #Take an average of the normals in normtionary
        avgnormtionary = {k:np.array([ float(sum(i))/float(len(i)) for i in zip(*normtionary[k])]) for k in normtionary}
        #Renormalize the normals
        avgnormtionary = {k: avgnormtionary[k]/sqrt(np.dot(avgnormtionary[k],avgnormtionary[k].conj())) for k in avgnormtionary}
        return avgnormtionary
    
    def build_boundary_data(self):
        """
        Builds two boundary data dictionaries
        1.doftionary key- dofnumber, value - coordinates
        2.normtionary key - dofnumber, value - average of all facet normal components associated to a DOF
        """
        
        mesh = self.V.mesh()
        #Initialize the mesh data
        mesh.init()
        d = mesh.topology().dim()
        dm = self.V.dofmap()
        boundarydofs = self.get_boundary_dofs(self.V)
        

        #It is very import that this vector has the right length
        #It holds the local dof numbers associated to a facet 
        facetdofs = np.zeros(dm.num_facet_dofs(),dtype=np.uintc)

        #Initialize dof-to-normal dictionary
        doftonormal = {}
        doftionary = {}
        #Loop over boundary facets
        for facet in facets(mesh):
            cells = facet.entities(d)
            #one cell means we are on the boundary
            if len(cells) ==1:
                #######################################
                #Shared Data for Normal and coordinates 
                #######################################
                
                #create one cell (since we have CG)
                cell = Cell(mesh,cells[0])
                #Local to global map
                globaldofcell = dm.cell_dofs(cells[0])

                #######################################
                #Find  Dof Coordinates
                #######################################
                        
                #Create the cell dofs and see if any
                #of the global numbers turn up in BoundaryDofs
                #If so update doftionary with the coordinates
                celldofcord = dm.tabulate_coordinates(cell)

                for locind,dof in enumerate(globaldofcell):
                    if dof in boundarydofs:
                        doftionary[dof] = celldofcord[locind]
                    
                #######################################
                #Find Normals
                #######################################
                local_fi = cell.index(facet)
                dm.tabulate_facet_dofs(facetdofs,local_fi)
                #Global numbers of facet dofs
                globaldoffacet = [globaldofcell[ld] for ld in facetdofs]
                #add the facet's normal to every dof it contains
                for gdof in globaldoffacet:
                    n = facet.normal()
                    ntup = tuple([n[i] for i in range(d)])
                    #If gdof not in dictionary initialize a list
                    if gdof not in doftonormal:
                        doftonormal[gdof] = []
                    #Prevent redundancy in Normals (for example 3d UnitCube CG1)
                    if ntup not in doftonormal[gdof]:
                        doftonormal[gdof].append(ntup) 
                    
            elif len(cells) == 2:
                #we are on the inside so continue
                continue
            else:
                assert 1==2,"Expected only two cells per facet and not " + str(len(cells))
                
        #Build the average normtionary and save data
        self.doftonormal = doftonormal
        self.normtionary = self.get_dof_normal_dict_avg(doftonormal)      
        self.doftionary = doftionary
        #numpy array with type double for use by instant (c++)
        self.doflist_double = np.array(doftionary.keys(),dtype = self.normtionary[self.normtionary.keys()[0]].dtype.name)
        self.bdofs = np.array(doftionary.keys())
    
    def restrict_to(self,bigvector):
        """Restrict a vector to the dofs in dofs (usually boundary)"""
        vector = np.zeros(len(self.doflist_double))
        #Recast bigvector as a double type array when calling restict_to
        self.crestrict_to(bigvector.array().view(vector.dtype.name),vector,self.bdofs)
        return vector

    def build_poisson_matrix(self):
        """assemble a poisson equation 'stiffness' matrix"""
        a = dot(grad(self.u),grad(self.v))*dx
        self.poisson_matrix = assemble(a)

    def solve_laplace_inside(self,function,solverparams = None):
        """Take a functions boundary data as a dirichlet BC and solve
            a laplace equation"""
        bc = DirichletBC(self.V,function, "on_boundary")
        
        #Buffer data independant of M
        if not hasattr(self,"poisson_matrix"):
            self.build_poisson_matrix()
        if not hasattr(self,"laplace_F"):
            #RHS = 0
            self.laplace_f = Function(self.V).vector()

        #Copy the poisson matrix it is shared and should
        #not have bc applied to it.
        laplace_A = self.poisson_matrix
        #Apply BC
        bc.apply(laplace_A)
        #Boundary values of laplace_f are overwritten on each call.
        bc.apply(self.laplace_f)
        self.linsolve_laplace_inside(function,laplace_A,solverparams)        
        return function

    def linsolve_laplace_inside(self,function,laplace_A,solverparams = None):
        """
        Linear solve for laplace_inside written for the
        convenience of changing solver parameters in subclasses
        """
        solve(laplace_A,function.vector(),self.laplace_f,\
                  solver_parameters = solverparams)
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
        wholesubmesh = SubMesh(wholemesh,dummymeshfunc,1)
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

