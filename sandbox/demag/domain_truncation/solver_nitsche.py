"""Nitsche's Method for a discontinuous variational formulation is used
to solve a given demag field problem. The parameter gamma tweaks the
amount of discontinuity we allow over the core boundary"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import dolfin as df
import finmag.util.doffinder as dff
import finmag.energies.demag.solver_base as sb

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

    def get_demagfield(self,phi = None,use_default_function_space = True):
        """
        Returns the projection of the negative gradient of
        phi onto a DG0 space defined on the same mesh
        Note: Do not trust the viper solver to plot the DeMag field,
        it can give some wierd results, paraview is recommended instead

        use_default_function_space - If true project into self.Hdemagspace,
                                     if false project into a Vector DG0 space
                                     over the mesh of phi.
        """
        if phi is None:
            phi = self.phi
        
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

class NitscheSolver(TruncDeMagSolver):
    def __init__(self,problem, degree = 1):
        """
        problem - Object from class derived from TruncDemagProblem

        degree - desired polynomial degree of basis functions
        """
        self.problem = problem
        self.degree = degree
        super(NitscheSolver,self).__init__(problem)
 
    def solve(self):
        """Solve the demag problem and store the Solution"""

        #Get the solution Space
        V = self.V

        W = df.MixedFunctionSpace((V,V)) # for phi0 and phi1
        u0,u1 = df.TestFunctions(W)
        v0,v1 = df.TrialFunctions(W)
        sol = df.Function(W) 
        phi0 = df.Function(V)
        phi1 = df.Function(V)
        phi = df.Function(V)
        self.phitest = df.Function(V)
        h = self.problem.mesh.hmin() #minimum edge length or smallest diametre of mesh
        gamma = self.problem.gamma

        #Define the magnetisation
        M = self.M

        N = df.FacetNormal(self.problem.coremesh) #computes normals on the submesh self.problem.coremesh
        dSC = self.problem.dSC #Boundary of Core
        dxV = self.problem.dxV #Vacuum
        dxC = self.problem.dxC #Core 

        #Define jumps and averages accross the boundary
        jumpu = u1('-') - u0('+')                         #create a difference symbol
                                                          #- is the inward normal
                                                          #+ is the outward pointing normal or direction.
        avggradu = (df.grad(u1('-')) + df.grad(u0('+')))*0.5
        jumpv = (v1('-') - v0('+'))
        avgv = (v1('-') + v0('+'))*0.5
        avggradv = (df.grad(v1('-')) + df.grad(v0('+')))*0.5

        #Forms for Poisson problem with Nitsche Method
        a0 = df.dot(df.grad(u0),df.grad(v0))*dxV #Vacuum 
        a1 = df.dot(df.grad(u1),df.grad(v1))*dxC #Core

        #right hand side
        f = (-df.div(M)*v1)*dxC   #Source term in core
        f += (df.dot(M('-'),N('+'))*avgv )*dSC  #Prescribed outer normal derivative
        #Cross terms on the interior boundary
        c = (-df.dot(avggradu,N('+'))*jumpv - df.dot(avggradv,N('+'))*jumpu + gamma*(1/h)*jumpu*jumpv)*dSC  

        a = a0 + a1 + c

        #Dirichlet BC for phi0
        dbc = df.DirichletBC(W.sub(0),0.0,"on_boundary") #Need to use W as assemble thinks about W-space

        #The following arguments
        #  cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc
        #tell the assembler about the marks 0 or 1 for the cells and markers 0, 1 and 2 for the facets.
        
        A = df.assemble(a,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)
        F = df.assemble(f,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)

        #Solve for the solution
        dbc.apply(A)
        dbc.apply(F)

        #Got to here working through the code with Gabriel (HF, 23 Feb 2011)

        A.ident_zeros()
        df.solve(A, sol.vector(),F)

        #Seperate the mixed function and then add the parts
        solphi0,solphi1 = sol.split()
        phi0.assign(solphi0)
        phi1.assign(solphi1)
        
        phi.vector()[:] = phi0.vector() + phi1.vector()
        self.phitest.assign(phi)
        #Divide the value of phial by 2 on the core boundary
        BOUNDNUM = 2
        #Get the boundary dofs
        corebounddofs = dff.bounddofs(V, self.problem.coreboundfunc, BOUNDNUM)
        #Halve their value    
        for index,dof in enumerate(phi.vector()):
            if index in corebounddofs:
                phi.vector()[index] = dof*0.5

        #Get the function restricted to the magnetic core
        self.phi_core = self.restrictfunc(phi,self.problem.coremesh)
        #Save the demag field over the core
        self.Hdemag_core = self.get_demagfield(self.phi_core,use_default_function_space = False)
        self.Hdemag = self.get_demagfield(phi)
        #Store variables for outside testing
        self.phi = phi
        self.phi0 = phi0
        self.phi1 = phi1
        self.sol = sol
        self.gamma = gamma
        return phi
