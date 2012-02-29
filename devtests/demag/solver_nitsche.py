"Nitsche's Method for a discontinuous variational formulation is used"
"to solve a given demag field problem. The parameter gamma tweaks the"
"amount of discontinuity we allow over the core boundary"

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import math
import doffinder as dff

class TruncDeMagSolver(object):
    """Base Class for truncated Demag Solvers"""
    def __init__(self,problem):
        """problem - Object from class derived from TruncDemagProblem"""
        self.problem = problem
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

class NitscheSolver(TruncDeMagSolver):
    def __init__(self,problem, degree = 1):
        """
        problem - Object from class derived from TruncDemagProblem

        degree - desired polynomial degree of basis functions
        """
        self.problem = problem
        self.degree = degree
        TruncDeMagSolver.__init__(self,problem)

    def solve(self):
        """Solve the demag problem and store the Solution"""

        V = FunctionSpace(self.problem.mesh,"CG",self.degree)

        if self.problem.mesh.topology().dim() == 1:
            Mspace = FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        #HF: I think this should always be
        #Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree,3)

        W = MixedFunctionSpace((V,V)) # for phi0 and phi1
        u0,u1 = TestFunctions(W)
        v0,v1 = TrialFunctions(W)
        sol = Function(W) 
        phi0 = Function(V)
        phi1 = Function(V)
        phitot = Function(V)
        self.phitest = Function(V)
        h = self.problem.mesh.hmin() #minimum edge length or smallest diametre of mesh
        gamma = self.problem.gamma

        #Define the magnetisation
        M = interpolate(Expression(self.problem.M),Mspace)

        N = FacetNormal(self.problem.coremesh) #computes normals on the submesh self.problem.coremesh
        dSC = self.problem.dSC #Boundary of Core
        dxV = self.problem.dxV #Vacuum
        dxC = self.problem.dxC #Core 

        #Define jumps and averages accross the boundary
        jumpu = u1('-') - u0('+')                         #create a difference symbol
                                                          #- is the inward normal
                                                          #+ is the outward pointing normal or direction.
        avggradu = (grad(u1('-')) + grad(u0('+')))*0.5
        jumpv = (v1('-') - v0('+'))
        avgv = (v1('-') + v0('+'))*0.5
        avggradv = (grad(v1('-')) + grad(v0('+')))*0.5

        #Forms for Poisson problem with Nitsche Method
        a0 = dot(grad(u0),grad(v0))*dxV #Vacuum 
        a1 = dot(grad(u1),grad(v1))*dxC #Core

        #right hand side
        f = (div(M)*v1)*dxC   #Source term in core
        f += (dot(M('-'),N('+'))*avgv )*dSC  #Prescribed outer normal derivative
        #Cross terms on the interior boundary
        c = (-dot(avggradu,N('+'))*jumpv - dot(avggradv,N('+'))*jumpu + gamma*(1/h)*jumpu*jumpv)*dSC  

        a = a0 + a1 + c

        #Dirichlet BC for phi0
        dbc = DirichletBC(W.sub(0),0.0,"on_boundary") #Need to use W as assemble thinks about W-space

        #The following arguments
        #  cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc
        #tell the assembler about the marks 0 or 1 for the cells and markers 0, 1 and 2 for the facets.
        
        A = assemble(a,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)
        F = assemble(f,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)

        #Solve for the solution
        dbc.apply(A)
        dbc.apply(F)

        #Got to here working through the code with Gabriel (HF, 23 Feb 2011)

        A.ident_zeros()
        solve(A, sol.vector(),F)

        #Seperate the mixed function and then add the parts
        solphi0,solphi1 = sol.split()
        phi0.assign(solphi0)
        phi1.assign(solphi1)

        #This might or might not be a better way to add phi1 and phi0
        #phitot = phi0 + phi1 
        
        phitot.vector()[:] = phi0.vector() + phi1.vector()
        self.phitest.assign(phitot)
        #Divide the value of phitotal by 2 on the core boundary
        BOUNDNUM = 2
        corebounddofs = dff.bounddofs(V,self.degree, self.problem.coreboundfunc, BOUNDNUM)
            
        for index,dof in enumerate(phitot.vector()):
            if index in corebounddofs:
                phitot.vector()[index] = dof*0.5

        #Get the function restricted to the magnetic core
        self.phi_core = self.restrictfunc(phitot,self.problem.coremesh)

        #Store variables for outside testing
        self.V = V
        self.phitot = phitot
        self.phi0 = phi0
        self.phi1 = phi1
        self.sol = sol
        self.M = M
        self.Mspace = Mspace
        self.gamma = gamma
        return phitot
