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

class NitscheSolver(object):
    def __init__(self,problem, degree = 1):
        """
        problem - Object from class derived from TruncDemagProblem

        degree - desired polynomial degree of basis functions
        """
        self.problem = problem
        self.degree = degree

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
