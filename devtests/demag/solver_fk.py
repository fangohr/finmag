"Solvers for the demagnetization field using the Fredkin-Koehler approach" 
#This solver does not work yet. The Neumann problem needs to be specified
#somehow. The current solution of fixing a boundary point creates an incorrect
#solution.

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import solver_base as sb
import numpy as np

class FKSolver(sb.DeMagSolver):
    """Class containing methods shared by FK solvers"""
    def __init__(self, problem, degree=1):
        super(FKSolver, self).__init__(problem, degree)
        self.phi1 = Function(self.V)
        self.phi2 = Function(self.V)

    def compute_phi1(self,M,V):
        """
        Get phi1 defined over the mesh with the point given the value 0.
        phi1 is the solution to the inhomogeneous Neumann problem.
        M is the magnetisation field.
        V is a FunctionSpace
        """
        #Define functions
        u = TrialFunction(V)
        v = TestFunction(V)
        
        #Define forms
        a = dot(grad(u),grad(v))*dx + dot(DOLFIN_EPS*u,v)*dx 
        f = (div(M)*v)*dx  #Source term
        N = FacetNormal(self.problem.mesh)
        f += (dot(M,N)*v)*ds #Neumann Condition

        A = assemble(a)
        F = assemble(f)
        """
        #Specify a the problem by changing a nonboundary DOF to 0.
        nonbounddof = self.nonboundarydof(V)

        #Specify the nonbounddof in A and F.
        F[nonbounddof] = 0.0
        #The dolfin Matrix class needs a very special input for ident
        arr = array([nonbounddof],dtype = I,order = "C")
        A.ident(arr)
        """
        #Solve for the DOFs in phi1
        solve(A,self.phi1.vector(),F)
 

    def nonboundarydof(functionspace):
        """Return a DOF not on the boundary of the mesh"""
        
        dummyBC = DirichletBC(functionSpace,0,"on_boundary")
        dofs = dummyBC.get_boundary_values()
        #Search for a DOF not in the BC
        for i in range(functionspace.dim()):
            if i not in dofs:
                return i

class FemBemFKSolver(FKSolver,sb.FemBemDeMagSolver):
    """FemBem solver for Demag Problems using the Fredkin-Koehler approach."""
    def __init__(self, problem, degree=1):
          super(FemBemFKSolver,self).__init__(problem, degree)
          self.doftionary = self.get_boundary_dof_coordinate_dict()

    def solve(self):
        self.compute_phi1(self.M, self.V)
        plot(self.phi1)
        interactive()
        self.solve_phi2_boundary(self.doftionary)
        self.solve_laplace_inside(self.phi2)
        self.phitot = self.calc_phitot(self.phi1, self.phi2)
        self.save_function(self.phitot, "fk")
        return self.phitot

    def solve_phi2_boundary(self, doftionary):
        B = self.build_BEM_matrix(doftionary)
        phi1 = self.restrict_to(self.phi1.vector().array(), doftionary.keys())
        phi2dofs = np.dot(B,phi1)
        bdofs = doftionary.keys()
        for i in range(len(bdofs)):
            self.phi2.vector()[bdofs[i]] = phi2dofs[i]
        
    def build_BEM_matrix(self, doftionary):
        n = len(doftionary)
        keys = doftionary.keys()
        BEM = np.zeros((n,n))
        for i, dof in enumerate(doftionary):
            BEM[i] = self.get_BEM_row(doftionary[dof], keys, i)
            print BEM[i]
        return BEM

    def get_BEM_row(self, R, dofs, i):
        v = self.BEMnumerator(R)
        w = self.BEMdenominator(R)
        psi = TestFunction(self.V)

        L = 1.0/(4*DOLFIN_PI)*psi*v*w*ds 

        # Second term containing solid angle (Knittel, eq. (2.52))
        # Use relation (23) - (24) in IEEE jan 2008.
        L2 = 1.0/(4*DOLFIN_PI)*v*w*ds
        L2 = (assemble(L2) - 1)
            
        bigrow = assemble(L)
        row = self.restrict_to(bigrow, dofs)
        row[i] += L2
        return row

    def BEMnumerator(self, R):
        dim = len(R)
        v = ''
        for i in range(dim-1):
            v += '"%g - x[%d]", ' % (R[i], i)
        v += '"%g - x[%d]"' % (R[-1], dim-1)

        # TODO: Do something better than this..
        exec("v = %s" % v)
        
        v = Expression(v)
        W = VectorFunctionSpace(self.problem.mesh, "CR", 1, dim=dim)
        v = interpolate(v, W)
        n = FacetNormal(self.problem.mesh)
        v = dot(v, n)
        return v

    def BEMdenominator(self, R):
        w = "pow(1.0/sqrt("
        dim = len(R-1)
        for i in range(dim-1):
            w += "(%g - x[%d])*(%g - x[%d]) + " % (R[i],i,R[i],i)
        w += "(%g - x[%d])*(%g - x[%d])), 3)" % (R[dim-1],dim -1,R[dim-1],dim-1)
        return Expression(w) 
	
          
class FKSolverTrunc(sb.TruncDeMagSolver,FKSolver):
    """FK Solver using domain truncation"""
    def __init__(self,problem, degree = 1):
        self.problem = problem
        self.degree = degree
        
    def solve(self):
        #Set up spaces,functions, measures etc.
        V = FunctionSpace(self.problem.mesh,"CG",self.degree)
        if self.problem.mesh.topology().dim() == 1:
            Mspace = FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        phi0 = Function(V)
        phi1 = Function(V)

        dxC = self.problem.dxC
        dSC = self.problem.dSC
        N = FacetNormal(self.problem.coremesh)

        #Define the magnetisation
        M = interpolate(Expression(self.problem.M),Mspace)

        ########################################
        #Solve for phi0
        ########################################
##        #A boundary point used to specify the pure neumann problem
        r = self.problem.r
        class BoundPoint(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.5 - r)

        dbc1 = DirichletBC(V, 0.0, BoundPoint())

        #Forms for Neumann Poisson Equation for phi0

        u = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(u),grad(v))*dxC
        f = (div(M)*v)*dxC  #Source term in core
        f += (dot(M,N)*v)('-')*dSC   #Neumann Conditions on edge of core

        A = assemble(a,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)
        F = assemble(f,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)

        dbc1.apply(A,F)
        A.ident_zeros()
        print A.array()
        solve(A,phi0.vector(),F)

        ########################################
        #Solve for phi1
        ########################################
        L = FunctionSpace(self.problem.mesh,"CG",self.degree)
        VD = FunctionSpace(self.problem.mesh,"DG",self.degree)
        W = MixedFunctionSpace((V,L))
        u,l = TrialFunctions(W)
        v,q = TestFunctions(W)
        sol = Function(W)

        #Forms for phi1
        a = dot(grad(u),grad(v))*dx
        f = q('-')*phi0('-')*dSC
        a += q('-')*jump(u)*dSC #Jump in solution on core boundary
        a += (l*v)('-')*dSC

        #Dirichlet BC at our approximate boundary
        dbc = DirichletBC(W.sub(0),0.0,"on_boundary")

        A = assemble(a,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)
        F = assemble(f,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)

        dbc.apply(A)
        dbc.apply(F)
        A.ident_zeros()
        solve(A, sol.vector(),F)
        solphi,sollag = sol.split()
        phi1.assign(solphi)

        phitot = Function(V)
        print phi0.vector().array()
        print phi1.vector().array()
        phitot.vector()[:] = phi0.vector() + phi1.vector()

        #Store Variables for outside testing
        self.V = V
        self.phitot = phitot
        self.phi0 = phi0
        self.phi1 = phi1
        self.sol = sol
        self.M = M
        self.Mspace = Mspace
        return phitot

if __name__ == "__main__":
     import prob_fembem_testcases as pft
     problem = pft.MagUnitCircle()
     solver = FemBemFKSolver(problem)
     phi = solver.solve()
     print phi.vector().array()
