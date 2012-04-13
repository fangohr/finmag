from dolfin import *
from finmag.demag import solver_base as sb
import numpy as np
import progressbar as pb


##class FKSolver(sb.DeMagSolver):
##    """Class containing methods shared by FK solvers"""
##    def __init__(self, problem, degree=1):
##        super(FKSolver, self).__init__(problem, degree)


class FKSolverTrunc(sb.TruncDeMagSolver):
    """FK Solver using domain truncation"""
    ###Only partially implemented at the moment
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
