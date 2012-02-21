"Nitsche's Method for a discontinuous variational formulation is used"
"to solve a given demag field problem. The parameter gamma tweaks the"
"amount of discontinuity we allow over the core boundary"

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from interiorboundary import *
import math
import numpy as np
import pylab as pl
import doffinder as dff

class NitscheSolver(object):
    def __init__(self,problem,gamma = 1.0, degree = 1):
        self.gamma = gamma
        self.problem = problem
        self.degree = degree

    def solve(self):
        #Solve the demag problem and store the Solution
        V = FunctionSpace(self.problem.mesh,"CG",self.degree)
        if self.problem.mesh.topology().dim() == 1:
            Mspace = FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        W = MixedFunctionSpace((V,V))
        u0,u1 = TestFunctions(W)
        v0,v1 = TrialFunctions(W)
        sol = Function(W) 
        phi0 = Function(V)
        phi1 = Function(V)
        phitot = Function(V)
        self.phitest = Function(V)
        h = self.problem.mesh.hmin()

        #Define the magnetisation
        M = interpolate(Expression(self.problem.M),Mspace)

        N = FacetNormal(self.problem.coremesh)
        dSC = self.problem.dSC #Boundary of Core
        dxV = self.problem.dxV #Vacuum
        dxC = self.problem.dxC #Core 

        #Define jumps and averages accross the boundary
        jumpu = u1('-') - u0('+')
        avggradu = (grad(u1('-')) + grad(u0('+')))*0.5
        jumpv = (v1('-') - v0('+'))
        avgv = (v1('-') + v0('+'))*0.5
        avggradv = (grad(v1('-')) + grad(v0('+')))*0.5

        #Forms for Poisson with Nitsche Method
        a0 = dot(grad(u0),grad(v0))*dxV #Vacuum 
        a1 = dot(grad(u1),grad(v1))*dxC #Core

        #right hand side
        f = (div(M)*v1)*dxC   #Source term in core
        f += (dot(M('-'),N('+'))*avgv )*dSC  #Presribed outer normal derivative
        #Cross terms on the interior boundary
        c = (-dot(avggradu,N('+'))*jumpv - dot(avggradv,N('+'))*jumpu + self.gamma*(1/h)*jumpu*jumpv)*dSC  

        a = a0 + a1 + c

        #Dirichlet BC at our approximate boundary
        dbc = DirichletBC(W.sub(0),0.0,"on_boundary")

        A = assemble(a,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)
        F = assemble(f,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)

        #Solve for the solution
        dbc.apply(A)
        dbc.apply(F)
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
        return phitot
