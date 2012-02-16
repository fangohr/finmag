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

class NitscheSolver(object):
    def __init__(self,problem,gamma = 1.0, degree = 1):
        self.gamma = gamma
        self.problem = problem
        self.degree = degree

    def solve(self):
        #Solve the demag problem and store the Solution
        V = FunctionSpace(self.problem.mesh,"CG",self.degree)
        W = MixedFunctionSpace((V,V))
        u0,u1 = TestFunctions(W)
        v0,v1 = TrialFunctions(W)
        self.sol = Function(W) 
        self.phi0 = Function(V)
        self.phi1 = Function(V)
        self.phitot = Function(V)
        h = self.problem.mesh.hmin()

        #Define the magnetisation
        M = interpolate(Expression(self.problem.M),V)

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
        solve(A, self.sol.vector(),F)

        #Seperate the mixed function and then add the parts
        solphi0,solphi1 = self.sol.split()
        self.phi0.assign(solphi0)
        self.phi1.assign(solphi1)

        #This might or might not be a better way to add phi1 and phi0
        #phitot = phi0 + phi1 

        self.phitot.vector()[:] = self.phi0.vector() + self.phi1.vector()
        return self.phitot
        
#Move this to a test suite
##
##        ###############################
##        #Test the Solution
##        ###############################
##        #1 Test dirichlet boundary condition on outside
##        one = interpolate(Constant(1),V)
##        a = abs(phitot)*ds
##        c = one*ds
##        L1error = assemble(a)/assemble(c)
##        print "Average Error in Outer Dirichlet BC", L1error
##
##        #2 Test Continuity accross the interior boundary
##        one = interpolate(Constant(1),V)
##        jumpphi = phi1('-') - phi0('+')
##        a1 = abs(jumpphi)*dSC
##        a2 = abs(jump(phitot))*dSC
##        c = one('-')*dSC
##        L1error1 = assemble(a1,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
##        L1error2 = assemble(a2,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
##        print "Average Error in continuity in inner boundary for phi1 and phi2", L1error1
##        print "Average Error in continuity in inner boundary for phi total", L1error2
##
##
##        #3 Test jump in normal derivative across the interior boundary
##        one = interpolate(Constant(1),V)
##        jumpphinor = dot(grad(phi1('-') - phi0('+')),N('+'))
##        a1 = abs(jumpphinor - dot(M,N)('-'))*dSC
##        a2 = abs(dot(jump(grad(phitot)),N('+')) - dot(M,N)('+'))*dSC
##        c = one('-')*dSC
##        L1error1 = assemble(a1,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
##        L1error2 = assemble(a2,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
##        print "Average Error in jump in normal derivative in inner boundary for phi1 and phi2", L1error1
##        print "Average Error in jump in normal derivative in inner boundary for phi total", L1error1
##
