#A set of tests to insure that the NitscheSolver works properly
#So far just 1-d 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from prob_testcases import *
from solver_nitsche import NitscheSolver

#Global Tolerance for closeness to 0.
TOL = e-6
class Tester(object):
    def __init__(self):
        self.problem1d = MagUnitInterval()
        self.solver = NitscheSolver(problem)
        self.solution1d = self.solver.solve()

    def test_dbc(self):
        #1 Test dirichlet boundary condition on outside
        one = interpolate(Constant(1),self.problem1d.V)
        a = abs(self.solution1d)*ds
        c = one*ds
        L1error = assemble(a)/assemble(c)
        assert L1error < TOL,"Error in Nitsche Solver with 1d problem, outer dirichlet BC condition not satisfied" 

    def test_continuity(self):
        #2 Test Continuity accross the interior boundary
        dSC = self.problem1d.dSC
        one = interpolate(Constant(1),self.problem1d.V)
        jumpphi = self.solver.phi1('-') - self.solver.phi0('+')
        a1 = abs(jumpphi)*dSC
        a2 = abs(jump(self.solution1d))*dSC
        c = one('-')*dSC
        L1error1 = assemble(a1,interior_facet_domains = self.problem1d.coreboundfunc)/assemble(c,interior_facet_domains = self.problem1d.coreboundfunc)
        L1error2 = assemble(a2,interior_facet_domains = self.problem1d.coreboundfunc)/assemble(c,interior_facet_domains = self.problem1d.coreboundfunc)
        assert L1error1 < TOL,"Error in Nitsche Solver with 1d problem, continuity accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g"%(TOL)
        assert L1error2 < TOL,"Error in Nitsche Solver with 1d problem, continuity accross magnetic core boundary not satisfied for phi total \
                               TOL = %g"%(TOL)

    def test_normalderivativejump(self):
        #3 Test jump in normal derivative across the interior boundary
        dSC = self.problem1d.dSC
        N = FacetNormal(self.problem1d.coremesh)
        M = self.solver.N
        
        one = interpolate(Constant(1),self.problem1d.V)
        jumpphinor = dot(grad(self.solver.phi1('-') - self.solver.phi0('+')),N('+'))
        a1 = abs(jumpphinor - dot(M,N)('-'))*dSC
        a2 = abs(dot(jump(grad(phitot)),N('+')) - dot(M,N)('+'))*dSC
        c = one('-')*dSC
        L1error1 = assemble(a1,interior_facet_domains = self.problem1d.coreboundfunc )/assemble(c,interior_facet_domains = self.problem1d.coreboundfunc)
        L1error2 = assemble(a2,interior_facet_domains = self.problem1d.coreboundfunc)/assemble(c,interior_facet_domains = self.problem1d.coreboundfunc)
        assert L1error1 < TOL,"Error in Nitsche Solver with 1d problem, normal derivative jump accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g"%(TOL)
        assert L1error2 < TOL,"Error in Nitsche Solver with 1d problem, normal derivative jump accross magnetic core boundary not satisfied for phi total \
                               TOL = %g"%(TOL)

