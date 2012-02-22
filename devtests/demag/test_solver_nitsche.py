#A set of tests to insure that the NitscheSolver works properly

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from prob_testcases import *
from solver_nitsche import NitscheSolver

#This suite tests the solutions for the demag scalar potential function from the Nitsche Solver.
#Global Tolerance for closeness to 0.
TOL = 1.0 #Fixme This is a bad tolerance, maybe the nitsche solver can be made more precise
          #TODO the averaging by volume causes the error to increase since the surface volumes are <1,
          #this gets worse for increased dimension. So get rid of averaging and recalibrate the gammas and TOL
         
class TestNischeSolver(object):
    #Wierd that we cannot use __init__
    def setup_class(self):
        self.problem1d = MagUnitInterval()
        self.problem2d = MagUnitCircle()
        self.problem3d = MagUnitSphere()

        self.solver1d = NitscheSolver(self.problem1d)
        self.solver2d = NitscheSolver(self.problem2d)
        self.solver3d = NitscheSolver(self.problem3d)

        self.solution1d = self.solver1d.solve()
        self.solution2d = self.solver2d.solve()
        self.solution3d = self.solver3d.solve()

    def test_1d(self):
        self.probtest(self.problem1d,self.solver1d,self.solution1d)
    def test_2d(self):
        self.probtest(self.problem2d,self.solver2d,self.solution2d)
    def test_3d(self):
        self.probtest(self.problem3d,self.solver3d,self.solution3d)
        
    def test_compare_3danalytical(self):
        #Test against the known analytical solution
        soltrue = Expression("-x[0]/3.0")
        soltrue = project(soltrue,self.solver3d.V)
        l1form = abs(self.solution3d - soltrue)*self.problem3d.dxC
        L1error = assemble(l1form, cell_domains = self.problem3d.corefunc)
        print "3d Analtical solution comparison L1error =",L1error
        assert L1error < TOL,"L1 Error in 3d computed solution from the analytical solution, %g is not less than the Tolerance %g"%(L1error,TOL)
        
    def probtest(self,problem,solver,solution):
        self.dbc_test(problem,solution)
        self.continuity_test(problem,solver,solution)
        self.normalderivativejump_test(problem,solver,solution)

    def dbc_test(self,problem, solution):
        #1 Test dirichlet boundary condition on outside
        one = interpolate(Constant(1),solution.function_space())
        a = abs(solution)*ds
        c = one*ds
        L1error = assemble(a)/assemble(c)
        print "dbc_test: L1error=",L1error
        errmess = "Error in Nitsche Solver with problem " + problem.desc() + \
        "outer dirichlet BC condition not satisfied, average solution boundary integral is %g"%(L1error)
        assert L1error < TOL,errmess 

    def continuity_test(self,problem,solver,solution):
        #2 Test Continuity accross the interior boundary
        dSC = problem.dSC
        one = interpolate(Constant(1),solution.function_space())
        jumpphi = solver.phi1('-') - solver.phi0('+')
        a1 = abs(jumpphi)*dSC
        a2 = abs(jump(solution))*dSC
        c = one('-')*dSC
        L1error1 = assemble(a1,interior_facet_domains = problem.coreboundfunc)#/assemble(c,interior_facet_domains = problem.coreboundfunc)
        L1error2 = assemble(a2,interior_facet_domains = problem.coreboundfunc)#/assemble(c,interior_facet_domains = problem.coreboundfunc)
        print "continuity_test: L1error1=",L1error1
        print "continuity_test: L1error2=",L1error2
        assert L1error1 < TOL,"Error in Nitsche Solver with problem" + problem.desc() + " continuity accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g, L1error = %g"%(TOL,L1error1)
        assert L1error2 < TOL,"Error in Nitsche Solver with 1d problem, continuity accross magnetic core boundary not satisfied for phi total \
                               TOL = %g, L1error = %g"%(TOL, L1error2)

    def normalderivativejump_test(self,problem,solver,solution):
        #3 Test jump in normal derivative across the interior boundary
        dSC = problem.dSC
        N = FacetNormal(problem.coremesh)
        M = solver.M
        
        one = interpolate(Constant(1),solution.function_space())
        jumpphinor = dot(grad(solver.phi1('-') - solver.phi0('+')),N('+'))
        a1 = abs(jumpphinor - dot(M,N)('-'))*dSC
        a2 = abs(dot(jump(grad(solution)),N('+')) - dot(M,N)('+'))*dSC
        c = one('-')*dSC
        L1error1 = assemble(a1,interior_facet_domains = problem.coreboundfunc )#/assemble(c,interior_facet_domains = problem.coreboundfunc)
        L1error2 = assemble(a2,interior_facet_domains = problem.coreboundfunc)#/assemble(c,interior_facet_domains = problem.coreboundfunc)
        print "normalderivativejump_test: L1error1=",L1error1
        print "normalderivativejump_test: L1error2=",L1error2
        assert L1error1 < TOL,"Error in Nitsche Solver with 1d problem, normal derivative jump accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g, L1error = %g"%(TOL,L1error1)
        ##This test is failed since the total function is twice as high as it should be on the boundary. When this is solved I expect this to be passed.
        assert L1error2 < TOL,"Error in Nitsche Solver with 1d problem, normal derivative jump accross magnetic core boundary not satisfied for phi total \
                               TOL = %g, L1error = %g"%(TOL,L1error2)
            
if __name__ == "__main__":
    t = TestNischeSolver()
    t.setup_class()
    print "* Doing test 1d ==========="
    t.test_1d()
    print "gamma = ", t.problem1d.gamma
    print
    print "* Doing test 2d ==========="
    t.test_2d()
    print "gamma = ", t.problem2d.gamma
    print
    print "* Doing test 3d ==========="
    t.test_3d()
    print "gamma = ", t.problem3d.gamma
    print
    print "* Doing Analytical comparison ======="
    t.test_compare_3danalytical()
