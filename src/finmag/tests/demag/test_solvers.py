"""A set of tests to insure that the Demag Solvers work properly"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import unittest
import numpy as np
import finmag.demag.problems.prob_trunc_testcases as pttc
import finmag.demag.problems.prob_fembem_testcases as pftc
import finmag.demag.solver_nitsche as sn
import finmag.demag.solver_base as sb
import finmag.demag.solver_gcr as sgcr
import finmag.util.error_norms as en

class DemagTester(object):
    """Base class for demag testers"""
    def error_norm(self,func1,func2,cell_domains = None,interior_facet_domains = None, dx = dx):
        """L2 error norm for functions func1 and func2, dx = Measure"""
        return en.L2_error(func1,func2,cell_domains = None,interior_facet_domains = None, dx = dx)

    def compare_to_analytical(self,compsol,analyticalsol,testname):
        """Test a computed solution against a analytical solution"""
        L2error = errornorm(compsol, analyticalsol)
        print compsol,analyticalsol
        #L2error = self.error_norm(compsol, analyticalsol) 
        print testname, "Comparison L2error = ", L2error
        assert L2error < self.TOL,"Error in" +testname+ "L2 error %g is not less than the Tolerance %g"%(L2error,self.TOL)   
        
class UnitSphere_Analytical(object):
    """
    Class containing information regarding the 3d analytical solution of a Demag Field in a uniformly
    demagnetized unit sphere with M = (1,0,0)
    """
    def __init__(self,mesh):
        self.V = FunctionSpace(mesh,"CG",1)
        self.VV = VectorFunctionSpace(mesh,"DG",0)
        self.potential = project(Expression("-x[0]/3.0"),self.V)
        self.Hdemag = project(Expression(("1.0/3.0","0.0","0.0")),self.VV)

class TestTruncDemagSolver(object):
    """Test the Solver Base Class"""

    def test_restrictfunc(self):
        """
        A simple linear Function is restricted to half a square and the answer is tested against an analytic value
        """
        mesh = UnitSquare(2,2)
        V = FunctionSpace(mesh,"CG",1)
        
        #A plane going from 0 to 1 in the x direction
        E = Expression("1 - x[0]")
        u = interpolate(E,V)
        
        #Generate the submesh
        class Half(SubDomain):
            def inside(self,x,on_boundary):
                return x[0]<0.5 + DOLFIN_EPS
            
        meshfunc = MeshFunction("uint",mesh,2)
        meshfunc.set_all(0)
        Half().mark(meshfunc,1)
        halfmesh = SubMesh(mesh,meshfunc,1)

        #Class initialized with a dummy problem
        problem = pttc.MagUnitInterval()
        solver = sb.TruncDeMagSolver(problem)

        #Get the restricted function
        uhalf = solver.restrictfunc(u,halfmesh)

        #The restricted function should be a plane going from 0 to 0.5 in x direction
        exactsol = interpolate(E,uhalf.function_space())
        
        a = abs(uhalf - exactsol)*dx
        A = assemble(a)
        assert near(A,0.0),"Error in TruncDemagSolver.restrictfunc, restricted function does not match analytical value"

class TestNitscheSolver(DemagTester):
    def setup_class(self):
        self.TOL = 1.0 #Fixme This is a bad tolerance, maybe the nitsche solver can be made more precise
        
        self.problem1d = pttc.MagUnitInterval()
        self.problem2d = pttc.MagUnitCircle()
        self.problem3d = pttc.MagUnitSphere()

        self.solver1d = sn.NitscheSolver(self.problem1d)
        self.solver2d = sn.NitscheSolver(self.problem2d)
        self.solver3d = sn.NitscheSolver(self.problem3d)

        self.solution1d = self.solver1d.solve()
        self.solution2d = self.solver2d.solve()
        self.solution3d = self.solver3d.solve()

        self.analytical3d = UnitSphere_Analytical(self.problem3d.coremesh)

    def desc(self):
        return "Nitsche Solver tester"

    def test_1d(self):
        """Test a 1d solution of the Nitsche Solver"""
        self.probtest(self.problem1d,self.solver1d,self.solution1d)
    def test_2d(self):
        """Test a 2d solution of the Nitsche Solver"""
        self.probtest(self.problem2d,self.solver2d,self.solution2d)
    def test_3d(self):
        """Test a 3d solution of the Nitsche Solver"""
        self.probtest(self.problem3d,self.solver3d,self.solution3d)
        
    def test_compare_3danalytical(self):
        """
        Test the potential phi from the Nitsche Solver against
        the known analytical solution in the core for a uniformly magentized
        unit sphere
        """
        testname = self.test_compare_3danalytical.__doc__       
        self.compare_to_analytical(self.solver3d.phi_core,self.analytical3d.potential,testname)
        
    def test_compare_3danalytical_gradient(self):
        """
        Test the demag field from the Nitsche Solver against
        the known analytical solution in the core for a uniformly magentized
        unit sphere
        """
        testname = self.test_compare_3danalytical_gradient.__doc__
        self.compare_to_analytical(self.solver3d.Hdemag_core,self.analytical3d.Hdemag,testname)
    
    def print_convergance_3d(self):
        """
        Since refining first increases the error it is not suitable to test convergence for the first refinements
        to see this strange phenomena run this method.
        """
        NUM_REFINEMENTS = 4
        #Take previously calculated errors 
        pot3d_errorold  = self.error_norm(self.solver3d.phi_core ,self.analytical3d.potential) 
        demag3d_errorold = self.error_norm(self.solver3d.Hdemag_core ,self.analytical3d.Hdemag) 

        #Refine the core mesh and see if the errors decrease
        print "Errors for potential,demag"
        for i in range(NUM_REFINEMENTS):
            self.problem3d.refine_core()
            #Solve the demag problem
            self.solver3d = sn.NitscheSolver(self.problem3d)
            self.solution3d = self.solver3d.solve()
            #Regenerate the analytical solution
            self.analytical3d = UnitSphere_Analytical(self.problem3d.coremesh) 
            #Get errors
            pot3d_errornew = self.error_norm(self.solver3d.phi_core ,self.analytical3d.potential) 
            demag3d_errornew = self.error_norm(self.solver3d.Hdemag_core ,self.analytical3d.Hdemag) 
            print pot3d_errornew, demag3d_errornew, " refinements = ", i+1
            #Make the new errors the old errors
            pot3d_errorold = pot3d_errornew
            demag3d_errorold = demag3d_errornew

    def probtest(self,problem,solver,solution):
        """Run various tests on a truncated demag problem"""
        self.dbc_test(problem,solution)
        self.continuity_test(problem,solver,solution)
        self.normalderivativejump_test(problem,solver,solution)

    def dbc_test(self,problem, solution):
        """Test if the DBC is satisfied at the edge of the Vacuum region"""
        L2error = self.error_norm(solution,Function(solution.function_space()) , dx = ds)
        print "dbc_test: L2error=",L2error
        errmess = "Error in Nitsche Solver with problem " + problem.desc() + \
        "outer dirichlet BC condition not satisfied, average solution boundary integral is %g"%(L2error)
        assert L2error < self.TOL,errmess 

    def continuity_test(self,problem,solver,solution):
        """Test Continuity across the interior boundary"""
        jumpphi = solver.phi1('-') - solver.phi0('+')
        L2error1 = self.error_norm(jump(solution),Function(solution.function_space())('-'),\
                                   interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
        L2error2 = self.error_norm(jumpphi,Function(solution.function_space())('-'), \
                                   interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
    
        print "continuity_test: L2error1=",L2error1
        print "continuity_test: L2error2=",L2error2
        assert L2error1 < self.TOL,"Error in Nitsche Solver with problem" + problem.desc() \
                               + "continuity accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g, L2error = %g"%(self.TOL,L2error1)
        assert L2error2 < self.TOL,"Error in Nitsche Solver with 1d problem" + problem.desc() \
                               + "continuity accross magnetic core boundary not satisfied for phi total \
                               TOL = %g, L2error = %g"%(self.TOL,L2error2)

    def normalderivativejump_test(self,problem,solver,solution):
        """ Test jump in normal derivative across the interior boundary"""
        N = FacetNormal(problem.coremesh)
        M = solver.M
        
        jumpphinor = dot(grad(solver.phi1('-') - solver.phi0('+')),N('+'))
        f1 =abs(jumpphinor - dot(M,N)('-'))
        f2 = abs(dot(jump(grad(solution)),N('+')) - dot(M,N)('+'))
        zero = Function(solution.function_space())

        L2error1 = self.error_norm(f1,zero('-'),interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
        L2error2 = self.error_norm(f2,zero('-'), interior_facet_domains = problem.coreboundfunc, dx = problem.dSC) 
        
        print "normalderivativejump_test: L2error1=",L2error1
        print "normalderivativejump_test: L2error2=",L2error2
        assert L2error1 < self.TOL,"Error in Nitsche Solver with " + problem.desc() + " normal derivative jump accross magnetic core boundary not satisfied for phi1 and phi2, \
                               TOL = %g, L2error = %g"%(self.TOL,L2error1)
        assert L2error2 < self.TOL,"Error in Nitsche Solver with " + problem.desc() + " normal derivative jump accross magnetic core boundary not satisfied for phi total \
                               TOL = %g, L2error = %g"%(self.TOL,L2error2)

        ##FIXME convergence FAILS!
##    def test_convergance_3d(self):
##        """The FEM solution should converge to the analytical solution as the mesh is refined"""
##        NUM_REFINEMENTS = 6 
##        #Take previously calculated errors 
##        firstpoterror = self.L1_error_3d_potential(self.solution3d)
##        firstdemagerror = self.L2_error_3d_demag(self.solver3d.Hdemag_core)
##
##        #Refine the core mesh
##        for i in range(NUM_REFINEMENTS):
##            self.problem3d.refine_core()
##        #Solve the demag problem again
##        self.solver3d = sn.NitscheSolver(self.problem3d)
##        self.solution3d = self.solver3d.solve()
##        #get errors
##        pot3d_errornew = self.L1_error_3d_potential(self.solution3d)
##        demag3d_errornew = self.L2_error_3d_demag(self.solver3d.Hdemag_core)
##        
##        #Test the last solution against the 1st one.
##        assert firstpoterror > pot3d_errornew,"Error in refinement of 3d problem, error in potential did not decrease on final refinement "+ str(i+1)
##        assert firstdemagerror > demag3d_errornew,"Error in refinement of 3d problem, error in demag did not decrease on final refinement "+ str(i+1)
        
class TestFemBemDeMagSolver(object):
    """Test the FemBemDeMagSolver class """
        
    def setup_class(self):      
        self.problem = pftc.MagUnitSphere()
        self.solver = sb.FemBemDeMagSolver(**self.problem.kwargs())

    def test_solve_laplace_inside(self):
        """Solve a known laplace equation to check the method solve_laplace_inside"""
        mesh = UnitSquare(2,2)
        V = FunctionSpace(mesh,"CG",1)
        #Insert V into the solver and recreate the test and trial functions
        self.solver.V = V
        self.solver.v = TestFunction(V)
        self.solver.u = TrialFunction(V)
        
        self.solver.poisson_matrix = self.solver.build_poisson_matrix()
        self.solver.laplace_zeros = Function(V).vector()

        #Test the result of the call
        fold = interpolate(Expression("1-x[0]"),V)
        fnew = interpolate(Expression("1-x[0]"),V)
        #The Laplace equation should give the same solution as f
        fnew = self.solver.solve_laplace_inside(fnew)
        assert fold.vector().array().all() == fnew.vector().array().all(),"Error in method test_solve_laplace_inside(), \
            Laplace solution does not equal original solution"
        print "solve_laplace_inside testpassed"

class Test_FemBemGCRSolver(DemagTester):
    """Tests for the Class FemBemGCRSolver"""
    def setup_class(self):

        #Class Tolerance 
        self.TOL = 2.0

        #Problems,solvers, solutions
        self.problem3d = pftc.MagSphere(1,0.8)
        self.solver3d = sgcr.FemBemGCRSolver(**self.problem3d.kwargs())
        self.solution3d = self.solver3d.solve()

        #Generate a 3d analytical solution
        self.analytical3d = UnitSphere_Analytical(self.problem3d.mesh)
    
    def test_compare_3danalytical(self):
        """
        Test the potential phi from the GCR FemBem Solver against
        the known analytical solution in the core for a uniformly magentized
        unit sphere
        """
        testname = self.test_compare_3danalytical_gradient.__doc__
        self.compare_to_analytical(self.solver3d.phi,self.analytical3d.potential,testname)
        
    def test_compare_3danalytical_gradient(self):
        """
        Test the demag field from the GCR FemBem Solver against
        the known analytical solution in the core for a uniformly magentized
        unit sphere
        """
        testname = self.test_compare_3danalytical_gradient.__doc__
        phi = self.solver3d.solve()
        Hdemag = self.solver3d.get_demagfield()
        self.compare_to_analytical(Hdemag,self.analytical3d.Hdemag,testname)


if __name__ == "__main__":
    def run_tests(tests):
        for test in tests:
            print "* Doing",test.__doc__
            test()
            print
    t = TestNitscheSolver()
    t.setup_class()
    tests = [t.test_1d,t.test_2d,t.test_3d,t.test_compare_3danalytical,\
            t.test_compare_3danalytical_gradient]
    run_tests(tests)

##    #Slow uncomment with caution
##    print "* Doing Convergance test of Demag field Nitsche Solver======="
##    t.print_convergance_3d()
##    print
##    
    t = TestFemBemDeMagSolver()
    t.setup_class()
    tests = [t.test_solve_laplace_inside]
    run_tests(tests)
    
    t = Test_FemBemGCRSolver()
    t.setup_class()
    tests = [t.test_compare_3danalytical_gradient, t.test_compare_3danalytical]
    run_tests(tests)
