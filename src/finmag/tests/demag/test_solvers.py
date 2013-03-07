"""A set of tests to insure that the Demag Solvers work properly"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from finmag.util.versions import get_version_dolfin
from dolfin import *
import pytest
import problems.prob_fembem_testcases as pftc
import finmag.energies.demag.solver_base as sb
import finmag.energies.demag.solver_gcr as sgcr


def L2_error(f1,f2,cell_domains = None,interior_facet_domains = None, dx = dx):
    """L2 error norm for functions f1 and f2, dx = Measure"""
    Eform = inner(f1-f2,f1-f2)*dx
    E = assemble(Eform, cell_domains =  cell_domains, interior_facet_domains =interior_facet_domains)
    return sqrt(E)


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


class DemagTester(object):
    """Base class for demag testers"""
    def error_norm(self,func1,func2,cell_domains = None,interior_facet_domains = None, dx = dx):
        """L2 error norm for functions func1 and func2, dx = Measure"""
        return L2_error(func1,func2,cell_domains = None,interior_facet_domains = None, dx = dx)

    def compare_to_analytical(self,compsol,analyticalsol,testname):
        """Test a computed solution against a analytical solution"""
        L2error = errornorm(compsol, analyticalsol)
        print compsol,analyticalsol
        #L2error = self.error_norm(compsol, analyticalsol) 
        print testname, "Comparison L2error = ", L2error
        assert L2error < self.TOL,"Error in" +testname+ "L2 error %g is not less than the Tolerance %g"%(L2error,self.TOL)   


class TestFemBemDeMagSolver(object):
    """Test the FemBemDeMagSolver class """
        
    def setup_class(self):      
        self.problem = pftc.MagUnitSphere()
        self.solver = sb.FemBemDeMagSolver(mesh=self.problem.mesh,
                Ms=self.problem.Ms, m=self.problem.M)

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


@pytest.mark.skipif("get_version_dolfin()[:3] != '1.0'")
class Test_FemBemGCRSolver(DemagTester):
    """Tests for the Class FemBemGCRSolver"""
    def setup_class(self):

        #Class Tolerance 
        self.TOL = 2.0

        #Problems,solvers, solutions
        self.problem3d = pftc.MagSphereBase(0.8, 1)
        self.solver3d = sgcr.FemBemGCRSolver(mesh=self.problem3d.mesh,
                Ms=self.problem3d.Ms, m=self.problem3d.M)
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
