"""Test the runtime of demag linear solver combinations for a given mesh and
initial magnetisation"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.solver_fk import FemBemFKSolver
import finmag.demag.problems.prob_base as pb

from dolfin import *
#Importing this module activates the benchmark tests
from solver_benchmark import *

class GCRSolverBenchmark(FemBemGCRSolver):
    def __init__(self,problem,degree = 1):
        FemBemGCRSolver.__init__(self,problem,degree = 1)
        self.countdown = False
    
    def linsolve_phia(self,A,F):
        """Linear solve for phia"""
        print "\n GCR Solver First linear solve Benchmark \n"
        solve(A,self.phia.vector(),F,solver_parameters = self.phiasolverparams,\
              benchmark = True)
        
    def linsolve_laplace_inside(self,function,laplace_A,solverparams = None):
        """Linear solve for laplace_inside"""
        print "\n GCR Solver Second linear solve Benchmark \n"
        solve(laplace_A,function.vector(),self.laplace_f,\
                  solver_parameters = solverparams, benchmark = True)
        return function
    
def test_linalgtimes(mesh,M,solver):
    """Test the runtime of all possible solver combinations for the demag field"""
    problem = pb.FemBemDeMagProblem(mesh,M)
    if solver == "FK":
        solver = FemBemFKSolver(problem)
    elif solver == "GCR":
        solver = GCRSolverBenchmark(problem)
    else:
        raise Exception("Only 'FK, and 'GCR' solver values possible")
    solver.solve()

if __name__ == "__main__":
    #As a default benchmark GCR solver on a unit sphere mesh
    import finmag.demag.problems.prob_fembem_testcases as pft
    problem = pft.MagSphere(10,1.5)
    print "Mesh size ",problem.mesh.num_vertices()
    test_linalgtimes(problem.mesh,problem.M,"GCR")
