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


#Methods involving a linear solve have been duplicated here so that they 
class GCRSolverBenchmark(FemBemGCRSolver):
    def solve_phia(self,phia):
        """
        Solve for potential phia in the Magentic region using FEM.
        """
        V = phia.function_space()

        #Source term depends on m (code-block 1 - second line)
        #So it should be recalculated at every time step.
        f = -self.Ms*(df.div(self.m)*self.v)*df.dx  #Source term
        F = df.assemble(f)
        self.phia_bc.apply(F)

        #Solve for phia
        self.poisson_solver.solve(phia.vector(),F)
        #Replace with LU solve
        #df.solve(self.poisson_matrix_dirichlet,phia.vector(),F)
        
        return phia
    
    def solve_laplace_inside(self, function, solverparams=None):
        """Take a functions boundary data as a dirichlet BC and solve
            a laplace equation"""
        bc = df.DirichletBC(self.V, function, df.DomainBoundary())
        A = self.poisson_matrix.copy()
        b = self.laplace_zeros.copy()
        bc.apply(A, b)
        self.laplace_solver.solve(A, function.vector(), b)
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
