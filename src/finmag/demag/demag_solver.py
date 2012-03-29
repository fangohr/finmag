import dolfin as df
from finmag.demag.problems.prob_base import FemBemDeMagProblem as FBProblem
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.demag.solver_gcr import FemBemGCRSolver

class Demag(object):
    def __init__(self, V, m, method="CGR"):
        self.V = V
        mesh = V.mesh()
        problem = FBProblem(mesh, m)
        if method == "FK":
            self.solver = FemBemFKSolver(problem)
        elif method == "GCR":
            self.solver = FemBemGCRSolver(problem)
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'FK'
                                    * 'GCR'""")

    def compute_field(self):
        phi = self.solver.solve()
        demag_field = df.project(-df.grad(phi), self.V)
        return demag_field.vector().array()
