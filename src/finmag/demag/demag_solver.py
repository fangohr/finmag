import dolfin as df
from finmag.demag.problems.prob_base import FemBemDeMagProblem as FBProblem
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.util.timings import timings

class Demag(object):
    def __init__(self, V, m, Ms, method="CGR"):
        timings.start("Demag-init")
        self.V = V
        mesh = V.mesh()
        self.Ms = Ms
        problem = FBProblem(mesh, m)
        if method == "FK":
            self.solver = FemBemFKSolver(problem)
        elif method == "GCR":
            self.solver = FemBemGCRSolver(problem)
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'FK'
                                    * 'GCR'""")

        timings.stop("Demag-init")

    def compute_field(self):
        timings.start("Demag-computephi")
        phi = self.solver.solve()
        timings.stop("Demag-computephi")
        timings.start("Demag-computefield")
        demag_field = df.project(-df.grad(phi), self.V)
        H = demag_field.vector().array()*self.Ms
        timings.stop("Demag-computefield")
        return H
