import dolfin as df
from finmag.energies.demag.solver_fk import FemBemFKSolver
from finmag.energies.demag.solver_gcr import FemBemGCRSolver
from finmag.util.timings import timings

class Demag(object):
    def __init__(self, V, m, Ms, method="FK",bench = False):
        timings.start("Demag-init")
        self.V = V
        mesh = V.mesh()
        self.Ms = Ms
        timings.startnext("Demag-init-FemBemConstructorCall")
        if method == "FK":
            self.solver = FemBemFKSolver(mesh,m, Ms = Ms, bench = bench)
        elif method == "GCR":
            self.solver = FemBemGCRSolver(mesh,m, Ms = Ms,bench = bench)
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'FK'
                                    * 'GCR'""")
        self.method = method
        timings.stop("Demag-init-FemBemConstructorCall")

    def compute_field(self):
        #timings.start("Demag-computephi")
        phi = self.solver.solve()
        #timings.startnext("Demag-computefield")
        timings.start("Project demag field")
        demag_field = df.project(-df.grad(phi), self.V).vector().array()
        timings.stop("Project demag field")
        if self.method == "GCR":
            demag_field *= self.Ms
        #timings.stop("Demag-computefield")
        return demag_field
