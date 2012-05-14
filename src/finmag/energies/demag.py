import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from energy_base import EnergyBase
from finmag.demag.problems.prob_base import FemBemDeMagProblem
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.demag.solver_gcr import FemBemGCRSolver

log = logging.getLogger("finmag")

class Demag(EnergyBase):
    """
    A wrapper for the demag solvers that also implements the functionality of
    an energy class.

    solver
        type of demag solver GCR or FK
    """
    def __init__(self,solver = "FK"):
        self.in_jacobian = False
        log.info("Creating Demag object with " + solver + " solver.")
        if solver in ["FK","GCR"]:
            self.solver = solver
        else:
            raise Exception("Only 'FK', and 'GCR' are possible solver values") 

    def setup(self, S3, m, Ms, unit_length):
        """S3
                dolfin VectorFunctionSpace
            m
                the Dolfin object representing the (unit) magnetisation
            Ms
                the saturation magnetisation
            
            unit_length
                The scale of the mesh, default is 1.
        """
        timings.start("create-demag-problem")
        problem = FemBemDeMagProblem(S3.mesh(), m, Ms)
        if self.solver == "FK":
            self.demag = FemBemFKSolver(problem, unit_length=unit_length)
        elif self.solver == "GCR":
            self.demag = FemBemGCRSolver(problem, unit_length=unit_length)
            
        timings.startnext("Solve-demag-problem")
        self.demag.solve()

    def compute_field(self):
        return self.demag.compute_field()

    def compute_energy(self):
        return self.demag.compute_energy()

    def compute_potential(self):
        return self.demag.phi

if __name__ == "__main__":
    test == "GCR"
    if test == "GCR":
        mesh = df.UnitSphere(4)
        Ms = 1
        V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
        m = df.project(df.Constant((1, 0, 0)), V)
        gcrdemag = Demag("GCR")
        gcrdemag.setup(V,m,Ms,unit_length = 1)
        df.plot(gcrdemag.compute_potential())
        df.interactive()
        print timings
        
    elif test == "FK":
        #TODO Add a test here.
        pass
