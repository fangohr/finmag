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
    phi1TOL
        Relative tolerance for the first linear solve
    phi2TOL
        Relative tolerance for the second linear solve    
    """
    def __init__(self,solver = "FK", phi1TOL = 1e-7, phi2TOL = 1e-7):
        self.in_jacobian = False
        log.info("Creating Demag object with " + solver + " solver.")
        if solver in ["FK","GCR"]:
            self.solver = solver
        else:
            raise Exception("Only 'FK', and 'GCR' are possible solver values") 
        self.phi1TOL = phi1TOL
        self.phi2TOL = phi2TOL

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
       # timings.startnext("create-demag-problem")
        problem = FemBemDeMagProblem(S3.mesh(), m, Ms)
        if self.solver == "FK":
            self.demag = FemBemFKSolver(problem, unit_length=unit_length,
                                         phi1TOL = self.phi1TOL,
                                         phi2TOL = self.phi2TOL)
        elif self.solver == "GCR":
            self.demag = FemBemGCRSolver(problem, unit_length=unit_length,
                                         phiaTOL = self.phi1TOL,
                                         phibTOL = self.phi2TOL)
            
        #timings.startnext("Solve-demag-problem")
        self.demag.solve()

    def compute_field(self):
        return self.demag.compute_field()

    def compute_energy(self):
        return self.demag.compute_energy()

    def compute_potential(self):
        return self.demag.phi

if __name__ == "__main__":
    #Generate a plot for a simple Demag problem
    test = "GCR"
    from finmag.demag.problems import prob_fembem_testcases as pft
    problem = pft.MagSphere20()
    mesh = problem.mesh
    Ms = problem.Ms
    m = problem.m
    V = problem.V

    if test == "GCR":
        demag = Demag("GCR")
        demag.setup(V,m,Ms,unit_length = 1)
        
    elif test == "FK":
        demag = Demag("FK")
        demag.setup(V,m,Ms,unit_length = 1)

    print timings
    df.plot(demag.compute_potential())
    df.interactive()    
