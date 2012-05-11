import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from energy_base import EnergyBase
from finmag.demag.problems.prob_base import FemBemDeMagProblem
from finmag.demag.solver_fk import FemBemFKSolver
from finmag.demag.solver_fk_test import SimpleFKSolver

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
        log.info("Creating Demag object with solver type" + solver)
        if solver in ["FK","GCR"]:
            self.solver = solver
        else:
            raise Exception("Only 'FK', and 'GCR' are possible solver values") 

    def setup(self, S3, m, Ms, unit_length):
        """S3
                Somebody please explain what I am.
            m
                the Dolfin object representing the (unit) magnetisation
            Ms
                the saturation magnetisation
            
            unit_length
                The scale of the mesh, default is 1.
        """
        timings.start("create-demag-problem")
        problem = FemBemDeMagProblem(S3.mesh(), m,Ms)
        if solver == "FK":
            self.demag = FemBemFKSolver(problem, unit_length=unit_length)
        else:
            raise Exception("CGR not implemented yet")
        timings.stop("create-demag-problem")

    def compute_field(self):
        return self.demag.compute_field()

    def compute_energy(self):
        return self.demag.compute_energy()


#GB make a test of this solver later
##        elif solver == "SimpleFK":
##            #Use a basic CG1 space for now.
##            Vv = VectorFunctionSpace(S3.mesh,"CG",1)
##            self.demag =  SimpleFKSolver(Vv, m, Ms)
