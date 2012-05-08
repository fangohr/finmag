import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from energy_base import EnergyBase
from finmag.demag.problems.prob_base import FemBemDeMagProblem
from finmag.demag.solver_fk import FemBemFKSolver

log = logging.getLogger("finmag")

class FKDemag(EnergyBase):
    """
    For now: Just a proxy object for what exists already
    for our new Simulation class.

    """
    def __init__(self):
        log.info("Creating FKDemag object.")

    def setup(self, S3, m, Ms, unit_length):
        timings.start("create-demag-problem")
        problem = FemBemDeMagProblem(S3.mesh(), m)
        problem.Ms = Ms
        self.demag = FemBemFKSolver(problem, unit_length=unit_length)
        timings.stop("create-demag-problem")

    def compute_field(self):
        return self.demag.compute_field()

    def compute_energy(self):
        pass
