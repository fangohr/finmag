import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from finmag.energies.energy_base import EnergyBase
from solver_fk import FemBemFKSolver
from solver_gcr import FemBemGCRSolver
from solver_fk_test import SimpleFKSolver

log = logging.getLogger("finmag")

class Demag(EnergyBase):
    """
    A wrapper for the demag solvers that also implements the functionality of
    an energy class.

    *Arguments*
        solver
            demag solver method: "FK", "GCR" or "weiwei"

    """
    def __init__(self, solver="FK", degree=1, element="CG", project_method="magpar",bench = False):
        self.in_jacobian = False
        log.debug("Creating Demag object with " + solver + " solver.")

        if solver in ["FK", "GCR", "weiwei"]:
            self.solver = solver
        else:
            raise NotImplementedError("Only 'FK', 'GCR' and 'weiwei' are implemented")

        self.parameters = df.Parameters("demag_options")
        poisson = df.Parameters("poisson_solver")
        poisson.add("method", "default")
        poisson.add("preconditioner", "default")
        laplace = df.Parameters("laplace_solver")
        laplace.add("method", "default")
        laplace.add("preconditioner", "default")
        self.parameters.add(poisson)
        self.parameters.add(laplace)

        self.degree = degree
        self.element = element
        self.method = project_method
        self.bench = bench

    def setup(self, S3, m, Ms, unit_length):
        """
        S3
            dolfin VectorFunctionSpace
        m
            the Dolfin object representing the (unit) magnetisation
        Ms
            the saturation magnetisation

        unit_length
            The scale of the mesh, default is 1.

        """

##        mesh,m, parameters=None, degree=1, element="CG", project_method='magpar',
##                 unit_length=1,Ms = 1.0
                 
        kwargs = {"mesh":S3.mesh(),
                  "m":m,
                  "Ms":Ms,
                  "unit_length":unit_length,
                  "parameters":None,
                  "degree":1,
                  "element":"CG",
                  "project_method":'magpar',
                  "bench": self.bench}

##
##       # timings.startnext("create-demag-problem")
##        problem = FemBemDeMagProblem(S3.mesh(), m, Ms)

        if self.solver == "FK":
            self.demag = FemBemFKSolver(**kwargs)
        elif self.solver == "FK_magpar":
            self.demag = MagparFKSolver(**kwargs)
        elif self.solver == "GCR":
            self.demag = FemBemGCRSolver(**kwargs)
        elif self.solver == "weiwei":
            self.demag = SimpleFKSolver(S3, m, Ms)

        #timings.startnext("Solve-demag-problem")
        """
        if self.solver != "weiwei":
            self.demag.solve()
        """

    def compute_field(self):
        return self.demag.compute_field()

    def compute_energy(self):
        return self.demag.compute_energy()

    def compute_potential(self):
        self.demag.solve()
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
