import logging
import textwrap
import dolfin as df
from finmag.util.timings import timings
from solver_fk import FemBemFKSolver
from solver_gcr import FemBemGCRSolver
from solver_base import default_parameters


log = logging.getLogger("finmag")


class Demag(object):
    """
    A wrapper for the demag solvers that also implements the functionality of
    an energy class.

    *Arguments*
        solver
            demag solver method: "FK", "GCR" or "weiwei"

    """
    def __init__(self, solver="FK", degree=1, element="CG", project_method="magpar",bench = False,
                 parameters = default_parameters):
        self.in_jacobian = False
        log.debug("Creating Demag object with " + solver + " solver.")

        if solver in ["FK", "GCR", "weiwei"]:
            self.solver = solver
        else:
            raise NotImplementedError("Only 'FK', 'GCR' and 'weiwei' are implemented")

        self.degree = degree
        self.element = element
        self.method = project_method
        self.bench = bench
        self.parameters = parameters

    def setup(self, S3, m, Ms, unit_length = 1):
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
                         
        kwargs = {"mesh":S3.mesh(),
                  "m":m,
                  "Ms":Ms,
                  "unit_length":unit_length,
                  "parameters":self.parameters,
                  "degree":1,
                  "element":"CG",
                  "project_method":'magpar',
                  "bench": self.bench}
        
        if self.solver == "FK":
            self.demag = FemBemFKSolver(**kwargs)
        #MagparFKSolver does not exist? (HF 17 June 2012)
        #elif self.solver == "FK_magpar":
        #    self.demag = MagparFKSolver(**kwargs)
        elif self.solver == "GCR":
            self.demag = FemBemGCRSolver(**kwargs)
        
        #Log the linear solver parameters
        
        for (name, solver) in (("Poisson", self.demag.poisson_solver), ("Laplace", self.demag.laplace_solver)):
            params = repr(solver.parameters.to_dict())
            log.debug("{}: {} solver parameters.\n{}".format(
                        self.__class__.__name__, name, textwrap.fill(params, width=100,
                        initial_indent=4*" ", subsequent_indent=4*" ")))

    def compute_field(self):
        return self.demag.compute_field()

    def compute_energy(self):
        return self.demag.compute_energy()

    def compute_potential(self):
        self.demag.solve()
        return self.demag.phi

if __name__ == "__main__":
    from finmag.tests.demag.problems import prob_fembem_testcases as pft
    prob = pft.MagSphereBase(10, 0.8)

    demag = Demag("GCR")
    demag.setup(prob.V, prob.m, prob.Ms, unit_length = 1)

    print timings

    df.plot(demag.compute_potential())
    df.interactive()
