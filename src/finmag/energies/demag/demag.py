import logging
import textwrap
import dolfin as df
from finmag.util.timings import default_timer, mtimed
from solver_fk import FemBemFKSolver
from solver_gcr import FemBemGCRSolver
from treecode_bem import TreecodeBEM
from fk_demag import FKDemag
from solver_base import default_parameters


log = logging.getLogger("finmag")


class Demag(object):
    """
    A wrapper for the demag solvers that also implements the functionality of
    an energy class.

    *Arguments*
        solver
            demag solver method: "FK", "GCR", "Treecode", "new_fk"

    """
    def __init__(self, solver="FK", degree=1, element="CG", project_method="magpar",bench = False,
                 parameters = default_parameters):
        self.in_jacobian = False
        log.debug("Creating Demag object with " + solver + " solver.")

        if solver in ["FK", "GCR","Treecode", "old_fk"]:
            self.solver = solver
        else:
            raise NotImplementedError("Only 'FK', 'GCR', 'Treecode' and 'old_FK' are implemented")

        self.degree = degree
        self.element = element
        self.method = project_method
        self.bench = bench
        self.parameters = parameters

    def setup(self, S3, m, Ms, unit_length = 1,
              p=4,mac=0.35,number_limit=1,correct_factor=10,type_I=True):
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

        self.S3 = S3
        kwargs = {"mesh":S3.mesh(),
                  "m":m,
                  "Ms":Ms,
                  "unit_length":unit_length,
                  "parameters":self.parameters,
                  "degree":1,
                  "element":"CG",
                  "project_method":'magpar',
                  "bench": self.bench}
        
        if self.solver == "old_FK":
            self.demag = FemBemFKSolver(**kwargs)
            for (name, solver) in (("Poisson", self.demag.poisson_solver), ("Laplace", self.demag.laplace_solver)):
                params = repr(solver.parameters.to_dict())
                #Log the linear solver parameters
                log.debug("{}: {} solver parameters.\n{}".format(
                            self.__class__.__name__, name, textwrap.fill(params, width=100,
                            initial_indent=4*" ", subsequent_indent=4*" ")))
        #MagparFKSolver does not exist? (HF 17 June 2012)
        #elif self.solver == "FK_magpar":
        #    self.demag = MagparFKSolver(**kwargs)
        elif self.solver == "GCR":
            self.demag = FemBemGCRSolver(**kwargs)
            for (name, solver) in (("Poisson", self.demag.poisson_solver), ("Laplace", self.demag.laplace_solver)):
                params = repr(solver.parameters.to_dict())
                log.debug("{}: {} solver parameters.\n{}".format(
                            self.__class__.__name__, name, textwrap.fill(params, width=100,
                            initial_indent=4*" ", subsequent_indent=4*" ")))
        elif self.solver == "Treecode":
            kwargs["p"]=p
            kwargs["mac"]=mac
            kwargs["num_limit"]=number_limit
            kwargs["correct_factor"]=correct_factor
            kwargs["type_I"]=type_I
            
            self.demag = TreecodeBEM(**kwargs)
        elif self.solver == "FK":
            demag = FKDemag()
            demag.setup(S3, m, Ms, unit_length)
            self.demag = demag

    @mtimed
    def compute_field(self):
        return self.demag.compute_field()

    @mtimed
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

    print default_timer

    df.plot(demag.compute_potential())
    df.interactive()
