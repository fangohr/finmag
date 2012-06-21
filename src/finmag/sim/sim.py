import time
import logging
import dolfin as df
from finmag.sim.llg import LLG
from finmag.util.timings import timings
from finmag.sim.integrator import LLGIntegrator

ONE_DEGREE_PER_NS = 17453292.5 # in rad/s

log = logging.getLogger(name="finmag")


class Simulation(object):
    def __init__(self, mesh, Ms, unit_length=1):
        timings.reset()
        timings.start("Sim-init")

        log.info("Creating Sim object (rank={}/{}) [{}].".format(
            df.MPI.process_number(), df.MPI.num_processes(), time.asctime()))
        log.info(mesh)

        self.mesh = mesh
        self.Ms = Ms
        self.unit_length = unit_length
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
        self.llg = LLG(self.S1, self.S3)
        self.llg.Ms = Ms
        self.Volume = df.assemble(df.Constant(1) * df.dx, mesh=mesh)
        self.t = 0

        timings.stop("Sim-init")

    def __get_m(self):
            return self.llg.m

    def set_m(self, value, **kwargs):
        self.llg.set_m(value, **kwargs)

    m = property(__get_m, set_m)

    @property
    def m_average(self):
        return self.llg.m_average

    def add(self, interaction, with_time_update=None):
        interaction.setup(self.S3, self.llg._m, self.Ms, self.unit_length)
        self.llg.interactions.append(interaction)

        if with_time_update:
            self.llg._pre_rhs_callables.append(with_time_update)

    def effective_field(self):
        self.llg.compute_effective_field()
        return self.llg.H_eff

    def dmdt(self):
        return self.llg.solve()

    def total_energy(self):
        energy = 0.
        for interaction in self.llg.interactions:
            energy += interaction.compute_energy()
        return energy

    def run_until(self, t):
        log.info("Will integrate until {}.".format(t))
        if not hasattr(self, "integrator"):
            self.integrator = LLGIntegrator(self.llg, self.llg.m)
        self.integrator.run_until(t)

    def relax(self, stopping_dmdt=ONE_DEGREE_PER_NS):
        log.info("Will integrate until relaxation.")
        if not hasattr(self, "integrator"):
            self.integrator = LLGIntegrator(self.llg, self.llg.m)
        self.integrator.run_until_relaxation(stopping_dmdt=stopping_dmdt)

    def __get_pins(self):
        return self.llg.pins

    def __set_pins(self, nodes):
        self.llg.pins = nodes

    pins = property(__get_pins, __set_pins)

    def __get_alpha(self):
        return self.llg.alpha

    def __set_alpha(self, value):
        self.llg.alpha = value

    alpha = property(__get_alpha, __set_alpha)

    def spatial_alpha(self, alpha, multiplicator):
        self.llg.spatially_varying_alpha(self, alpha, multiplicator)

    def __get_gamma(self):
        return self.llg.gamma

    def __set_gamma(self, value):
        self.llg.gamma = value

    gamma = property(__get_gamma, __set_gamma)

    def timings(self, n=20):
        """
        Prints an overview of wall time and number of calls for designated
        subparts of the code, listing up to n items, starting with those
        which took the longest.

        """
        return timings.report_str(n)
