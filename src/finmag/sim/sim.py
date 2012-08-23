import time
import logging
import dolfin as df
from finmag.sim.llg import LLG
from finmag.util.timings import timings
from finmag.util.helpers import quiver
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
        if not hasattr(self, "integrator"):
            self.integrator = LLGIntegrator(self.llg, self.llg.m)
        self.integrator.run_until(t)

    def relax(self, stopping_dmdt=ONE_DEGREE_PER_NS):
        """
        Do time integration of the magnetisation M until it reaches a state
        where the change of M magnetisation at each node is smaller than the
        threshold `stopping_dm_dt` (which should be given in rad/s).

        For details see the docstring of sim.integrator.BaseIntegrator.run_until_relaxation().
        """
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

    def set_stt(self, current_density, polarisation, thickness, direction):
        """
        Activate the computation of the Slonczewski spin-torque term in the LLG.

        Current density in A/m^2 is a dolfin expression,
        Polarisation is between 0 and 1,
        Thickness of the free layer in m,
        Direction (unit length) of the polarisation as a triple.

        """  
        self.llg.use_slonczewski(current_density, polarisation, thickness, direction)

    def toggle_stt(self, new_state=None):
        """
        Toggle the computation of the Slonczewski spin-torque term.

        You can optionally pass in a new state.
        """
        if new_state:
            self.llg.do_slonczewski = new_state
        else:
            self.llg.do_slonczewski = not self.llg.do_slonczewski

    def snapshot(self, filename=None):
        """
        Save a snapshot of the current magnetisation configuration to a file (using Mayavi).

        If `filename` is None, a default filename will be generated based on a
        sequentially increasing counter and the current timestep of the simulation.
        """
        if not hasattr(self, "snapshot_no"):
            self.snapshot_no = 1
        if filename is None:
            filename = "snapshot_{}_{:.3f}ns.pdf".format(self.snapshot_no, self.llg.t*1e9)
        nb_icons = 1000
        nb_nodes = len(self.llg.m)/3
        one_in_x = int(float(nb_nodes)/nb_icons) if nb_nodes > nb_icons else 1
        quiver(self.llg.m, self.mesh,
               filename=filename,
               mode="cone",
               mask_points=one_in_x)
        self.snapshot_no += 1
        log.info("Saved snapshot of magnetisation at t={} to {}.".format(self.llg.t, filename))
