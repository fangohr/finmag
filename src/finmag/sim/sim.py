from __future__ import division
import os
import re
import time
import glob
import logging
import dolfin as df
import numpy as np
from math import sqrt
from finmag.sim.llg import LLG
from finmag.util.timings import timings
from finmag.util.helpers import quiver
from finmag.util.consts import mu0
from finmag.util.meshes import mesh_info
from finmag.sim.integrator import LLGIntegrator
from finmag.energies.exchange import Exchange
from finmag.energies.anisotropy import UniaxialAnisotropy
from finmag.energies.zeeman import Zeeman
from finmag.energies import Demag

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
        """
        Add an interaction (such as Exchange, Anisotropy, Demag)
        """
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

    def exchange_length_and_bloch_parameter(self):
        """
        Compute the exchange length and the Bloch parameter.

           L1 = sqrt((2*A)/(mu0*Ms**2)

           L2 = sqrt(A/K_1)

        Returns the pair (L1, L2). If either of the values cannot be
        computed, NaN is returned in its place.

        These are relevant to estimate whether the mesh discretisation
        is too coarse and might result in numerical artefacts (see W. Rave,
        K. Fabian, A. Hubert, "Magnetic states ofsmall cubic particles with
        uniaxial anisotropy", J. Magn. Magn. Mater. 190 (1998), 332-348).

        The meaning of the constants is as follows:

           A    -  exchange coupling constant
           K_1  -  (first) anisotropy constant
           Ms   -  saturation magnetisation
           mu0  -  vacuum permeability
        """
        exs = [e for e in self.llg.interactions if isinstance(e, Exchange)]
        if len(exs) == 1:
            A = exs[0].A
            L1 = sqrt(2*A/(mu0*self.llg.Ms**2))
            log.info("Exchange length: {:.2f} nm (exchange coupling constant: A={:.2g} J/m)".format(L1*1e9, A))
        else:
            L1 = np.NaN
            log.warning("Cannot calculate exchange length or Bloch parameter ssince exactly one exchange interaction must be present in the simulation (found: {})".format(len(exs)))

        # Note that only uniaxial anisotropy interactions are being considered to compute the Bloch parameter
        # since this is the only type currently supported in Finmag and the common formula \sqrt(A/K1) only
        # works for a uniaxial anisotropy. It might be worth investigating whether there is a more general
        # expression once other types become available in Finmag, too.
        ans = [a for a in self.llg.interactions if isinstance(a, UniaxialAnisotropy)]
        if len(ans) == 1 and len(exs) == 1:
            K1 = float(ans[0].K1) # K1 can be a dolfin Constant, hence the conversion to float
            L2 = sqrt(A/K1)
            log.info("Bloch parameter: {:.2f} nm (uniaxial anisotropy constant: K1={:.2g} J/m)".format(L2*1e9, K1))
        else:
            L2 = np.NaN
            log.warning("Cannot calculate Bloch parameter since exactly one uniaxial anisotropy interaction (and one exchange interaction) must be present in the simulation (found: {})".format(len(ans)))

        return (L1, L2)

    def run_until(self, t):
        if not hasattr(self, "integrator"):
            self.integrator = LLGIntegrator(self.llg, self.llg.m)
        self.integrator.run_until(t)
        self.t = t

    def relax(self, save_snapshots=False, filename='', save_every=100e-12,
              save_final_snapshot=True, force_overwrite=False,
              stopping_dmdt=ONE_DEGREE_PER_NS, dt_limit=1e-10, dmdt_increased_counter_limit=20):
        """
        Do time integration of the magnetisation M until it reaches a
        state where the change of M magnetisation at each node is
        smaller than the threshold `stopping_dm_dt` (which should be
        given in rad/s).

        If save_snapshots is True (default: False) then a series of
        snapshots is saved to `filename` (which must be specified in
        this case). If `filename` contains directory components then
        these are created if they do not already exist. A snapshot is
        saved every `save_every` seconds (default: 100e-12, i.e. every
        100 picoseconds). Usually, one last snapshot is saved after
        the relaxation finishes (or aborts prematurely). This can be
        disabled by setting save_final_snapshot to False. If a file
        with the same name as `filename` already exists, the method
        will abort unless `force_overwrite` is True, in which case the
        existing .pvd and all associated .vtu files are deleted before
        saving the series of snapshots.

        For details and the meaning of the other keyword arguments see
        the docstring of sim.integrator.BaseIntegrator.run_until_relaxation().

        """
        log.info("Will integrate until relaxation.")
        if not hasattr(self, "integrator"):
            self.integrator = LLGIntegrator(self.llg, self.llg.m)

        if save_snapshots == True:
            if filename == '':
                raise ValueError("If save_snapshots is True, filename must be a non-empty string.")
            else:
                ext = os.path.splitext(filename)[1]
                if ext != '.pvd':
                    raise ValueError("File extension for vtk snapshot file must be '.pvd', but got: '{}'".format(ext))
            if os.path.exists(filename):
                if force_overwrite:
                    log.warning("Removing file '{}' and all associated .vtu files (because force_overwrite=True).".format(filename))
                    os.remove(filename)
                    basename = re.sub('\.pvd$', '', filename)
                    for f in glob.glob(basename+"*.vtu"):
                        os.remove(f)
                else:
                    raise IOError("Aborting snapshot creation. File already exists and would overwritten: '{}' (use force_overwrite=True if this is what you want)".format(filename))
        else:
            if filename != '':
                log.warning("Value of save_snapshot is False, but filename is given anyway: '{}'. Ignoring...".format(filename))

        self.integrator.run_until_relaxation(save_snapshots=save_snapshots,
                                             filename=filename,
                                             save_every=save_every,
                                             save_final_snapshot=save_final_snapshot,
                                             stopping_dmdt=stopping_dmdt,
                                             dmdt_increased_counter_limit=dmdt_increased_counter_limit,
                                             dt_limit=dt_limit)

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

    def snapshot(self, filename="", directory="", force_overwrite=False):
        """
        Save a snapshot of the current magnetisation configuration to a .pvd file
        (in VTK format) which can later be inspected using Paraview, for example.

        If `filename` is empty, a default filename will be generated based on a
        sequentially increasing counter and the current timestep of the simulation.

        If `directory` is non-empty then the file will be saved in the specified directory.

        Note that `filename` is also allowed to contain directory components
        (for example filename='snapshots/foo.pvd'), which are simply appended
        to `directory`. However, if `filename` contains an absolute path then
        the value of `directory` is ignored. If a file with the same filename
        already exists, the method will abort unless `force_overwrite` is True,
        in which case the existing .pvd and all associated .vtu files are
        deleted before saving the snapshot.

        All directory components present in either `directory` or `filename`
        are created if they do not already exist.

        """
        if not hasattr(self, "vtk_snapshot_no"):
            self.vtk_snapshot_no = 1
        if filename == "":
            filename = "snapshot_{}_{:.3f}ns.pvd".format(self.vtk_snapshot_no, self.t*1e9)

        ext = os.path.splitext(filename)[1]
        if ext != '.pvd':
            raise ValueError("File extension for vtk snapshot file must be '.pvd', but got: '{}'".format(ext))
        if os.path.isabs(filename) and directory != "":
            log.warning("Ignoring 'directory' argument (value given: '{}') because 'filename' contains an absolute path: '{}'".format(directory, filename))

        output_file = os.path.join(directory, filename)
        if os.path.exists(output_file):
            if force_overwrite:
                log.warning("Removing file '{}' and all associated .vtu files (because force_overwrite=True).".format(output_file))
                os.remove(output_file)
                basename = re.sub('\.pvd$', '', output_file)
                for f in glob.glob(basename+"*.vtu"):
                    os.remove(f)
            else:
                raise IOError("Aborting snapshot creation. File already exists and would overwritten: '{}' (use force_overwrite=True if this is what you want)".format(output_file))
        t0 = time.time()
        f = df.File(output_file, "compressed")
        f << self.llg._m
        t1 = time.time()
        log.info("Saved snapshot of magnetisation at t={} to file '{}' (saving took {:.3g} seconds).".format(self.t, output_file, t1-t0))

    def mesh_info(self):
        """
        Return a string containing some basic information about the
        mesh (such as the number of cells, interior/surface triangles,
        vertices, etc.).

        Also print a distribution of edge lengths present in the mesh
        and how they compare to the exchange length and the Bloch
        parameter (if these can be computed). This information is
        relevant to estimate whether the mesh discretisation is too
        coarse and might result in numerical artefacts (also see the
        docstring of the function `exchange_length_and_bloch_parameter`).
        """
        info_string = "{}\n".format(mesh_info(self.mesh))

        edgelengths = [e.length()*self.unit_length for e in df.edges(self.mesh)]
        emax = max(edgelengths)
        L1, L2 = self.exchange_length_and_bloch_parameter()

        def added_info(L, name, abbrev):
            if np.isnan(L):
                info = ""
            else:
                (a,b), _ = np.histogram(edgelengths, bins=[0, L, np.infty])
                if b == 0.0:
                    msg = "All edges are shorter"
                    msg2 = ""
                else:
                    msg = "Warning: {:.2f}% of edges are longer".format(100.0*b/(a+b))
                    msg2 = " (this may lead to discretisation artefacts)"
                info = "{} than the {} {} = {:.2f} nm{}.\n".format(msg, name, abbrev, L*1e9, msg2)
            return info

        info_string += added_info(L1, 'exchange length', 'L1')
        info_string += added_info(L2, 'Bloch parameter', 'L2')
        return info_string

    def print_mesh_info(self):
        print self.mesh_info()

def sim_with(mesh, Ms, m_init, unit_length=1, A=None, K1=None, K1_axis=None, H_ext=None, demag_solver='FK'):
    """
    Create a Simulation instance based on the given parameters.

    This is a convenience function which allows quick creation of a
    common simulation type where at most one exchange/anisotropy/demag
    interaction is present and the initial magnetisation is known.

    If a value for any of the optional arguments A, K1 (and K1_axis),
    or demag_solver are provided then the corresponding exchange /
    anisotropy / demag interaction is created automatically and added
    to the simulation. For example, providing the value A=13.0e-12 in
    the function call is equivalent to:

       exchange = Exchange(A)
       sim.add(exchange)
    """
    sim = Simulation(mesh, Ms, unit_length)

    sim.set_m(m_init)

    # If any of the optional arguments are provided, initialise
    # the corresponding interactions here:
    if A is not None:
        sim.add(Exchange(A))
    if (K1 != None and K1_axis is None) or (K1 is None and K1_axis != None):
        log.warning("Not initialising uniaxial anisotropy because only one of K1, K1_axis was specified (values given: K1={}, K1_axis={}).".format(K1, K1_axis))
    if K1 != None and K1_axis != None:
        sim.add(UniaxialAnisotropy(K1, K1_axis))
    if H_ext != None:
        sim.add(Zeeman(H_ext))
    if demag_solver != None:
        sim.add(Demag(solver=demag_solver))

    return sim
