from __future__ import division
import os
import re
import time
import glob
import logging
import textwrap
import fileinput
import itertools
import dolfin as df
import numpy as np
from numpy import NaN
from finmag.sim.llg import LLG
from finmag.util.timings import timings
from finmag.util.helpers import norm, vector_valued_function
from finmag.util.consts import exchange_length, bloch_parameter
from finmag.util.meshes import mesh_info, mesh_volume
from finmag.util.fileio import Tablewriter
from finmag.util import helpers
from finmag.sim.integrator import LLGIntegrator
from finmag.energies.exchange import Exchange
from finmag.energies.anisotropy import UniaxialAnisotropy
from finmag.energies.zeeman import Zeeman
from finmag.energies import Demag


ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

log = logging.getLogger(name="finmag")


class Simulation(object):
    """
    Unified interface to finmag's micromagnetic simulations capabilities.

    Attributes:
        t           the current simulation time

    """

    field_classes = {
        "exchange": Exchange,
        "demag": Demag,
        "anisotropy": UniaxialAnisotropy,
        "zeeman": Zeeman
        }

    def __init__(self, mesh, Ms, unit_length=1, name='unnamed', integrator_backend="sundials"):
        """Simulation object.

        *Arguments*

          mesh : a dolfin mesh

          Ms   : Magnetisation saturation (in A/m) of the material.

          unit_length: the distance (in metres) associated with the
                       distance 1.0 in the mesh object.

          name : the Simulation name (used for writing data files, for examples)
        """

        timings.reset()
        timings.start("Sim-init")

        # Store the simulation name and a 'sanitized' version of it which
        # contains only alphanumeric characters and underscores. The latter
        # will be used as a prefix for .log/.ndt files etc.
        self.name = name
        self.sanitized_name = helpers.clean_filename(name)

        self.logfilename = self.sanitized_name + '.log'
        self.ndtfilename = self.sanitized_name + '.ndt'

        helpers.start_logging_to_file(self.logfilename, mode='w')

        # Create a Tablewriter object for ourselves which will be used
        # by various methods to save the average magnetisation at given
        # timesteps.
        self.tablewriter = Tablewriter(self.ndtfilename, self, override=True)

        log.info("Creating Sim object '{}' (rank={}/{}) [{}].".format(
            self.name, df.MPI.process_number(),
            df.MPI.num_processes(), time.asctime()))
        log.info(mesh)

        self.mesh = mesh
        self.Ms = Ms
        self.unit_length = unit_length
        self.integrator_backend = integrator_backend
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
        self.llg = LLG(self.S1, self.S3)
        self.llg.Ms = Ms
        self.Volume = mesh_volume(mesh)
        self.t = 0

        timings.stop("Sim-init")

    def __str__(self):
        """String briefly describing simulation object"""
        return "finmag.Simulation(name='%s') with %s" % (self.name, self.mesh)

    def __get_m(self):
        """The unit magnetisation"""
        return self.llg.m

    def set_m(self, value, **kwargs):
        self.llg.set_m(value, **kwargs)

    m = property(__get_m, set_m)

    @property
    def m_average(self):
        """
        Compute and return the average magnetisation over the entire
        mesh, according to the formula :math:`\\langle m \\rangle =
        \\frac{1}{V} \int m \: \mathrm{d}V`
        """
        return self.llg.m_average

    def add(self, interaction, with_time_update=None):
        """
        Add an interaction (such as Exchange, Anisotropy, Demag).

        *Arguments:*

        interaction

             The interaction to be added.

         with_time_update (optional)

             A function of the form f(t), which accepts a time step
             `t` as its only single parameter and updates the internal
             state of the interaction accordingly.
        """
        log.debug("Adding interaction %s to simulation '%s'" % (str(interaction),self.name))
        interaction.setup(self.S3, self.llg._m, self.Ms, self.unit_length)
        self.llg.effective_field.add(interaction, with_time_update)

    def total_energy(self):
        """
        Compute and return the total energy of all fields present in
        the simulation.

        """
        return self.llg.effective_field.total_energy()

    def get_interaction(self, interaction_type):
        """
        Returns the interaction of the given type.

        Allowed types: "exchange", "demag, "anisotropy", "zeeman".

        Returns None if no interaction of the given type is present in
        the simulation. Raises an exception if more than one
        interaction is found.
        """
        try:
            FieldClass = self.field_classes[interaction_type]
        except KeyError:
            raise ValueError(
                "'interaction_type' must be a string representing one of the "
                "known field types: {}".format(self.field_classes.keys()))

        field_lst = [e for e in self.llg.effective_field.interactions
                     if isinstance(e, FieldClass)]

        if len(field_lst) > 1:
            raise ValueError(
                "Expected at most one interaction of type '{}' in simulation. "
                "Found: {}".format(interaction_type, len(field_lst)))

        try:
            res = field_lst[0]
        except IndexError:
            res = None

        return res

    def get_field_as_dolfin_function(self, field_type):
        """
        Return the given field as a dolfin.Function.

        *Arguments*

        field_type: string

            The field to be converted to a dolfin.Function. Must be
            one of: "exchange", "demag", "anisotropy", "zeeman", "m"
            (where the last one is the normalised magnetisation).

        *Returns*

        A dolfin.Function representing the given field.
        """
        if field_type == 'm':
            field_fun = self.llg._m
        else:
            # Mmh, upon re-reading this implementation it feels that
            # this is somehow redunant. Don't we already keep the
            # interactions as dolfin.Functions somewhere?
            #    -- Max, 13.12.2012
            field = self.get_interaction(field_type)
            if field is None:
                raise ValueError("No interaction of type '{}' present "
                                 "in simulation.".format(field_type))

            S3 = df.VectorFunctionSpace(self.mesh, 'CG', 1)
            # XXX TODO: The following line probably copies the data in a
            # while creating the function (should check whether this is
            # really the case!). If so, can we avoid this somehow? -- Max
            a = field.compute_field()
            field_fun = vector_valued_function(a, S3)
        return field_fun

    def probe_field(self, field_type, pts):
        """
        Probe the field of type `field_type` at point(s) `pts`.

        *Arguments*

        field_type: string or classname

            The field to probe. Must be one of: "exchange", "demag",
            "anisotropy", "zeeman".

        pts: numpy.array

            An array of points where the field should be probed. Can
            have arbitrary shape, except that the last axis must have
            dimension 3. For example, if pts.shape == (10,20,5,3) then
            the field is probed at all points on a regular grid of
            size 10 x 20 x 5.

        *Returns*

            A numpy.array of the same shape as `pts`, where the last
            axis contains the field values instead of the point
            locations.

        *Limitations*

        Currently the points where the field is probed must lie inside
        the mesh. For points which lie outside the mesh the returned
        field values will be NaN.
        """
        pts = np.array(pts)
        if not pts.shape[-1] == 3:
            raise ValueError(
                "Arguments 'pts' must be a numpy array of 3D points, "
                "i.e. the last axis must have dimension 3. Shape of "
                "'pts' is: {}".format(pts.shape))

        fun_field = self.get_field_as_dolfin_function(field_type)
        def _fun_field_impl(pt):
            try:
                return fun_field(pt)
            except RuntimeError:
                return np.array([NaN, NaN, NaN])
        res = np.empty_like(pts)

        # Apply the given function to each point and fill 'res' with
        # the results:
        loop_indices = itertools.product(*map(xrange, pts.shape[:-1]))
        for idx in loop_indices:
            res[idx] = _fun_field_impl(pts[idx])

        return res

    def run_until(self, t, save_averages=True):
        """
        Run the simulation until the given time step is reached.

        *Arguments*

        t: float

            The time up to which the simulation is to be run.

        save_averages: bool

            If True (the default) then the method `save_averages` is
            called automatically when the given time step is reached
            (this adds a line to the .ndt file in which the average
            fields for this simulation object are recorded).
        """
        if not hasattr(self, "integrator"):
            self.integrator = LLGIntegrator(self.llg, self.llg.m,
                                            backend=self.integrator_backend,
                                            tablewriter=self.tablewriter)
        log.debug("Integrating dynamics up to t = %g" % t)
        self.integrator.run_until(t)
        self.t = t
        if save_averages:
            self.save_averages()

    def save_averages(self):
        """
        Save average field values (such as the magnetisation) to a file.

        The filename is derived from the simulation name (as given when the
        simulation was initialised) and has the extension .ndt'.
        """
        log.debug("Saving average field values for simulation "
                  "'{}'".format(self.name))
        self.tablewriter.save()

    def relax(self, save_snapshots=False, filename='', save_every=100e-12,
              save_final_snapshot=True, force_overwrite=False,
              stopping_dmdt=ONE_DEGREE_PER_NS, dt_limit=1e-10,
              dmdt_increased_counter_limit=50):
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
            self.integrator = LLGIntegrator(self.llg, self.llg.m,
                                            backend=self.integrator_backend,
                                            tablewriter=self.tablewriter)

        if save_snapshots == True:
            if filename == '':
                raise ValueError("If save_snapshots is True, filename must "
                                 "be a non-empty string.")
            else:
                ext = os.path.splitext(filename)[1]
                if ext != '.pvd':
                    raise ValueError(
                        "File extension for vtk snapshot file must be '.pvd', "
                        "but got: '{}'".format(ext))
            if os.path.exists(filename):
                if force_overwrite:
                    log.warning(
                        "Removing file '{}' and all associated .vtu files "
                        "(because force_overwrite=True).".format(filename))
                    os.remove(filename)
                    basename = re.sub('\.pvd$', '', filename)
                    for f in glob.glob(basename + "*.vtu"):
                        os.remove(f)
                else:
                    raise IOError(
                        "Aborting snapshot creation. File already exists and "
                        "would overwritten: '{}' (use force_overwrite=True if "
                        "this is what you want)".format(filename))
        else:
            if filename != '':
                log.warning("Value of save_snapshot is False, but filename is "
                            "given anyway: '{}'. Ignoring...".format(filename))

        self.integrator.run_until_relaxation(
            save_snapshots=save_snapshots, filename=filename,
            save_every=save_every, save_final_snapshot=save_final_snapshot,
            stopping_dmdt=stopping_dmdt,
            dmdt_increased_counter_limit=dmdt_increased_counter_limit,
            dt_limit=dt_limit)

    def hysteresis(self, H_ext_list, leave_last_field_on=False, fun=None,
                   save_snapshots=False, **kwargs):
        """
        Set the applied field to the first value in `H_ext_list` (which should
        be a list of external field vectors) and then call the relax() method.
        When convergence is reached, the field is changed to the next one in
        H_ext_list, and so on until all values in H_ext_list are exhausted.

        Note: The fields in H_ext_list are applied *in addition to* any Zeeman
              interactions that are already present in the simulation.
              In particular, if only one external field should be present then
              do not add any Zeeman interactions before calling this method.

        *Arguments*

            H_ext_list -- list of external fields, where each field can have
                          any of the forms accepted by Zeeman.__init__()
                          (see its docstring for more details)

           leave_last_field_on -- if False (the default) then the external
                                  field is switched off again after all
                                  relaxations are finished.

           fun -- the user can pass a function here (which should accept the
                  Simulation object as its only argument); this function is
                  called after each relaxation and can be used for example to
                  save the value of the averaged magnetisation.

        If `fun` is not None then this function returns a list containing
        an accumulation of all the return values of `fun` after each stage.
        Otherwise this function returns None

        For a list of keyword arguments accepted by this method see the
        documentation of the relax() method, to which all given keyword
        arguments are passed on. Note that if a `filename` argument is
        provided, the string 'stage_xxx' is appended to it, where xxx is
        a running counter which indicates for which field in H_ext_list
        the relax() method is being executed.
        """
        if H_ext_list == []:
            return

        filename = kwargs.get('filename', None)
        force_overwrite = kwargs.get('force_overwrite', False)
        if filename != None and force_overwrite == True:
            if os.path.exists(filename):
                # Delete the global .pvd file as well as all existing .pvd
                # and .vtu file from any previously run hysteresis stages.
                # Although the relax() command also checks for existing files,
                # it would miss any stages that were previously run but are
                # not reached during this run.
                log.debug("Removing the file '{}' as well as all associated "
                          ".pvd and .vtu files of previously run hysteresis "
                          "stages.".format(filename))
                pvdfiles = glob.glob(re.sub('\.pvd$', '', filename) + '*.pvd')
                vtufiles = glob.glob(re.sub('\.pvd$', '', filename) + '*.vtu')
                for f in pvdfiles + vtufiles:
                    os.remove(f)

        # Add a new Zeeman interaction, initialised to zero.
        H = Zeeman((0, 0, 0))
        self.add(H)

        # We keep track of the current stage of the hysteresis loop.
        # Each stage is saved to a different .pvd file, whose name
        # includes the current stage number.
        cur_stage = 0
        num_stages = len(H_ext_list)
        filename = re.sub('\.pvd$', '', kwargs.pop('filename', ''))
        cur_filename = ''

        res = []

        try:
            while True:
                H_cur = H_ext_list[cur_stage]
                log.info("Entering hysteresis stage #{} "
                         "({} out of {}).".format(cur_stage,
                                                  cur_stage + 1, num_stages))
                H.set_value(H_cur)

                if filename != '':
                    cur_filename = filename + "__stage_{:03d}__.pvd".format(cur_stage)
                self.relax(filename=cur_filename, save_snapshots=save_snapshots, **kwargs)
                cur_stage += 1
                if fun is not None:
                    retval = fun(self)
                    res.append(retval)
                    log.debug("hysteresis callback function '{}' returned "
                              "value: {}".format(fun.__name__, retval))
        except IndexError:
            log.info("Hysteresis is finished.")

        if not leave_last_field_on:
            log.info("Switching off the applied field used for hysteresis.")
            self.llg.effective_field.interactions.remove(H)

        if save_snapshots:
            # We now remove trailing underscores from output filenames
            # (for cosmetic resons only ... ;-) and create a 'global'
            # output file which combines all stages of the simulation.
            #
            f_global = open(filename + '.pvd', 'w')
            f_global.write(textwrap.dedent("""\
                <?xml version="1.0"?>
                <VTKFile type="Collection" version="0.1">
                  <Collection>
                """))

            cur_stage = 0
            cur_timestep = 0
            for f in sorted(glob.glob(filename + "__stage_[0-9][0-9][0-9]__.pvd")):
                f_global.write("    <!-- Hysteresis stage #{:03d} -->\n".format(cur_stage))
                for line in fileinput.input([f]):
                    if re.match('^\s*<DataSet .*/>$', line):
                        # We require sequentially increasing timesteps, so
                        # we have to manually substitue them (TODO: unless
                        # we can already do this when saving the snapshot?!?)
                        line = re.sub('timestep="[0-9]+"', 'timestep="{}"'.format(cur_timestep), line)
                        f_global.write(line)
                        cur_timestep += 1
                f_global.write("\n")
                os.rename(f, re.sub('__\.pvd', '.pvd', f))
                cur_stage += 1
            f_global.write("  </Collection>\n</VTKFile>\n")
            f_global.close()

        return res or None

    def hysteresis_loop(self, H_max, direction, N, **kwargs):
        """
        Compute a hysteresis loop. This is a specialised convenience
        version of the more general `hysteresis` method. It computes a
        hysteresis loop where the external field is applied along a
        single axis and changes magnitude from +H_max to -H_max and
        back (using N steps in each direction).

        The return value is a pair (H_vals, m_vals), where H_vals is
        the list of field strengths at which a relaxation is performed
        and m_vals is a list of scalar values containing, for each
        field value, the averaged value of the magnetisation along the
        axis `direction` (after relaxation has been reached). Thus the
        command plot(H_vals, m_vals) could be used to plot the
        hysteresis loop.

           direction -- a vector indicating the direction of the
                        external field (will be normalised
                        automatically)

           H_max -- maximum field strength

           N -- number of data points to compute in each direction
                (thus the total number of data points for the entire
                loop will be 2*N-1)

           kwargs -- any keyword argument accepted by the hysteresis() method
        """
        d = np.array(direction)
        H_dir = d / norm(d)
        H_norms = list(np.linspace(H_max, -H_max, N)) + \
            list(np.linspace(-H_max, H_max, N))
        H_vals = [h * H_dir for h in H_norms]
        m_avg = self.hysteresis(H_vals, fun=lambda sim: sim.m_average, **kwargs)
        # projected lengths of the averaged magnetisation values along the axis `H_dir`
        m_vals = [np.dot(m, H_dir) for m in m_avg]
        return (H_norms, m_vals)

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
        self.llg.spatially_varying_alpha(alpha, multiplicator)

    def __get_gamma(self):
        return self.llg.gamma

    def __set_gamma(self, value):
        self.llg.gamma = value

    gamma = property(__get_gamma, __set_gamma)

    def reinit_integrator(self):
        """
        If an integrator is already present in the simulation, call
        its reinit() method. Otherwise do nothing.
        """
        if hasattr(self, "integrator"):
            self.integrator.reinit()
        else:
            log.warning("Integrator reinit was requested, but no integrator "
                        "is present in the simulation!")

    def timings(self, n=20):
        """
        Prints an overview of wall time and number of calls for designated
        subparts of the code, listing up to n items, starting with those
        which took the longest.

        """
        log.warning("Note that this method is currently broken because it "
                    "reports global timings, not just the times taken by "
                    "this specific simulation only.")

        return timings.report_str(n)

    def set_stt(self, current_density, polarisation, thickness, direction):
        """
        Activate the computation of the Slonczewski spin-torque term
        in the LLG.

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

    def snapshot(self, filename="", directory="", force_overwrite=False,
                 infix="", save_averages=True):
        """
        Save a snapshot of the current magnetisation configuration to
        a .pvd file (in VTK format) which can later be inspected using
        Paraview, for example.

        If `filename` is empty, a default filename will be generated
        based on a sequentially increasing counter and the current
        timestep of the simulation. A user-defined string can be
        inserted into the generated filename by passing `infix`.

        If `directory` is non-empty then the file will be saved in the
        specified directory.

        Note that `filename` is also allowed to contain directory
        components (for example filename='snapshots/foo.pvd'), which
        are simply appended to `directory`. However, if `filename`
        contains an absolute path then the value of `directory` is
        ignored. If a file with the same filename already exists, the
        method will abort unless `force_overwrite` is True, in which
        case the existing .pvd and all associated .vtu files are
        deleted before saving the snapshot.

        All directory components present in either `directory` or
        `filename` are created if they do not already exist.

        If save_averages is True (the default) then the averaged fields
        will also be saved to an .ndt file.
        """
        if not hasattr(self, "vtk_snapshot_no"):
            self.vtk_snapshot_no = 1
        if filename == "":
            infix_insert = "" if infix == "" else "_" + infix
            filename = "m{}_{}_{:.3f}ns.pvd".format(infix_insert,
                self.vtk_snapshot_no, self.t * 1e9)

        ext = os.path.splitext(filename)[1]
        if ext != '.pvd':
            raise ValueError("File extension for vtk snapshot file must be "
                             "'.pvd', but got: '{}'".format(ext))
        if os.path.isabs(filename) and directory != "":
            log.warning("Ignoring 'directory' argument (value given: '{}') "
                        "because 'filename' contains an absolute path: "
                        "'{}'".format(directory, filename))

        output_file = os.path.join(directory, filename)
        if os.path.exists(output_file):
            if force_overwrite:
                log.warning(
                    "Removing file '{}' and all associated .vtu files "
                    "(because force_overwrite=True).".format(output_file))
                os.remove(output_file)
                basename = re.sub('\.pvd$', '', output_file)
                for f in glob.glob(basename + "*.vtu"):
                    os.remove(f)
            else:
                raise IOError(
                    "Aborting snapshot creation. File already exists and "
                    "would overwritten: '{}' (use force_overwrite=True if "
                    "this is what you want)".format(output_file))
        t0 = time.time()
        f = df.File(output_file, "compressed")
        f << self.llg._m
        t1 = time.time()
        log.info(
            "Saved snapshot of magnetisation at t={} to file '{}' (saving "
            "took {:.3g} seconds).".format(self.t, output_file, t1 - t0))
        self.vtk_snapshot_no += 1

        if save_averages:
            self.save_averages()

    save_vtk = snapshot  # alias with a more telling name

    def mesh_info(self):
        """
        Return a string containing some basic information about the
        mesh (such as the number of cells, interior/surface triangles,
        vertices, etc.).

        Also print a distribution of edge lengths present in the mesh
        and how they compare to the exchange length and the Bloch
        parameter (if these can be computed). This information is
        relevant to estimate whether the mesh discretisation
        is too coarse and might result in numerical artefacts (see W. Rave,
        K. Fabian, A. Hubert, "Magnetic states ofsmall cubic particles with
        uniaxial anisotropy", J. Magn. Magn. Mater. 190 (1998), 332-348).

        """
        info_string = "{}\n".format(mesh_info(self.mesh))

        edgelengths = [e.length() * self.unit_length for e in df.edges(self.mesh)]

        def added_info(L, name, abbrev):
            (a, b), _ = np.histogram(edgelengths, bins=[0, L, np.infty])
            if b == 0.0:
                msg = "All edges are shorter"
                msg2 = ""
            else:
                msg = "Warning: {:.2f}% of edges are longer".format(100.0 * b / (a + b))
                msg2 = " (this may lead to discretisation artefacts)"
            info = "{} than the {} {} = {:.2f} nm{}.\n".format(msg, name, abbrev, L * 1e9, msg2)
            return info

        if hasattr(self.llg.effective_field, "exchange"):
            A = self.llg.effective_field.exchange.A
            l_ex = exchange_length(A, self.llg.Ms)
            info_string += added_info(l_ex, 'exchange length', 'l_ex')
            if hasattr(self.llg.effective_field, "anisotropy"):
                K1 = float(self.llg.effective_field.anisotropy.K1)
                l_bloch = bloch_parameter(A, K1)
                info_string += added_info(l_bloch, 'Bloch parameter', 'l_bloch')

        return info_string


def sim_with(mesh, Ms, m_init, alpha=0.5, unit_length=1, integrator_backend="sundials",
             A=None, K1=None, K1_axis=None, H_ext=None, demag_solver='FK', 
             D=None, name="unnamed"):
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
    sim = Simulation(mesh, Ms, unit_length=unit_length, 
                     integrator_backend=integrator_backend,
                     name=name)

    sim.set_m(m_init)
    sim.alpha = alpha

    # If any of the optional arguments are provided, initialise
    # the corresponding interactions here:
    if A is not None:
        sim.add(Exchange(A))
    if (K1 != None and K1_axis is None) or (K1 is None and K1_axis != None):
        log.warning(
            "Not initialising uniaxial anisotropy because only one of K1, "
            "K1_axis was specified (values given: K1={}, K1_axis={}).".format(
                K1, K1_axis))
    if K1 != None and K1_axis != None:
        sim.add(UniaxialAnisotropy(K1, K1_axis))
    if H_ext != None:
        sim.add(Zeeman(H_ext))
    if D != None:
        from finmag.energies import DMI
        sim.add(DMI(D))
    if demag_solver != None:
        sim.add(Demag(solver=demag_solver))

    return sim
