import time
import inspect
import logging
import dolfin as df
import numpy as np
from finmag.sim.llg import LLG
from finmag.util.timings import mtimed
from finmag.util.consts import exchange_length, bloch_parameter
from finmag.util.meshes import mesh_info, mesh_volume
from finmag.util.fileio import Tablewriter
from finmag.util import helpers
from finmag.util.vtk_saver import VTKSaver
from finmag.sim.hysteresis import hysteresis as hyst, hysteresis_loop as hyst_loop
from finmag.sim import sim_helpers
from finmag.energies import Exchange, Zeeman, Demag, UniaxialAnisotropy, DMI
from finmag.integrators.llg_integrator import llg_integrator
from finmag.integrators import scheduler, events

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

log = logging.getLogger(name="finmag")


class Simulation(object):
    """
    Unified interface to finmag's micromagnetic simulations capabilities.

    Attributes:
        t           the current simulation time

    """
    @mtimed
    def __init__(self, mesh, Ms, unit_length=1, name='unnamed', integrator_backend="sundials"):
        """Simulation object.

        *Arguments*

          mesh : a dolfin mesh

          Ms   : Magnetisation saturation (in A/m) of the material.

          unit_length: the distance (in metres) associated with the
                       distance 1.0 in the mesh object.

          name : the Simulation name (used for writing data files, for examples)
        """
        # Store the simulation name and a 'sanitized' version of it which
        # contains only alphanumeric characters and underscores. The latter
        # will be used as a prefix for .log/.ndt files etc.
        self.name = name
        self.sanitized_name = helpers.clean_filename(name)

        self.logfilename = self.sanitized_name + '.log'
        self.ndtfilename = self.sanitized_name + '.ndt'

        helpers.start_logging_to_file(self.logfilename, mode='w', level=logging.DEBUG)

        # Create a Tablewriter object for ourselves which will be used
        # by various methods to save the average magnetisation at given
        # timesteps.
        self.tablewriter = Tablewriter(self.ndtfilename, self, override=True)

        log.info("Creating Sim object '{}' (rank={}/{}).".format(
            self.name, df.MPI.process_number(), df.MPI.num_processes()))
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

        self.scheduler = scheduler.Scheduler()
        
        self.domains =  df.CellFunction("uint", self.mesh)
        self.domains.set_all(0)
        self.region_id=0

        self.overwrite_pvd_files = False
        self.vtk_export_filename = self.sanitized_name + '.pvd'
        self.vtk_saver = VTKSaver(self.vtk_export_filename)

        self.scheduler_shortcuts = {
            'save_restart_data' : Simulation.save_restart_data,
            'save_ndt' : sim_helpers.save_ndt,
            'save_vtk' : Simulation.save_vtk,
            'switch_off_H_ext' : Simulation.switch_off_H_ext,
            }

        # At the moment, we can only have cvode as the driver, and thus do
        # time development of a system. We may have energy minimisation at some 
        # point (the driver would be an optimiser), or something else.
        self.driver = 'cvode'


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
        
    def save_m_in_region(self,region,name='unnamed'):
        
        self.region_id+=1
        helpers.mark_subdomain_by_function(region, self.mesh, self.region_id,self.domains)
        self.dx = df.Measure("dx")[self.domains]
        
        if name=='unnamed':
            name='region_'+str(self.region_id)
        
        region_id=self.region_id
        self.tablewriter.entities[name]={
                        'unit': '<>',
                        'get': lambda sim: sim.llg.m_average_fun(dx=self.dx(region_id)),
                        'header': (name+'_m_x', name+'_m_y', name+'_m_z')}
        
        self.tablewriter.update_entity_order()

    @property
    def t(self):
        """
        Returns the current simulation time.

        """
        if hasattr(self, "integrator"):
            return self.integrator.cur_t  # the real thing
        return 0.0

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
        interaction.setup(self.S3, self.llg._m, self.llg._Ms_dg, self.unit_length)
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

        *Arguments*

        interaction_type: string

            The allowed types are those finmag knows about by classname, for
            example: 'Demag', 'Exchange', 'UniaxialAnisotropy', 'Zeeman'.

        *Returns*

        The matching interaction object. If no or more than one matching
        interaction is found, a ValueError is raised.

        """
        return self.llg.effective_field.get_interaction(interaction_type)

    def remove_interaction(self, interaction_type):
        """
        Remove the interaction of the given type.

        *Arguments*

        interaction_type: string

            The allowed types are those finmag knows about by classname, for
            example: 'Demag', 'Exchange', 'UniaxialAnisotropy', 'Zeeman'.
        """
        log.debug("Removing interaction '{}' from simulation '{}'".format(
                interaction_type, self.name))
        return self.llg.effective_field.remove_interaction(interaction_type)

    def set_H_ext(self, H_ext):
        """
        Convenience function to set the external field.
        """
        H = self.get_interaction("Zeeman")
        H.set_value(H_ext)

    def switch_off_H_ext(self, remove_interaction=True):
        """
        Convenience function to switch off the external field.

        If `remove_interaction` is True (the default), the Zeeman
        interaction will be completely removed from the Simulation
        class, which should make the time integration run faster.
        Otherwise its value is just set to zero.
        """
        if remove_interaction:
            dbg_str = "(removing Zeeman interaction)"
            self.remove_interaction("Zeeman")
        else:
            dbg_str = "(setting value to zero)"
            H = self.get_interaction("Zeeman")
            H.set_value([0, 0, 0])

        log.debug("Switching off external field {}".format(dbg_str))

    def get_field_as_dolfin_function(self, field_type):
        """
        Returns the field of the interaction of the given type or of the
        magnetisation as a dolfin function.

        *Arguments*

        field_type: string

            The allowed types are those finmag knows about by classname, for
            example: 'Demag', 'Exchange', 'UniaxialAnisotropy', 'Zeeman',
            as well as 'm' which stands for the normalised magnetisation.

        *Returns*

        A dolfin.Function representing the given field. If no or more than one
        matching field is found, a ValueError is raised.

        """
        if field_type == 'm':
            return self.llg._m
        return self.llg.effective_field.get_dolfin_function(field_type)

    def probe_field(self, field_type, pts):
        """
        Probe the field of type `field_type` at point(s) `pts`, where
        the point coordinates must be specified in metres (not in
        multiples of unit_length!).

        See the documentation of the method get_field_as_dolfin_function
        to know which ``field_type`` is allowed, and helpers.probe for the
        shape of ``pts``.

        """
        pts = np.array(pts) / self.unit_length
        return helpers.probe(self.get_field_as_dolfin_function(field_type), pts)

    def advance_time(self, t):
        """
        The lower-level counterpart to run_until, this runs without a schedule.

        """
        if not hasattr(self, "integrator"):
            self.integrator = llg_integrator(self.llg, self.llg.m, backend=self.integrator_backend)

        log.debug("Advancing time to t = {} s.".format(t))
        self.integrator.advance_time(t)

    def run_until(self, t):
        """
        Run the simulation until the given time `t` is reached.

        """
        if not hasattr(self, "integrator"):
            self.integrator = llg_integrator(self.llg, self.llg.m, backend=self.integrator_backend)

        log.info("Simulation will run until t = {:.2g} s.".format(t))
        exit_at = events.StopIntegrationEvent(t)
        self.scheduler._add(exit_at)

        self.integrator.run_with_schedule(self.scheduler)
        log.info("Simulation has reached time t = {:.2g} s.".format(self.t))

        self.scheduler._remove(exit_at)

    def relax(self, stopping_dmdt=ONE_DEGREE_PER_NS, dt_limit=1e-10,
              dmdt_increased_counter_limit=500):
        """
        Run the simulation until the magnetisation has relaxed.

        This means the magnetisation reaches a state where its change over time
        at each node is smaller than the threshold `stopping_dm_dt` (which
        should be given in rad/s).

        """
        if not hasattr(self, "integrator"):
            self.integrator = llg_integrator(self.llg, self.llg.m, backend=self.integrator_backend)
        log.info("Simulation will run until relaxation of the magnetisation.")

        if hasattr(self, "relaxation"):
            del(self.relaxation)

        self.relaxation = events.RelaxationEvent(self, stopping_dmdt, dmdt_increased_counter_limit, dt_limit)
        self.scheduler._add(self.relaxation)

        self.integrator.run_with_schedule(self.scheduler)
        self.integrator.reinit()
        log.info("Relaxation finished at time t = {:.2g}.".format(self.t))

        self.scheduler._remove(self.relaxation) 
        del(self.relaxation.sim) # help the garbage collection by avoiding circular reference

    save_restart_data = sim_helpers.save_restart_data

    def restart(self, filename=None, t0=None):
        """If called, we look for a filename of type sim.name + '-restart.npz',
        and load it. The magnetisation in the restart file will be assigned to
        sim.llg._m. If this is from a cvode time integration run, it will also
        initialise (create) the integrator with that m, and the time at which the 
        restart data was saved.

        The time can be overriden with the optional parameter t0 here.

        The method will fail if no restart file exists.
        """

        if filename == None:
            filename = sim_helpers.canonical_restart_filename(self)
        log.debug("Loading restart data from {}. ".format(filename))

        data = sim_helpers.load_restart_data(filename)
    
        if not data['driver'] in ['cvode']:
            log.error("Requested unknown driver `{}` for restarting. Known: {}.".format(data["driver"], "cvode"))
            raise NotImplementedError("Unknown driver `{}` for restarting.".format(data["driver"]))
        
        self.llg._m.vector()[:] = data['m']
       
        self.reset_time(t0 or data["simtime"])

        log.info("Reloaded and set m (<m>=%s) and time=%s from %s." % \
            (self.llg.m_average, self.t, filename))

    def reset_time(self, t0):
        """
        Reset the internal clock time of the simulation to `t0`.

        This also adjusts the internal time of the scheduler and time integrator.
        """
        self.integrator = llg_integrator(self.llg, self.llg.m,
                                         backend=self.integrator_backend, t0=t0)
        self.scheduler.reset(t0)
        assert self.t == t0  # self.t is read from integrator

    def save_averages(self):
        """
        Save average field values (such as the magnetisation) to a file.

        The filename is derived from the simulation name (as given when the
        simulation was initialised) and has the extension .ndt'.
        """
        log.debug("Saving average field values for simulation '{}'.".format(self.name))
        self.tablewriter.save()

    hysteresis = hyst
    hysteresis_loop = hyst_loop

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

    def clear_schedule(self):
        self.scheduler.clear()
        self.scheduler.reset(self.t)

    def schedule(self, func, *args, **kwargs):
        """
        Register a function that should be called during the simulation.

        By default the schedule operates on simulation time expressed in
        seconds. Use either the `at` keyword argument to define a single point
        in time at which your function is called, or use the `every` keyword to
        specify an interval between subsequent calls to your function. When
        specifying the interval, you can optionally use the `after` keyword to
        delay the first execution of your function. Additionally, you can set
        the `at_end` option to `True` to have your function called at the end
        of the simulation. This can be combined with `at` and `every`.

        Note that if the internal simulation time is not zero (i.e.. if the
        simulation has already run for some time) then using the 'every'
        keyword will implicitly set 'after' to the current simulation time,
        so that the event repeats in regular intervals from the current time
        onwards). If this is undesired, you should explicitly provide 'after'
        (which is interpreted as an 'absolute' time, i.e. not as an offset to
        the current simulation time).

        You can also schedule actions using real time instead of simulation
        time by setting the `realtime` option to True. In this case you can
        use the `after` keyword on its own.

        The function func(sim) you provide should expect the simulation object
        as its first argument. All arguments to the 'schedule' function (except
        the special ones 'at', 'every', 'at_end' and 'realtime' mentioned
        above) will be passed on to this function.

        If func is a string, it will be looked up in self.scheduler_shortcuts,
        which includes 'save_restart_data', 'save_ndt' and 'save_vtk'.
        """
        if isinstance(func, str):
            if func in self.scheduler_shortcuts:
                func = self.scheduler_shortcuts[func]
            else:
                msg = "Scheduling keyword '%s' unknown. Known values are %s" \
                    % (func, self.scheduler_shortcuts.keys())
                log.error(msg)
                raise KeyError(msg)

        func_args = inspect.getargspec(func).args
        illegal_argnames = ['at', 'after', 'every', 'at_end', 'realtime']
        for kw in illegal_argnames:
            if kw in func_args:
                raise ValueError(
                    "The scheduled function must not use any of the following "
                    "argument names: {}".format(illegal_argnames))

        at = kwargs.pop('at', None)
        every = kwargs.pop('every', None)
        after = kwargs.pop('after', self.t if (every != None) else None)
        at_end = kwargs.pop('at_end', False)
        realtime = kwargs.pop('realtime', False)

        self.scheduler.add(func, [self] + list(args), kwargs,
                at=at, at_end=at_end, every=every, after=after, realtime=realtime)

    def snapshot(self, filename="", directory="", force_overwrite=False):
        """
        Deprecated. Use 'save_vtk' instead.

        """
        log.warning("Method 'snapshot' is deprecated. Use 'save_vtk' instead.")
        self.vtk(self, filename, directory, force_overwrite)

    def set_vtk_export_filename(self, filename=""):
        """
        Set the filename which is used for saving VTK snapshots.
        """
        self.vtk_export_filename = filename

    def save_vtk(self, filename=None):
        """
        Save the magnetisation to a VTK file.
        """
        if filename != None:
            # Explicitly provided filename overwrites the previously used one.
            self.vtk_export_filename = filename

        # Check whether we're still writing to the same file.
        if self.vtk_saver.filename != self.vtk_export_filename:
            self.vtk_saver.open(self.vtk_export_filename, self.overwrite_pvd_files)

        self.vtk_saver.save_field(self.llg._m, self.t)

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
        sim.add(DMI(D))
    if demag_solver != None:
        sim.add(Demag(solver=demag_solver))

    return sim
