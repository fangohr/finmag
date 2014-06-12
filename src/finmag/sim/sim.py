from __future__ import division
import os
import time
import inspect
import logging
import itertools
import dolfin as df
import numpy as np
from aeon import mtimed
from finmag.physics.llg import LLG
from finmag.physics.llg_stt import LLG_STT
from finmag.physics.llb.sllg import SLLG
from finmag.sim import sim_details
from finmag.sim import sim_relax
from finmag.sim import sim_savers
from finmag.util.meshes import mesh_volume, mesh_size_plausible, \
    describe_mesh_size, plot_mesh, plot_mesh_with_paraview
from finmag.util.fileio import Tablewriter, FieldSaver
from finmag.util import helpers
from finmag.util.vtk_saver import VTKSaver
from finmag.util.helpers import plot_dynamics
from finmag.sim.hysteresis import hysteresis as hyst, hysteresis_loop as hyst_loop
from finmag.sim import sim_helpers, magnetisation_patterns
from finmag.drivers.llg_integrator import llg_integrator
from finmag.drivers.sundials_integrator import SundialsIntegrator
from finmag.scheduler import scheduler
from finmag.util.pbc2d import PeriodicBoundary1D, PeriodicBoundary2D
from finmag.energies import Exchange, Zeeman, TimeZeeman, Demag, UniaxialAnisotropy, DMI

log = logging.getLogger(name="finmag")


class Simulation(object):
    """
    Unified interface to finmag's micromagnetic simulations capabilities.

    Attributes:
        t           the current simulation time

    """
    @mtimed
    def __init__(self, mesh, Ms, unit_length=1, name='unnamed', kernel='llg' ,integrator_backend="sundials", pbc=None, average=False):
        """Simulation object.

        *Arguments*

          mesh : a dolfin mesh

          Ms   : Magnetisation saturation (in A/m) of the material.

          unit_length: the distance (in metres) associated with the
                       distance 1.0 in the mesh object.

          name : the Simulation name (used for writing data files, for examples)

          pbc : Periodic boundary type: None or '2d'

          kernel : 'llg', 'sllg' or 'llg_stt'

          average : take the cell averaged effective field, only for test, will delete it if doesn't work.

        """
        # Store the simulation name and a 'sanitized' version of it which
        # contains only alphanumeric characters and underscores. The latter
        # will be used as a prefix for .log/.ndt files etc.
        self.name = name
        self.sanitized_name = helpers.clean_filename(name)

        self.logfilename = self.sanitized_name + '.log'
        self.ndtfilename = self.sanitized_name + '.ndt'

        self.logging_handler = helpers.start_logging_to_file(self.logfilename, mode='w', level=logging.DEBUG)

        # Create a Tablewriter object for ourselves which will be used
        # by various methods to save the average magnetisation at given
        # timesteps.
        self.tablewriter = Tablewriter(self.ndtfilename, self, override=True)

        self.tablewriter.entities['E_total'] = {
            'unit': '<J>',
            'get': lambda sim: sim.total_energy(),
            'header': 'E_total'}
        self.tablewriter.entities['H_total'] = {
            'unit': '<A/m>',
            'get': lambda sim: helpers.average_field(sim.effective_field()),
            'header': ('H_total_x', 'H_total_y', 'H_total_z')}
        self.tablewriter.update_entity_order()

        log.info("Creating Sim object '{}' (rank={}/{}).".format(
            self.name, df.MPI.rank(df.mpi_comm_world()), df.MPI.size(df.mpi_comm_world())))
        log.info(mesh)

        self.pbc = pbc
        if pbc == '2d':
            self.pbc = PeriodicBoundary2D(mesh)
        elif pbc == '1d':
            self.pbc = PeriodicBoundary1D(mesh)

        if not mesh_size_plausible(mesh, unit_length):
            log.warning("The mesh is {}.".format(describe_mesh_size(mesh, unit_length)))
            log.warning("unit_length is set to {}. Are you sure this is correct?".format(unit_length))

        self.mesh = mesh
        self.Ms = Ms
        self.unit_length = unit_length
        self.integrator_backend = integrator_backend
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=self.pbc)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3, constrained_domain=self.pbc)

        if kernel=='llg':
            self.llg = LLG(self.S1, self.S3, average = average, unit_length=unit_length)
        elif kernel=='sllg':
            self.llg = SLLG(self.S1, self.S3, unit_length=unit_length)
        elif kernel=='llg_stt':
            self.llg = LLG_STT(self.S1, self.S3, unit_length=unit_length)
        else:
            raise ValueError("kernel must be one of llg, sllg or llg_stt.")

        self.kernel = kernel

        self.llg.Ms = Ms
        self.Volume = mesh_volume(mesh)

        self.scheduler = scheduler.Scheduler()
        self.callbacks_at_scheduler_events = []

        self.domains = df.CellFunction("uint", self.mesh)
        self.domains.set_all(0)
        self.region_id = 0

        # XXX TODO: this separation between vtk_savers and
        # field_savers is artificial and should/will be removed once
        # we have a robust, unified field saving mechanism.
        self.vtk_savers = {}
        self.field_savers = {}
        self._render_scene_indices = {}

        self.scheduler_shortcuts = {
            'eta': sim_helpers.eta,
            'ETA': sim_helpers.eta,
            'plot_relaxation': sim_helpers.plot_relaxation,
            'render_scene': Simulation._render_scene_incremental,
            'save_averages': sim_helpers.save_ndt,
            'save_field': sim_savers._save_field_incremental,
            'save_m': sim_savers._save_m_incremental,
            'save_ndt': sim_helpers.save_ndt,
            'save_restart_data': sim_helpers.save_restart_data,
            'save_vtk': self.save_vtk,
            'switch_off_H_ext': Simulation.switch_off_H_ext,
        }

        # At the moment, we can only have cvode as the driver, and thus do
        # time development of a system. We may have energy minimisation at some
        # point (the driver would be an optimiser), or something else.
        self.driver = 'cvode'

        # let's use 1e-6 as default and we can change it later
        self.reltol = 1e-6
        self.abstol = 1e-6

        # We used to only create the integrator when needed. However, this can
        # lead to a bug when the user saves information to an .ndt file before
        # a time integrator exists because some columns will be missing (as the
        # time integrator creates additional columns in the .ndt file). Therefore
        # we play it safe and create the time integrator here at the beginning.
        #self.create_integrator()

    def __str__(self):
        """String briefly describing simulation object"""
        return "finmag.Simulation(name='%s') with %s" % (self.name, self.mesh)

    def __get_m(self):
        """The unit magnetisation"""
        return self.llg.m

    def set_m(self, value, normalise=True, **kwargs):
        """
        Set the magnetisation (if `normalise` is True, it is automatically
        normalised to unit length).

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        You can call this method anytime during the simulation. However, when
        providing a numpy array during time integration, the use of
        the attribute m instead of this method is advised for performance
        reasons and because the attribute m doesn't normalise the vector.

        """
        self.llg.set_m(value, normalise=normalise, **kwargs)

    m = property(__get_m, set_m)

    @property
    def m_average(self):
        """
        Compute and return the average magnetisation over the entire
        mesh, according to the formula :math:`\\langle m \\rangle =
        \\frac{1}{V} \int m \: \mathrm{d}V`
        """
        return self.llg.m_average

    @property
    def _m(self):
        return self.llg._m

    def save_m_in_region(self,region,name='unnamed'):

        self.region_id += 1
        helpers.mark_subdomain_by_function(region, self.mesh, self.region_id, self.domains)
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

    @property
    def dmdt_max(self):
        """
        Gets dmdt values for each mesh node. Finds the max of
        the L2 Norms. Returns (x,y,z) components of dmdt, where
        this max occurs.
        """
        #FIXME:error here
        dmdts = self.llg.dmdt.reshape((3,-1))
        norms = np.sqrt(np.sum(dmdts**2,axis=0))
        index = norms.argmax()
        dmdt_x = dmdts[0][index]
        dmdt_y = dmdts[1][index]
        dmdt_z = dmdts[2][index]
        return np.array([dmdt_x, dmdt_y, dmdt_z])

    @property
    def dmdt(self):
        """
        Returns dmdt for all mesh nodes.

        *** What is the best format (e.g. numpy of dolfin) for this? ***

        """
        return self.llg.dmdt


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
        self.llg.effective_field.add(interaction, with_time_update)

        if isinstance(interaction, TimeZeeman):
            # The following line ensures that the field value is updated
            # correctly whenever the time integration reaches a scheduler
            # "checkpoint" (i.e. whenever integrator.advance_time(t) finishes
            # successfully).
            self.callbacks_at_scheduler_events.append(interaction.update)

        energy_name = 'E_{}'.format(interaction.name)
        field_name = 'H_{}'.format(interaction.name)
        self.tablewriter.entities[energy_name] = {
            'unit': '<J>',
            'get': lambda sim: sim.get_interaction(interaction.name).compute_energy(),
            'header': energy_name}
        self.tablewriter.entities[field_name] = {
            'unit': '<A/m>',
            'get': lambda sim: sim.get_interaction(interaction.name).average_field(),
            'header': (field_name + '_x', field_name + '_y', field_name + '_z')}
        self.tablewriter.update_entity_order()

    def effective_field(self):
        """
        Compute and return the effective field.

        """
        return self.llg.effective_field.compute(self.t)

    def total_energy(self):
        """
        Compute and return the total energy of all fields present in
        the simulation.

        """
        return self.llg.effective_field.total_energy()

    def compute_energy(self, name):
        """
        Compute and return the energy contribution from a specific
        interaction (Exchange, Demag, Zeeman, etc.). If the simulation
        does not contain an interaction with this name, a value of
        zero is returned.

        *Arguments*

        name:  string

            The name of the interaction for which the energy should be
            computed, or 'total' for the total energy present in the
            simulation.

        """
        if name.lower() == 'total':
            res = self.total_energy()
        else:
            interaction = self.get_interaction(name)
            res = interaction.compute_energy()
        return res

    def has_interaction(self, interaction_name):
        """
        Returns True if an interaction with the given name exists, and False otherwise.

        *Arguments*

        interaction_name: string

            Name of the interaction.

        """
        return self.llg.effective_field.exists(interaction_name)

    def interactions(self):
        """ Return the names of the known interactions. """
        return self.llg.effective_field.all()

    def get_interaction(self, interaction_name):
        """
        Returns the interaction with the given name

        *Arguments*

        interaction_name: string

            Name of the interaction.

        *Returns*

        The matching interaction object. If no or more than one matching
        interaction is found, a ValueError is raised.

        """
        return self.llg.effective_field.get(interaction_name)


    def get_interaction_list(self):
        """
        Returns a list of interaction names.

        *Returns*

        A list of strings, each string corresponding to the name of one interaction.
        """
        return self.llg.effective_field.all()


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
        return self.llg.effective_field.remove(interaction_type)

    def set_H_ext(self, H_ext):
        """
        Convenience function to set the external field.
        """
        try:
            H = self.get_interaction("Zeeman")
            H.set_value(H_ext)
        except KeyError:
            H = Zeeman(H_ext)
            self.add(H)

    def switch_off_H_ext(self, remove_interaction=False):
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

    def get_field_as_dolfin_function(self, field_type, region=None):
        """
        Returns the field of the interaction of the given type or of the
        magnetisation as a dolfin function.

        *Arguments*

        field_type: string

            The allowed types are those finmag knows about by classname, for
            example: 'Demag', 'Exchange', 'UniaxialAnisotropy', 'Zeeman',
            as well as 'm' which stands for the normalised magnetisation.

        region:

            Some identifier that uniquely identifies a mesh region. This required
            that the method `mark_regions` has been called previously so that the
            simulation knows about the regions and their IDs.

        *Returns*

        A dolfin.Function representing the given field. If no or more than one
        matching field is found, a ValueError is raised.

        """
        if field_type == 'm':
            field = self.llg._m
        else:
            field = self.llg.effective_field.get_dolfin_function(field_type)
        res = field

        if region:
            # XXX TODO: The function space V_region was created using a 'restriction'.
            #           This means that the underlying mesh of the function space is
            #           still the full mesh. However, dolfin functions in V_region will
            #           only have degrees of freedom corresponding to the specific region.
            #           Strangely, however, if we construct a function such as 'field' below
            #           and ask for its vector, it seems to contain elements that don't make
            #           sense. Need to debug this, but for now we go the route below and
            #           explicitly construct a function space on the submesh for the region
            #           and interpolate into that.
            V_region = self._get_region_space(region)
            #field_restr = df.interpolate(field, V_region)
            #res = field_restr

            # Define a new function space on the submesh belonging to the region and interpolate
            # the field into it. I would have thought that this is precisely what the restricted
            # function spaces are intended for in dolfin, but the previous lines don't seem to
            # work as intended, so I'm using this intermediate fix for now.
            submesh = self.get_submesh(region)
            if (V_region.ufl_element().family() != 'Lagrange'):
                raise NotImplementedError("XXX The following lines assume that we have a CG1 function space. Fix this!!")
            if (V_region.ufl_element().value_shape() != (3,)):
                raise NotImplementedError("This functioality is currently only implemented for 3-vector fields.")
            V_submesh = df.VectorFunctionSpace(submesh, 'CG', 1, dim=3)
            f_submesh = df.interpolate(field, V_submesh)
            res = f_submesh

        return res

    def _get_region_space(self, region=None):
        if region:
            try:
                V_region = self.region_spaces[region]
            except AttributeError:
                raise RuntimeError("No regions defined in mesh. Please call 'mark_regions' first to define some.")
            except KeyError:
                raise ValueError("Region not defined: '{}'. Allowed values: {}".format(region, self.region_ids.keys()))
        else:
            V_region = self.S3
        return V_region

    def _get_region_id(self, region=None):
        if region:
            try:
                region_id = self.region_ids[region]
            except AttributeError:
                raise RuntimeError("No regions defined in mesh. Please call 'mark_regions' first to define some.")
            except KeyError:
                raise ValueError("Region not defined: '{}'. Allowed values: {}".format(region, self.region_ids.keys()))
        return region_id

    def probe_field(self, field_type, pts, region=None):
        """
        Probe the field of type `field_type` at point(s) `pts`, where
        the point coordinates must be specified in mesh coordinates.

        See the documentation of the method get_field_as_dolfin_function
        to know which ``field_type`` is allowed, and helpers.probe for the
        shape of ``pts``.

        """
        return helpers.probe(self.get_field_as_dolfin_function(field_type, region=region), pts)

    def probe_field_along_line(self, field_type, pt_start, pt_end, N=100, region=None):
        """
        Probe the field of type `field_type` at `N` equidistant points
        along a straight line connecting `pt_start` and `pt_end`.

        See the documentation of the method get_field_as_dolfin_function
        to know which ``field_type`` is allowed.

        Example:

           probe_field_along_line('m', [-200, 0, 0], [200, 0, 0], N=200)

        """
        return helpers.probe_along_line(self.get_field_as_dolfin_function(field_type, region=region), pt_start, pt_end, N)

    def create_integrator(self, backend=None, **kwargs):

        if not hasattr(self, "integrator"):
            if backend == None:
                backend = self.integrator_backend
            log.info("Create integrator {} with kwargs={}".format(backend, kwargs))
            if self.kernel == 'llg_stt':
                self.integrator = SundialsIntegrator(self.llg, self.llg.dy_m, method="bdf_diag", **kwargs)
            elif self.kernel == 'sllg':
                self.integrator = self.llg
            else:
                self.integrator = llg_integrator(self.llg, self.llg.m, backend=backend, **kwargs)
                self.integrator.integrator.set_scalar_tolerances(self.reltol, self.abstol)

            self.tablewriter.entities['steps'] = {
                'unit': '<1>',
                'get': lambda sim: sim.integrator.stats()['nsteps'],
                'header': 'steps'}

            self.tablewriter.entities['last_step_dt'] = {
                'unit': '<1>',
                'get': lambda sim: sim.integrator.stats()['hlast'],
                'header': 'last_step_dt'}
            self.tablewriter.entities['dmdt'] = {
                'unit': '<A/ms>',
                'get': lambda sim: sim.dmdt_max,
                'header': ('dmdt_x', 'dmdt_y', 'dmdt_z')}

            self.tablewriter.update_entity_order()

        else:
            log.warning("Cannot create integrator - exists already: {}".format(self.integrator))
        return self.integrator

    def set_tol(self, reltol=1e-6, abstol=1e-6):
        """
        Set the tolerences of the default integrator.
        """
        self.reltol = reltol
        self.abstol = abstol

        if hasattr(self, "integrator"):
            self.integrator.integrator.set_scalar_tolerances(reltol, abstol)


    def advance_time(self, t):
        """
        The lower-level counterpart to run_until, this runs without a schedule.

        """
        if not hasattr(self, "integrator"):
            self.create_integrator()

        log.debug("Advancing time to t = {} s.".format(t))
        self.integrator.advance_time(t)
        # The following line is necessary because the time integrator may
        # slightly overshoot the requested end time, so here we make sure
        # that the field values represent that requested time exactly.
        self.llg.effective_field.update(t)

    def run_until(self, t):
        """
        Run the simulation until the given time `t` is reached.

        """
        if not hasattr(self, "integrator"):
            self.create_integrator()

        log.info("Simulation will run until t = {:.2g} s.".format(t))
        self.t_max = t

        # Define function that stops integration and add it to scheduler. The
        # at_end parameter is required because t can be zero, which is
        # considered as False for comparison purposes in scheduler.add.
        def call_to_end_integration():
            return False
        self.scheduler.add(call_to_end_integration, at=t, at_end=True)

        self.scheduler.run(self.integrator, self.callbacks_at_scheduler_events)

        # The following line is necessary because the time integrator may
        # slightly overshoot the requested end time, so here we make sure
        # that the field values represent that requested time exactly.
        self.llg.effective_field.update(t)

        log.info("Simulation has reached time t = {:.2g} s.".format(self.t))

    relax = sim_relax.relax

    save_restart_data = sim_helpers.save_restart_data

    def restart(self, filename=None, t0=None):
        """If called, we look for a filename of type sim.name + '-restart.npz',
        and load it. The magnetisation in the restart file will be assigned to
        self._m. If this is from a cvode time integration run, it will also
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

        self.reset_time(data["simtime"] if (t0 == None) else t0)

        log.info("Reloaded and set m (<m>=%s) and time=%s from %s." % \
            (self.llg.m_average, self.t, filename))

    def reset_time(self, t0):
        """
        Reset the internal clock time of the simulation to `t0`.

        This also adjusts the internal time of the scheduler and time integrator.
        """
        #WW: Is it good to create a new integrator and with the name of reset_time? this
        #is a bit confusing and dangerous because the user doesn't know a new integrator
        #is created and the other setting that the user provided such as the tolerances
        #actually doesn't have influence at all.
        self.integrator = llg_integrator(self.llg, self.llg.m,
                                         backend=self.integrator_backend, t0=t0)
        
        self.set_tol(self.reltol, self.abstol)
        self.scheduler.reset(t0)
        assert self.t == t0  # self.t is read from integrator

    # Include magnetisation initialisation functions.
    initialise_helix_2D = magnetisation_patterns.initialise_helix_2D
    initialise_skyrmions = magnetisation_patterns.initialise_skyrmions
    initialise_skyrmion_hexlattice_2D = magnetisation_patterns.initialise_skyrmion_hexlattice_2D
    initialise_vortex = magnetisation_patterns.initialise_vortex

    save_averages = sim_helpers.save_ndt
    save_ndt = sim_helpers.save_ndt
    hysteresis = hyst
    hysteresis_loop = hyst_loop

    skyrmion_number = sim_helpers.skyrmion_number
    skyrmion_number_density_function = sim_helpers.skyrmion_number_density_function

    def __get_pins(self):
        return self.llg.pins

    def __set_pins(self, nodes):
        pinlist = []
        if hasattr(nodes, '__call__'):
            coords = self.mesh.coordinates()
            for i, c in enumerate(coords):
                if nodes(c):
                    pinlist.append(i)
            pinlist = np.array(pinlist)
            self.llg.pins = pinlist
        else:
            self.llg.pins = nodes

    pins = property(__get_pins, __set_pins)

    @property
    def do_precession(self):
        return self.llg.do_precession

    @do_precession.setter
    def do_precession(self, value):
        self.llg.do_precession = value

    @property
    def alpha(self):
        """
        The damping constant :math:`\\alpha`.

        It is stored as a scalar valued df.Function. However, it can be set
        using any type accepted by the function
        :py:func:`finmag.util.helpers.scalar_valued_function`.

        """
        return self.llg.alpha

    @alpha.setter
    def alpha(self, value):
        self.llg.set_alpha(value)

    def __get_gamma(self):
        return self.llg.gamma

    def __set_gamma(self, value):
        self.llg.gamma = value

    gamma = property(__get_gamma, __set_gamma)

    def __get_dt(self):
        return self.llg.dt

    def __set_dt(self, value):
        self.llg.dt = value

    dt = property(__get_dt, __set_dt)

    def __get_T(self):
        return self.llg.T

    def __set_T(self, value):
        self.llg.T = value

    T = property(__get_T, __set_T)

    run_normal_modes_computation = sim_helpers.run_normal_modes_computation

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

    def set_stt(self, current_density, polarisation, thickness, direction, with_time_update=None):
        """
        Activate the computation of the Slonczewski spin-torque term
        in the LLG.

        *Arguments*

        - Current density in A/m^2 as a number, dolfin function or expression.

        - Polarisation is between 0 and 1. It is defined as P = (x-y)/(x+y),
          where x and y are the fractions of spin up/down electrons).

        - Thickness of the free layer in m.

        - Direction of the polarisation as a triple (is automatically
          normalised to unit length).

        - with_time_update:

             A function of the form J(t), which accepts a time step `t`
             as its only argument and returns the new current density.

             N.B.: For efficiency reasons, the return value is currently
                   assumed to be a number, i.e. J is assumed to be spatially
                   constant (and only varying with time).

        """
        self.llg.use_slonczewski(current_density, polarisation, thickness,
                                 direction, with_time_update=with_time_update)

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
        To un-schedule this function later, store the return value `item`
        of this function and call sim.unschedule(item) later. Alternatively,
        you can call `sim.clear_schedule`, which unregisters *all* scheduled
        functions.

        By default, the schedule operates on simulation time expressed in
        seconds. Use either the `at` keyword argument to define a single
        point in time at which your function is called, or use the `every`
        keyword to specify an interval between subsequent calls to your
        function. When specifying the interval, you can optionally use the
        `after` keyword to delay the first execution of your function.
        Additionally, you can set the `at_end` option to `True` (default
        is `False`) to have your function called at the end of a simulation
        stage (e.g. when the run_until() command has reached its end time,
        or when the relax() command has reached relaxation). This can be
        combined with `at` and `every`.

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
        which includes 'save_restart_data', 'save_ndt', 'save_vtk' and
        'save_field'. For example, to save a vtk snapshot of the magnetisation
        every nanosecond, use:

            sim.schedule('save_vtk', every=1e-9, filename='m.pvd')

        and to save the magnetisation to a .npy file every 2 nanoseconds, use:

            sim.schedule('save_field', 'm', every=2e-9, filename='m.npy')

        In both of these cases, a suffix representing the number of
        the current snapshot will be added automatically, e.g.:
        'm_00000.npy', 'm_000001.npy', etc.

        """
        if isinstance(func, str):
            if func in self.scheduler_shortcuts:
                if func == 'save_vtk':
                    # This is a special case which needs some pre-processing
                    # as we need to open a .pvd file first.
                    filename = kwargs.pop('filename', None)
                    overwrite = kwargs.pop('overwrite', False)
                    try:
                        # Check whether a vtk_saver for this filename already exists; this is
                        # necessary to if 'save_vtk' is scheduled multiple times with the same
                        # filename.
                        vtk_saver = self._get_vtk_saver(filename=filename, overwrite=False)
                    except IOError:
                        # If none exists, create a new one.
                        vtk_saver = self._get_vtk_saver(filename=filename, overwrite=overwrite)

                    def aux_save(sim):
                        sim._save_m_to_vtk(vtk_saver)

                    func = aux_save
                    func = lambda sim: sim._save_m_to_vtk(vtk_saver)
                elif func == "eta" or func == "ETA":
                    eta = self.scheduler_shortcuts[func]
                    started = time.time()
                    func = lambda sim: eta(sim, when_started=started)
                else:
                    func = self.scheduler_shortcuts[func]
            else:
                msg = "Scheduling keyword '%s' unknown. Known values are %s" \
                    % (func, self.scheduler_shortcuts.keys())
                log.error(msg)
                raise KeyError(msg)

        try:
            func_args = inspect.getargspec(func).args
        except TypeError:
            # This can happen when running the binary distribution, since compiled
            # functions cannot be inspected. Not a great problem, though, because
            # this will result in an error once the scheduled function is called,
            # even though it would be preferable to catch this early.
            func_args = None

        if func_args != None:
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

        scheduled_item = self.scheduler.add(func, [self] + list(args), kwargs,
                                            at=at, at_end=at_end, every=every,
                                            after=after, realtime=realtime)
        return scheduled_item

    def unschedule(self, item):
        """
        Unschedule a previously scheduled callback function. The
        argument `item` should be the object returned by the call to
        sim.schedule(...) which scheduled that function.
        """
        self.scheduler._remove(item)

    def snapshot(self, filename="", directory="", force_overwrite=False):
        """
        Deprecated. Use 'save_vtk' instead.

        """
        log.warning("Method 'snapshot' is deprecated. Use 'save_vtk' instead.")
        self.vtk(self, filename, directory, force_overwrite)

    def _get_vtk_saver(self, filename=None, overwrite=False):
        if filename == None:
            filename = self.sanitized_name + '.pvd'

        if self.vtk_savers.has_key(filename) and (overwrite == False):
            # Retrieve an existing VTKSaver for appending data
            s = self.vtk_savers[filename]
        else:
            # Create a  new VTKSaver and store it for later re-use
            s = VTKSaver(filename, overwrite=overwrite)
            self.vtk_savers[filename] = s

        return s

    def _save_m_to_vtk(self, vtk_saver):
        vtk_saver.save_field(self.llg._m, self.t)

    def _save_field_to_vtk(self, field_name, vtk_saver, region=None):
        field_data = self.get_field_as_dolfin_function(field_name, region=region)
        field_data.rename(field_name, field_name)
        vtk_saver.save_field(field_data, self.t)

    def save_vtk(self, filename=None, overwrite=False, region=None):
        """
        Save the magnetisation to a VTK file.
        """
        self.save_field_to_vtk('m', filename=filename, overwrite=overwrite, region=region)

    def save_field_to_vtk(self, field_name, filename=None, overwrite=False, region=None):
        """
        Save the field with the given name to a VTK file.
        """
        vtk_saver = self._get_vtk_saver(filename, overwrite)
        self._save_field_to_vtk(field_name, vtk_saver, region=region)

    save_m = sim_helpers.save_m

    save_field = sim_savers.save_field

    length_scales = sim_details.length_scales
    mesh_info = sim_details.mesh_info


    def render_scene(self, outfile=None, region=None, **kwargs):
        """
        This is a convenience wrapper around the helper function
        `finmag.util.visualization.render_paraview_scene`. It saves
        the current magnetisation to a temporary file and uses
        `render_paraview_scene` to plot it. All keyword arguments
        are passed on to `render_paraview_scene`; see its docstring
        for details (one useful option is `outfile`, which can be used
        to save the resulting image to a png or jpg file).

        Returns the IPython.core.display.Image produced by
        `render_paraview_scene`.

        """
        from finmag.util.visualization import render_paraview_scene

        field_name = kwargs.get('field_name', 'm')

        with helpers.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'paraview_scene_{}.pvd'.format(self.name))
            self.save_field_to_vtk(field_name=field_name, filename=filename, region=region)
            return render_paraview_scene(filename, outfile=outfile, **kwargs)

    def _render_scene_incremental(self, filename, **kwargs):
        # XXX TODO: This should be tidied up by somehow combining it
        # with the other incremental savers. We should have a more
        # general mechanism which abstracts out the functionality of
        # incremental saving.
        try:
            cur_index = self._render_scene_indices[filename]
        except KeyError:
            self._render_scene_indices[filename] = 0
            cur_index = 0

        basename, ext = os.path.splitext(filename)
        outfilename = basename + '_{:06d}'.format(cur_index) + ext
        self.render_scene(outfile=outfilename, **kwargs)
        self._render_scene_indices[filename] += 1

    def plot_mesh(self, region=None, use_paraview=True, **kwargs):
        """
        Plot the mesh associated with the given region (or the entire mesh
        if `region` is `None`).

        This is a convenience function which internally calls either
        `finmag.util.helpers.plot_mesh_with_paraview` (if the argument
        `use_paraview` is True) or `finmag.util.hepers.plot_mesh`
        (otherwise), where the latter uses Matplotlib to plot the mesh.
        All keyword arguments are passed on to the respective helper
        function that is called internally.

        """
        mesh = self.get_submesh(region)
        if use_paraview:
            return plot_mesh_with_paraview(mesh, **kwargs)
        else:
            return plot_mesh(mesh, **kwargs)


    def close_logfile(self):
        """
        Stop logging to the logfile associated with this simulation object.

        This closes the file and removed the associated logging
        handler from the 'finmag' logger. Note that logging to other
        files (in particular the global logfile) is not affected.

        """
        self.logging_handler.stream.close()
        log.removeHandler(self.logging_handler)
        self.logging_handler = None

    def plot_dynamics(self, components='xyz', **kwargs):
        ndt_file = kwargs.pop('ndt_file', self.ndtfilename)
        if not os.path.exists(ndt_file):
            raise RuntimeError("File was not found: '{}'. Did you forget to schedule saving the averages to a .ndt file before running the simulation?".format(ndt_file))
        return plot_dynamics(ndt_file, components=components, **kwargs)

    def mark_regions(self, fun_regions):
        """
        Mark certain subdomains of the mesh.

        The argument `fun_regions` should be a callable of the form

           (x, y, z)  -->  region_id

        which takes the coordinates of a mesh point as input and returns
        the region_id for this point. Here `region_id` can be anything
        (it doesn't need to be an integer).

        """
        # Determine all region identifiers and associate each of them with a unique integer.
        # XXX TODO: This is probably quite inefficient since we loop over all mesh nodes.
        #           Can this be improved?
        all_ids = set([fun_regions(pt) for pt in self.mesh.coordinates()])
        self.region_ids = dict(itertools.izip(all_ids, xrange(len(all_ids))))

        # Create the CellFunction which marks the different mesh regions with integers
        self.region_markers = df.CellFunction('size_t', self.mesh)
        for region_id, i in self.region_ids.items():
            class Domain(df.SubDomain):
                def inside(self, pt, on_boundary):
                        return fun_regions(pt) == region_id
            subdomain = Domain()
            subdomain.mark(self.region_markers, i)

        def create_restricted_space(region_id):
            i = self.region_ids[region_id]
            restriction = df.Restriction(self.region_markers, i)
            V_restr = df.VectorFunctionSpace(restriction, 'CG', 1, dim=3)
            return V_restr

        # Create a restricted VectorFunctionSpace for each region
        try:
            self.region_spaces = {region_id: create_restricted_space(region_id) for region_id in self.region_ids}
        except AttributeError:
            raise RuntimeError("Marking mesh regions is only supported for dolfin > 1.2.0. "
                               "You may need to install a nightly snapshot (e.g. via an Ubuntu PPA). "
                               "See http://fenicsproject.org/download/snapshot_releases.html for details.")

    get_submesh = sim_helpers.get_submesh


    def set_zhangli(self, J_profile=(1e10,0,0), P=0.5, beta=0.01, using_u0=False):
        """
        Activates the computation of the zhang-li spin-torque term in the LLG.

        *Arguments*

            `J_profile` can have any of the forms accepted by the function
            'finmag.util.helpers.vector_valued_function' (see its
            docstring for details).

            if using_u0 = True, the factor of 1/(1+beta^2) will be dropped.

        """
        self.llg.use_zhangli(J_profile=J_profile, P=P, beta=beta, using_u0=using_u0)

def sim_with(mesh, Ms, m_init, alpha=0.5, unit_length=1, integrator_backend="sundials",
             A=None, K1=None, K1_axis=None, H_ext=None, demag_solver='FK',
             demag_solver_params={}, D=None, name="unnamed", pbc=None, sim_class=Simulation):
    """
    Create a Simulation instance based on the given parameters.

    This is a convenience function which allows quick creation of a
    common simulation type where at most one exchange/anisotropy/demag
    interaction is present and the initial magnetisation is known.

    By default, a generic Simulation object will be returned, but the
    argument `sim_class` can be used to create a more
    specialised type of simulation, e.g. a NormalModeSimulation.

    If a value for any of the optional arguments A, K1 (and K1_axis),
    or demag_solver are provided then the corresponding exchange /
    anisotropy / demag interaction is created automatically and added
    to the simulation. For example, providing the value A=13.0e-12 in
    the function call is equivalent to:

       exchange = Exchange(A)
       sim.add(exchange)

    The argument `demag_solver_params` can be used to configure the demag solver
    (if the chosen solver class supports this). Example with the 'FK' solver:

       demag_solver_params = {'phi_1_solver': 'cg', phi_1_preconditioner: 'ilu'}

    See the docstring of the individual solver classes (e.g. finmag.energies.demag.FKDemag)
    for more information on possible parameters.
    """
    sim = sim_class(mesh, Ms, unit_length=unit_length,
                    integrator_backend=integrator_backend,
                    name=name, pbc=pbc)

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
        demag = Demag(solver=demag_solver)
        if demag_solver_params != {}:
            for (k, v) in demag_solver_params.items():
                log.debug("Setting demag solver parameter {}='{}' for simulation '{}'".format(k, v, sim.name))
                demag.parameters[k] = v
            log.debug("Demag parameters now: {}".format(demag.parameters))
        sim.add(demag)
    log.debug("Successfully created simulation '{}'".format(sim.name))

    return sim
