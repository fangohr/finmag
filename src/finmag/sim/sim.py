from __future__ import division
import os
import re
import math
import types
import shutil
import inspect
import logging
import tempfile
import itertools
import subprocess as sp
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from aeon import mtimed
from glob import glob
from finmag.sim.llg import LLG
from finmag.sim.llg_stt import LLG_STT
from finmag.util.consts import exchange_length, bloch_parameter
from finmag.util.meshes import mesh_info, mesh_volume, plot_mesh, plot_mesh_with_paraview
from finmag.util.fileio import Tablewriter, FieldSaver
from finmag.util import helpers
from finmag.util.vtk_saver import VTKSaver
from finmag.util.fft import power_spectral_density, plot_power_spectral_density, find_peak_near_frequency, _plot_spectrum, export_normal_mode_animation_from_ringdown
from finmag.util.normal_modes import compute_eigenproblem_matrix, compute_generalised_eigenproblem_matrices, compute_normal_modes, compute_normal_modes_generalised, export_normal_mode_animation, plot_spatially_resolved_normal_mode
from finmag.util.helpers import plot_dynamics, pvd2avi
from finmag.sim.hysteresis import hysteresis as hyst, hysteresis_loop as hyst_loop
from finmag.sim import sim_helpers
from finmag.energies import Exchange, Zeeman, TimeZeeman, Demag, UniaxialAnisotropy, DMI
from finmag.integrators.llg_integrator import llg_integrator
from finmag.integrators.sundials_integrator import SundialsIntegrator
from finmag.integrators import scheduler, events
from finmag.integrators.common import run_with_schedule
from finmag.util.pbc2d import PeriodicBoundary1D, PeriodicBoundary2D

from finmag.llb.sllg import SLLG

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

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
            self.name, df.MPI.process_number(), df.MPI.num_processes()))
        log.info(mesh)

        self.pbc = pbc
        if pbc == '2d':
            self.pbc = PeriodicBoundary2D(mesh)
        elif pbc == '1d':
            self.pbc = PeriodicBoundary1D(mesh)


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
            'save_restart_data': sim_helpers.save_restart_data,
            'save_ndt': sim_helpers.save_ndt,
            'save_m': Simulation._save_m_incremental,
            'save_averages': sim_helpers.save_ndt,
            'save_vtk': self.save_vtk,
            'save_field': Simulation._save_field_incremental,
            'render_scene': Simulation._render_scene_incremental,
            'switch_off_H_ext': Simulation.switch_off_H_ext,
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

    def initialise_vortex(self, type, center=None, **kwargs):
        """
        Initialise the magnetisation to a pattern that resembles a vortex state.
        This can be used as an initial guess for the magnetisation, which should
        then be relaxed to actually obtain the true vortex pattern (in case it is
        energetically stable).

        If `center` is None, the vortex core centre is placed at the sample centre
        (which is the point where each coordinate lies exactly in the middle between
        the minimum and maximum coordinate for each component). The vortex lies in
        the x/y-plane (i.e. the magnetisation is constant in z-direction). The
        magnetisation pattern is such that m_z=1 in the vortex core centre, and it
        falls off in a radially symmetric way.

        The exact vortex profile depends on the argument `type`. Currently the
        following types are supported:

           'simple':

               m_z falls off in a radially symmetric way until m_z=0 at
               distance `r` from the centre.

           'feldtkeller':

               m_z follows the profile m_z = exp(-2*r^2/beta^2), where `beta`
               is a user-specified parameter.

        All provided keyword arguments are passed on to the helper functions which
        implement the vortex profiles (e.g., finmag.util.helpers.vortex_simple or
        finmag.util.helpers.vortex_feldtkeller). See their documentation for details
        and other allowed arguments.

        """
        coords = np.array(self.mesh.coordinates())
        if center == None:
            center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))

        vortex_funcs = {
            'simple': helpers.vortex_simple,
            'feldtkeller': helpers.vortex_feldtkeller,
            }

        kwargs['center'] = center

        try:
            fun_m_init = vortex_funcs[type](**kwargs)
            log.debug("Initialising vortex of type '{}' with arguments: {}".format(type, kwargs))
        except KeyError:
            raise ValueError("Vortex type must be one of {}. Got: {}".format(vortex_funcs.keys(), type))

        self.set_m(fun_m_init)

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
        # Make sure that interaction names are unique
        if interaction.name in [i.name for i in self.llg.effective_field.interactions]:
            raise ValueError("Interaction names must be unique, but an "
                             "interaction with the same name already "
                             "exists: {}".format(interaction.name))

        log.debug("Adding interaction %s to simulation '%s'" % (str(interaction),self.name))
        interaction.setup(self.S3, self.llg._m, self.llg._Ms_dg, self.unit_length)
        # TODO: The following feels somewhat hack-ish because we
        #       explicitly check for TimeZeeman and it's likely that
        #       there will be other classes in the future that also
        #       come with time updates which would then also have to
        #       be added here by hand. Is there a more elegant and
        #       automatic solution?
        if isinstance(interaction, TimeZeeman):
            # The following line ensures that the time integrator is notified
            # about the correct field values at the time steps it chooses.
            with_time_update = interaction.update
            # The following line ensures that the field value is updated
            # correctly whenever the time integration reaches a scheduler
            # "checkpoint" (i.e. whenever integrator.advance_time(t) finishes
            # successfully).
            self.callbacks_at_scheduler_events.append(interaction.update)
        self.llg.effective_field.add(interaction, with_time_update)

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
            try:
                interaction = self.get_interaction(name)
                res = interaction.compute_energy()
            except ValueError:
                res = 0.0
        return res

    def has_interaction(self, interaction_name):
        """
        Returns True if an interaction with the given name exists, and False otherwise.

        *Arguments*

        interaction_name: string

            Name of the interaction.

        """
        try:
            self.llg.effective_field.get_interaction(interaction_name)
            res = True
        except ValueError:
            res = False
        return res

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
        return self.llg.effective_field.get_interaction(interaction_name)

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
        try:
            H = self.get_interaction("Zeeman")
            H.set_value(H_ext)
        except ValueError:
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

        if region:
            V_region = self._get_region_space(region)
            field = df.interpolate(field, V_region)

        return field

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

    def probe_field(self, field_type, pts):
        """
        Probe the field of type `field_type` at point(s) `pts`, where
        the point coordinates must be specified in mesh coordinates.

        See the documentation of the method get_field_as_dolfin_function
        to know which ``field_type`` is allowed, and helpers.probe for the
        shape of ``pts``.

        """
       
        return helpers.probe(self.get_field_as_dolfin_function(field_type), pts)

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
                
            self.tablewriter.entities['steps'] = {
                'unit': '<1>',
                'get': lambda sim: sim.integrator.stats()['nsteps'],
                'header': 'steps'}
            
            self.tablewriter.entities['last_step_dt'] = {
                'unit': '<1>',
                'get': lambda sim: sim.integrator.stats()['hlast'],
                'header': 'last_step_dt'}
            
            self.tablewriter.update_entity_order()
        
        else:
            log.warning("Cannot create integrator - exists already: {}".format(self.integrator))
        return self.integrator
    
    def set_tol(self, reltol=1e-6, abstol=1e-6):
        """
        Set the tolerences of the default integrator.
        """
        if not hasattr(self, "integrator"):
            self.create_integrator()
        
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
        exit_at = events.StopIntegrationEvent(t)
        self.scheduler._add(exit_at)

        run_with_schedule(self.integrator, self.scheduler, self.callbacks_at_scheduler_events)
        # The following line is necessary because the time integrator may
        # slightly overshoot the requested end time, so here we make sure
        # that the field values represent that requested time exactly.
        self.llg.effective_field.update(t)
        log.info("Simulation has reached time t = {:.2g} s.".format(self.t))

        self.scheduler._remove(exit_at)

    def relax(self, save_vtk_snapshot_as=None, save_restart_data_as=None, stopping_dmdt=1.0,
              dt_limit=1e-10, dmdt_increased_counter_limit=10):
        """
        Run the simulation until the magnetisation has relaxed.

        This means the magnetisation reaches a state where its change over time
        at each node is smaller than the threshold `stopping_dm_dt` (which
        should be given in multiples of degree/nanosecond).

        If `save_vtk_snapshot_as` and/or `restart_restart_data_as` are
        specified, a vtk snapshot and/or restart data is saved to a
        file with the given name. This can also be achieved using the
        scheduler but provides a slightly more convenient mechanism.
        Note that any previously existing files with the same name
        will be automatically overwritten!

        """
        if not hasattr(self, "integrator"):
            self.create_integrator()
        log.info("Simulation will run until relaxation of the magnetisation.")
        log.debug("Relaxation parameters: stopping_dmdt={} (degrees per nanosecond), "
                  "dt_limit={}, dmdt_increased_counter_limit={}".format(
                              stopping_dmdt, dt_limit, dmdt_increased_counter_limit))

        if hasattr(self, "relaxation"):
            del(self.relaxation)

        self.relaxation = events.RelaxationEvent(self, stopping_dmdt*ONE_DEGREE_PER_NS, dmdt_increased_counter_limit, dt_limit)
        self.scheduler._add(self.relaxation)

        run_with_schedule(self.integrator, self.scheduler, self.callbacks_at_scheduler_events)
        self.integrator.reinit()
        self.set_m(self.m)
        log.info("Relaxation finished at time t = {:.2g}.".format(self.t))

        self.scheduler._remove(self.relaxation)
        del(self.relaxation.sim) # help the garbage collection by avoiding circular reference

        # Save a vtk snapshot and/or restart data of the relaxed state.
        if save_vtk_snapshot_as is not None:
            self.save_vtk(save_vtk_snapshot_as, overwrite=True)
        if save_restart_data_as is not None:
            self.save_restart_data(save_restart_data_as)

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

        self.reset_time(data["simtime"] if (t0 == None) else t0)

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

    save_averages = sim_helpers.save_ndt
    save_ndt = sim_helpers.save_ndt
    hysteresis = hyst
    hysteresis_loop = hyst_loop

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

    def _get_field_saver(self, field_name, filename=None, overwrite=False, incremental=False):
        if filename is None:
            filename = '{}_{}.npy'.format(self.sanitized_name, field_name.lower())
        if not filename.endswith('.npy'):
            filename += '.npy'

        s = None
        if self.field_savers.has_key(filename) and self.field_savers[filename].incremental == incremental:
            s = self.field_savers[filename]

        if s is None:
            s = FieldSaver(filename, overwrite=overwrite, incremental=incremental)
            self.field_savers[filename] = s

        return s

    def save_field(self, field_name, filename=None, incremental=False, overwrite=False, region=None):
        """
        Save the given field data to a .npy file.

        *Arguments*

        field_name : string

            The name of the field to be saved. This should be either 'm'
            or the name of one of the interactions present in the
            simulation (e.g. Demag, Zeeman, Exchange, UniaxialAnisotropy).

        filename : string

            Output filename. If not specified, a default name will be
            generated automatically based on the simulation name and the
            name of the field to be saved. If a file with the same name
            already exists, an exception of type IOError will be raised.

        incremental : bool

        region:

            Some identifier that uniquely identifies a mesh region. This required
            that the method `mark_regions` has been called previously so that the
            simulation knows about the regions and their IDs.

        """
        field_data = self.get_field_as_dolfin_function(field_name, region=region)
        field_saver = self._get_field_saver(field_name, filename, incremental=incremental, overwrite=overwrite)
        field_saver.save(field_data.vector().array())

    save_m = sim_helpers.save_m

    def _save_field_incremental(self, field_name, filename=None, overwrite=False):
        self.save_field(field_name, filename, incremental=True, overwrite=overwrite)

    def _save_m_incremental(self, filename=None, overwrite=False):
        self.save_field('m', filename, incremental=True, overwrite=overwrite)

    def mesh_info(self):
        """
        Return a string containing some basic information about the
        mesh (such as the number of cells, interior/surface triangles,
        vertices, etc.).

        Also print a distribution of edge lengths present in the mesh
        and how they compare to the exchange length and the Bloch
        parameter (if these can be computed, which requires an
        exchange interaction (plus anisotropy for the Bloch
        parameter); note that for spatially varying exchange strength
        and anisotropy constant the average values are used). This
        information is relevant to estimate whether the mesh
        discretisation is too coarse and might result in numerical
        artefacts (see W. Rave, K. Fabian, A. Hubert, "Magnetic states
        ofsmall cubic particles with uniaxial anisotropy", J. Magn.
        Magn. Mater. 190 (1998), 332-348).

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
            A = self.llg.effective_field.exchange.A.vector().array().mean()
            Ms = self.llg.Ms.vector().array().mean()
            l_ex = exchange_length(A, Ms)
            info_string += added_info(l_ex, 'exchange length', 'l_ex')
            if hasattr(self.llg.effective_field, "anisotropy"):
                K1 = float(self.llg.effective_field.anisotropy.K1)
                l_bloch = bloch_parameter(A, K1)
                info_string += added_info(l_bloch, 'Bloch parameter', 'l_bloch')

        return info_string

    def render_scene(self, region=None, **kwargs):
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
            return render_paraview_scene(filename, **kwargs)

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


class NormalModeSimulation(Simulation):
    """
    Thin wrapper around the Simulation class to make normal mode
    computations using the ringdown method more convenient.

    """
    def __init__(self, *args, **kwargs):
        super(NormalModeSimulation, self).__init__(*args, **kwargs)
        # Internal variables to store parameters/results of the ringdown method
        self.m_snapshots_filename = None
        self.t_step_ndt = None
        self.t_step_m = None
        self.fft_freqs = None
        self.fft_mx = None
        self.fft_my = None
        self.fft_mz = None

        # Internal variables to store parameters/results of the (generalised) eigenvalue method
        self.eigenfreqs = None
        self.eigenvecs = None

        # Internal variables to store the matrices which define the generalised eigenvalue problem
        self.A = None
        self.M = None
        self.D = None

    def run_ringdown(self, t_end, alpha, H_ext=None, reset_time=True, clear_schedule=True,
                     save_ndt_every=None, save_vtk_every=None, save_m_every=None,
                     vtk_snapshots_filename=None, m_snapshots_filename=None,
                     overwrite=False):
        """
        Run the ringdown phase of a normal modes simulation, optionally saving
        averages, vtk snapshots and magnetisation snapshots to the respective
        .ndt, .pvd and .npy files. Note that by default existing snapshots will
        not be overwritten. Use `overwrite=True` to achieve this.
        Also note that `H_ext=None` has the same effect as `H_ext=[0, 0, 0]`, i.e.
        any existing external field will be switched off during the ringdown phase.
        If you would like to have a field applied during the ringdown, you need to
        explicitly provide it.

        This function essentially wraps up the re-setting of parameters such as
        the damping value, the external field and the scheduled saving of data
        into a single convenient function call, thus making it less likely to
        forget any settings.

        The following snippet::

            sim.run_ringdown(t_end=10e-9, alpha=0.02, H_ext=[1e5, 0, 0],
                            save_m_every=1e-11, m_snapshots_filename='sim_m.npy',
                            save_ndt_every=1e-12)

        is equivalent to::

            sim.clear_schedule()
            sim.alpha = 0.02
            sim.reset_time(0.0)
            sim.set_H_ext([1e5, 0, 0])
            sim.schedule('save_ndt', every=save_ndt_every)
            sim.schedule('save_vtk', every=save_vtk_every, filename=vtk_snapshots_filename)
            sim.run_until(10e-9)

        """
        if reset_time:
            self.reset_time(0.0)
        if clear_schedule:
            self.clear_schedule()

        self.alpha = alpha
        if H_ext == None:
            if self.has_interaction('Zeeman'):
                log.warning("Found existing Zeeman field (for the relaxation "
                            "stage). Switching it off for the ringdown. "
                            "If you would like to keep it, please specify the "
                            "argument 'H_ext' explicitly.")
                self.set_H_ext([0, 0, 0])
        else:
            self.set_H_ext(H_ext)
        if save_ndt_every:
            self.schedule('save_ndt', every=save_ndt_every)
            log.debug("Setting self.t_step_ndt = {}".format(save_ndt_every))
            self.t_step_ndt = save_ndt_every

        def schedule_saving(which, every, filename, default_suffix):
            try:
                dirname, basename = os.path.split(filename)
                if dirname != '' and not os.path.exists(dirname):
                    os.makedirs(dirname)
            except AttributeError:
                dirname = os.curdir
                basename = self.name + default_suffix
            outfilename = os.path.join(dirname, basename)
            self.schedule(which, every=every, filename=outfilename, overwrite=overwrite)

            if which == 'save_m':
                # Store the filename so that we can later compute normal modes more conveniently
                self.m_snapshots_filename = outfilename

        if save_vtk_every:
            schedule_saving('save_vtk', save_vtk_every, vtk_snapshots_filename, '_m.pvd')

        if save_m_every:
            schedule_saving('save_m', save_m_every, m_snapshots_filename, '_m.npy')
            log.debug("Setting self.t_step_m = {}".format(save_m_every))
            self.t_step_m = save_m_every

        self.run_until(t_end)

    def _compute_spectrum(self, use_averaged_m=False, mesh_region=None, **kwargs):
        if use_averaged_m:
            log.warning("Using the averaged magnetisation to compute the spectrum is not recommended "
                        "because certain symmetric modes are likely to be missed.")
            if mesh_region != None:
                # TODO: we might still be able to compute this if the
                # user saved the averaged magnetisation in the
                # specified region to the .ndt file. We should check that.
                log.warning("Ignoring argument 'mesh_region' because "
                            "'use_averaged_m' was set to True.")
            filename = self.ndtfilename
        else:
            if self.m_snapshots_filename is None:
                log.warning("The spatially averaged magnetisation was not saved during run_ringdown, "
                            "thus it cannot be used to compute the spectrum. Falling back to using the  "
                            "averaged magnetisation (which is not recommended because it is likely to "
                            "miss normal modes which have certain symmetries!).")
                if mesh_region != None:
                    log.warning("Ignoring argument 'mesh_region' because the "
                                "spatially resolved magnetisation was not saved "
                                "during the ringdown.")
                filename = self.ndtfilename
            else:
                # Create a wildcard pattern so that we can read the files using 'glob'.
                filename = re.sub('\.npy$', '*.npy', self.m_snapshots_filename)
        log.debug("Computing normal mode spectrum from file(s) '{}'.".format(filename))

        # Derive a sensible value of t_step. Use the value in **kwargs
        # if one was provided, or else the one specified during a previous
        # call of run_ringdown().
        t_step = kwargs.pop('t_step', None)#
        if t_step == None:
            if use_averaged_m:
                t_step = self.t_step_ndt
            else:
                t_step = self.t_step_m
        # XXX TODO: Use t_step_npy if computing the spectrum from .npy files.
        if t_step == None:
            raise ValueError(
                "No sensible default for 't_step' could be determined. "
                "(It seems like 'run_ringdown()' was not run, or it was not "
                "given a value for its argument 'save_ndt_every'). Please "
                "call sim.run_ringdown() or provide the argument 't_step' "
                "explicitly.")

        if mesh_region != None:
            # If a mesh region is specified, restrict computation of the
            # spectrum to the corresponding mesh vertices.
            submesh = self.get_submesh(mesh_region)
            try:
                # Legacy syntax (for dolfin <= 1.2 or so).
                # TODO: This should be removed in the future once dolfin 1.3 is released!
                parent_vertex_indices = submesh.data().mesh_function('parent_vertex_indices').array()
            except RuntimeError:
                # This is the correct syntax now, see:
                # http://fenicsproject.org/qa/185/entity-mapping-between-a-submesh-and-the-parent-mesh
                parent_vertex_indices = submesh.data().array('parent_vertex_indices', 0)
            kwargs['restrict_to_vertices'] = parent_vertex_indices

        self.psd_freqs, self.psd_mx, self.psd_my, self.psd_mz = \
            power_spectral_density(filename, t_step=t_step, **kwargs)

    def plot_spectrum(self, t_step=None, t_ini=None, t_end=None, subtract_values='average',
                      components="xyz", xlim=None, ticks=5, figsize=None, title="",
                      outfilename=None, use_averaged_m=False, mesh_region=None):
        """
        Plot the normal mode spectrum of the simulation. If
        `mesh_region` is not None, restricts the computation to mesh
        vertices in that region (which must have been specified with
        `sim.mark_regions` before).

        This is a convenience wrapper around the function
        finmag.util.fft.plot_power_spectral_density. It accepts the
        same keyword arguments, but provides sensible defaults for
        some of them so that it is more convenient to use. For
        example, `ndt_filename` will be the simulation's .ndt file by
        default, and t_step will be taken from the value of the
        argument `save_ndt_every` when sim.run_ringdown() was run.

        The default method to compute the spectrum is the one described
        in [1], which uses a spatially resolved Fourier transform
        to compute the local power spectra and then integrates these over
        the sample (which yields the total power over the sample at each
        frequency). This is the recommended way to compute the spectrum,
        but it requires the spatially resolved magnetisation to be saved
        during the ringdown (provide the arguments `save_m_every` and
        `m_snapshots_filename` to `sim.run_ringdown()` to achieve this).

        An alternative method (used when `use_averaged_m` is True) simply
        computes the Fourier transform of the spatially averaged magnetisation
        and plots that. However, modes with certain symmetries (which are
        quite common) will not be detected by this method, so it is not
        recommended to use it. It will be used as a fallback, however,
        if the spatially resolved magnetisation was not saved during the
        ringdown phase.

        [1] McMichael, Stiles, "Magnetic normal modes of nanoelements", J Appl Phys 97 (10), 10J901, 2005.

        """
        self._compute_spectrum(t_step=t_step, t_ini=t_ini, t_end=t_end, subtract_values=subtract_values, use_averaged_m=use_averaged_m, mesh_region=mesh_region)

        fig = _plot_spectrum(self.psd_freqs, self.psd_mx, self.psd_my, self.psd_mz,
                             components=components, xlim=xlim, ticks=ticks,
                             figsize=figsize, title=title, outfilename=outfilename)
        return fig

    def _get_psd_component(self, component):
        try:
            res = {'x': self.psd_mx,
                   'y': self.psd_my,
                   'z': self.psd_mz
                   }[component]
        except KeyError:
            raise ValueError("Argument `component` must be exactly one of 'x', 'y', 'z'.")
        return res

    def find_peak_near_frequency(self, f_approx, component, use_averaged_m=False):
        """
        XXX TODO: Write me!

        The argument `use_averaged_m` has the same meaning as in the `plot_spectrum`
        method. See its documentation for details.
        """
        if f_approx is None:
            raise TypeError("Argument 'f_approx' must not be None.")
        if not isinstance(component, types.StringTypes):
            raise TypeError("Argument 'component' must be of type string.")

        psd_cmpnt = self._get_psd_component(component)
        if self.psd_freqs == None or self.psd_mx == None or \
                self.psd_my ==None or self.psd_mz == None:
            self._compute_spectrum(self, use_averaged_m=use_averaged_m)

        return find_peak_near_frequency(f_approx, self.psd_freqs, psd_cmpnt)

    def plot_peak_near_frequency(self, f_approx, component, **kwargs):
        """
        Convenience function for debugging which first finds a peak
        near the given frequency and then plots the spectrum together
        with a point marking the detected peak.

        Internally, this calls `sim.find_peak_near_frequency` and
        `sim.plot_spectrum()` and accepts all keyword arguments
        supported by these two functions.

        """
        peak_freq, peak_idx = self.find_peak_near_frequency(f_approx, component)
        psd_cmpnt = self._get_psd_component(component)
        fig = self.plot_spectrum(**kwargs)
        fig.gca().plot(self.psd_freqs[peak_idx] / 1e9, psd_cmpnt[peak_idx], 'bo')
        return fig

    def export_normal_mode_animation_from_ringdown(self, npy_files, f_approx=None, component=None,
                                                   peak_idx=None, outfilename=None, directory='',
                                                   t_step=None, scaling=0.2, dm_only=False,
                                                   num_cycles=1, num_frames_per_cycle=20,
                                                   use_averaged_m=False):
        """
        XXX TODO: Complete me!

        If the exact index of the peak in the FFT array is known, e.g.
        because it was computed via `sim.find_peak_near_frequency()`,
        then this can be given as the argument `peak_index`.
        Otherwise, `component` and `f_approx` must be given and these
        are passed on to `sim.find_peak_near_frequency()` to determine
        the exact location of the peak.

        The output filename can be specified via `outfilename`. If this
        is None then a filename of the form 'normal_mode_N__xx.xxx_GHz.pvd'
        is generated automatically, where N is the peak index and xx.xx
        is the frequency of the peak (as returned by the function
        `sim.find_peak_near_frequency()`).


        *Arguments*:

        f_approx:  float

           Find and animate peak closest to this frequency. This is
           ignored if 'peak_idx' is given.

        component:  str

           The component (one of 'x', 'y', 'z') of the FFT spectrum
           which should be searched for a peak. This is ignored if
           peak_idx is given.

        peak_idx:  int

           The index of the peak in the FFT spectrum. This can be
           obtained by calling `sim.find_peak_near_frequency()`.
           Alternatively, if the arguments `f_approx` and `component`
           are given then this index is computed automatically. Note
           that if `peak_idx` is given the other two arguments are
           ignored.

        use_averaged_m:  bool

           Determines the method used to compute the spectrum. See
           the method `plot_spectrum()` for details.

        """
        if self.psd_freqs == None or self.psd_mx == None or \
                self.psd_my ==None or self.psd_mz == None:
            self._compute_spectrum(self, use_averaged_m=use_averaged_m)

        if peak_idx is None:
            if f_approx is None or component is None:
                raise ValueError("Please specify either 'peak_idx' or both 'f_approx' and 'component'.")
            peak_freq, peak_idx = self.find_peak_near_frequency(f_approx, component)
        else:
            if f_approx != None:
                log.warning("Ignoring argument 'f_approx' because 'peak_idx' was specified.")
            if component != None:
                log.warning("Ignoring argument 'component' because 'peak_idx' was specified.")
            peak_freq = self.psd_freqs[peak_idx]

        if outfilename is None:
            if directory is '':
                raise ValueError("Please specify at least one of the arguments 'outfilename' or 'directory'")
            outfilename = 'normal_mode_{}__{:.3f}_GHz.pvd'.format(peak_idx, peak_freq / 1e9)
        outfilename = os.path.join(directory, outfilename)

        if t_step == None:
            if (self.t_step_ndt != None) and (self.t_step_m != None) and \
                    (self.t_step_ndt != self.t_step_m):
                log.warning("Values of t_step for previously saved .ndt and .npy data differ! ({} != {}). Using t_step_ndt, but please double-check this is what you want.".format(self.t_step_ndt, self.t_step_m))
        t_step = t_step or self.t_step_ndt
        export_normal_mode_animation_from_ringdown(npy_files, outfilename, self.mesh, t_step,
                                                   peak_idx, dm_only=dm_only, num_cycles=num_cycles,
                                                   num_frames_per_cycle=num_frames_per_cycle)

    def compute_normal_modes(self, n_values=10, discard_negative_frequencies=False, filename_mat_A=None, filename_mat_M=None, use_generalized=True,
                             tol=1e-8, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, Minv=None, OPinv=None, mode='normal',
                             force_recompute_matrices=False, check_hermitian=False):
        """
        Compute the eigenmodes of the simulation by solving a generalised
        eigenvalue problem and return the computed eigenfrequencies and
        eigenvectors.  The simulation must be relaxed for this to yield
        sensible results.

        All keyword arguments not mentioned below are directly passed on
        to `scipy.sparse.linalg.eigs`, which is used internally to solve
        the generalised eigenvalue problem.


        *Arguments*

        n_values:

            The number of eigenmodes to compute (returns the
            `n_values` smallest ones).

        discard_negative_frequencies:

            For every eigenmode there is usually a corresponding mode
            with the negative frequency which otherwise looks exactly
            the same (at least in sufficiently symmetric situations).
            If `discard_negative_frequencies` is True then these are
            discarded. Use this at your own risk, however, since you
            might miss genuinely different eigenmodes. Default is
            False.

        filename_mat_A:
        filename_mat_M:

            If given (default: None), the matrices A and M which
            define the generalised eigenvalue problem are saved to
            files with the specified names. This can be useful for
            debugging or later post-processing to avoid having to
            recompute these matrices.

        use_generalized:

            If True (the default), solve a generalised eigenvalue
            problem.

        force_recompute_matrices:

            If False (the default), the matrices defining the
            generalised eigenvalue problem are not recomputed if they
            have been computed before (this dramatically speeds up
            repeated computation of eigenmodes). Set to True to
            force recomputation.

        check_hermitian:

            If True, check whether the matrix A used in the generalised
            eigenproblem is Hermitian and print a warning if this is not
            the case. This is a temporary debugging flag which is likely
            to be removed again in the future. Default value: False.


        *Returns*

        A pair (omega, eigenvectors), where omega is a list of the `n_values`
        smallest eigenfrequencies and `eigenvectors` is a rectangular matrix
        whose columns are the corresponding eigenvectors (in the same order).

        """
        if use_generalized:
            if (self.A == None or self.M == None) or force_recompute_matrices:
                self.A, self.M, _, _ = compute_generalised_eigenproblem_matrices( \
                    self, frequency_unit=1e9, filename_mat_A=filename_mat_A, filename_mat_M=filename_mat_M, check_hermitian=check_hermitian)
            else:
                log.debug('Re-using previously computed eigenproblem matrices.')

            omega, w = compute_normal_modes_generalised(self.A, self.M, n_values=n_values, discard_negative_frequencies=discard_negative_frequencies,
                                                        tol=tol, sigma=sigma, which=which, v0=v0, ncv=ncv, maxiter=maxiter, Minv=Minv, OPinv=OPinv, mode=mode)
        else:
            if self.D == None or force_recompute_matrices:
                self.D = compute_eigenproblem_matrix(self, frequency_unit=1e9)
            else:
                log.debug('Re-using previously computed eigenproblem matrix.')

            omega, w = compute_normal_modes(self.D, n_values, sigma=0.0, tol=tol, which='LM')
            omega = np.real(omega)  # any imaginary part is due to numerical inaccuracies so we ignore them

        self.eigenfreqs = omega
        self.eigenvecs = w

        return omega, w


    def export_normal_mode_animation(self, k, filename=None, directory='', dm_only=False, num_cycles=1, num_snapshots_per_cycle=20, scaling=0.2, framerate=5, **kwargs):
        """
        XXX TODO: Complete me!

        Export an animation of the `k`-th eigenmode, where the value of `k`
        refers to the index of the corresponding mode frequency and eigenvector
        in the two arrays returnd by `sim.compute_normal_modes`. If that method
        was called multiple times the results of the latest call are used.

        """
        if self.eigenfreqs is None or self.eigenvecs is None:
            log.debug("Could not find any precomputed eigenmodes. Computing them now.")
            self.compute_normal_modes(max(k, 10))

        if filename is None:
            if directory is '':
                raise ValueError("Please specify at least one of the arguments 'filename' or 'directory'")
            filename = 'normal_mode_{}__{:.3f}_GHz.pvd'.format(k, self.eigenfreqs[k])
        filename = os.path.join(directory, filename)

        basename, suffix = os.path.splitext(filename)

        # Export VTK animation
        if suffix == '.pvd':
            pvd_filename = filename
        elif suffix in ['.jpg', '.avi']:
            pvd_filename = basename + '.pvd'
        else:
            raise ValueError("Filename must end in one of the following suffixes: .pvd, .jpg, .avi.")

        export_normal_mode_animation(self, self.eigenfreqs[k], self.eigenvecs[:, k],
                                     pvd_filename, num_cycles=num_cycles,
                                     num_snapshots_per_cycle=num_snapshots_per_cycle,
                                     scaling=scaling, dm_only=dm_only)

        # Export image files
        if suffix == '.jpg':
            jpg_filename = filename
        elif suffix == '.avi':
            jpg_filename = basename + '.jpg'
        else:
            jpg_filename = None

        if jpg_filename != None:
            from finmag.util.visualization import render_paraview_scene
            render_paraview_scene(pvd_filename, outfile=jpg_filename, **kwargs)

        # Convert image files to movie
        if suffix == '.avi':
            movie_filename = filename
            pvd2avi(pvd_filename, outfile=movie_filename)

        # Return the movie if it was created
        res = None
        if suffix == '.avi':
            from IPython.display import HTML
            from base64 import b64encode
            video = open(movie_filename, "rb").read()
            video_encoded = b64encode(video)
            #video_tag = '<video controls alt="test" src="data:video/x-m4v;base64,{0}">'.format(video_encoded)
            #video_tag = '<a href="files/{0}">Link to video</a><br><br><video controls alt="test" src="data:video.mp4;base64,{1}">'.format(movie_filename, video_encoded)
            #video_tag = '<a href="files/{0}" target="_blank">Link to video</a><br><br><embed src="files/{0}"/>'.format(movie_filename)  # --> kind of works
            #video_tag = '<a href="files/{0}" target="_blank">Link to video</a><br><br><object data="files/{0}" type="video.mp4"/>'.format(movie_filename)  # --> kind of works
            #video_tag = '<a href="files/{0}" target="_blank">Link to video</a><br><br><embed src="files/{0}"/>'.format(basename + '.avi')  # --> kind of works

            # Display a link to the video
            video_tag = '<a href="files/{0}" target="_blank">Link to video</a>'.format(basename + '.avi')
            return HTML(data=video_tag)

    def plot_spatially_resolved_normal_mode(self, k, slice_z='z_max', components='xyz',
                                            figure_title=None, yshift_title=0.0,
                                            plot_powers=True, plot_phases=True, num_phase_colorbar_ticks=3,
                                            cmap_powers=plt.cm.jet, cmap_phases=plt.cm.hsv, vmin_powers=None,
                                            show_axis_labels=True, show_axis_frames=True,
                                            show_colorbars=True, figsize=None,
                                            outfilename=None, dpi=None):
        """
        Plot a spatially resolved profile of the k-th normal mode as
        computed by `sim.compute_normal_modes()`.

        *Returns*

        A `matplotlib.Figure` containing the plot.


        See the docstring of the function
        `finmag.util.normal_modes.plot_spatially_resolved_normal_mode`
        for details about the meaning of the arguments.

        """
        if self.eigenvecs == None:
            log.warning("No eigenvectors have been computed. Please call "
                        "`sim.compute_normal_modes()` to do so.")

        fig = plot_spatially_resolved_normal_mode(
            self, self.eigenvecs[:, k], slice_z=slice_z, components=components,
            figure_title=figure_title, yshift_title=yshift_title,
            plot_powers=plot_powers, plot_phases=plot_phases,
            cmap_powers=cmap_powers, cmap_phases=cmap_phases, vmin_powers=vmin_powers,
            show_axis_labels=show_axis_labels, show_axis_frames=show_axis_frames,
            show_colorbars=show_colorbars, figsize=figsize,
            outfilename=outfilename, dpi=dpi)
        return fig


def sim_with(mesh, Ms, m_init, alpha=0.5, unit_length=1, integrator_backend="sundials",
             A=None, K1=None, K1_axis=None, H_ext=None, demag_solver='FK',
             demag_solver_params={}, D=None, name="unnamed", sim_class=Simulation):
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
        demag = Demag(solver=demag_solver)
        if demag_solver_params != {}:
            demag.parameters.update(demag_solver_params)
            for (k, v) in demag_solver_params.items():
                log.debug("Setting demag solver parameter {}='{}' for simulation '{}'".format(k, v, sim.name))
        sim.add(demag)
    log.debug("Successfully created simulation '{}'".format(sim.name))

    return sim


def normal_mode_simulation(mesh, Ms, m_init, **kwargs):
    """
    Same as `sim_with` (it accepts the same keyword arguments apart from
    `sim_class`), but returns an instance of `NormalModeSimulation`
    instead of `Simulation`.

    """
    # Make sure we don't inadvertently create a different kind of simulation
    kwargs.pop('sim_class', None)

    return sim_with(mesh, Ms, m_init, sim_class=NormalModeSimulation, **kwargs)
