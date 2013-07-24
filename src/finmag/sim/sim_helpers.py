import finmag
import logging
import shutil
import os
import numpy as np
from datetime import datetime

log = logging.getLogger("finmag")


def save_ndt(sim):
    """
    Save the average field values (such as the magnetisation) to a file.

    The filename is derived from the simulation name (as given when the
    simulation was initialised) and has the extension .ndt'.
    """
    if sim.driver == 'cvode':
        log.debug("Saving data to ndt file at t={} "
                  "(sim.name={}).".format(sim.t, sim.name))
    else:
        raise NotImplementedError("Only cvode driver known.")
    sim.tablewriter.save()


def save_m(sim, filename=None, incremental=False, overwrite=False):
    """
    Convenience function to save the magnetisation to a file (as a numpy array).

    The following two calls do exactly the same thing:

        sim.save_m(...)
        sim.save_field('m', ...)

    Thus see the documentation of sim.save_field() for details on
    the arguments.

    """
    sim.save_field('m', filename=filename, incremental=incremental, overwrite=overwrite)


def create_backup_file_if_file_exists(filename, backupextension='.backup'):
    if os.path.exists(filename):
        backup_file_name = filename + backupextension
        shutil.copy(filename, backup_file_name)
        log.extremedebug("Creating backup %s of %s" % (backup_file_name, filename))


def canonical_restart_filename(sim):
    return sim.sanitized_name + "-restart.npz"


def save_restart_data(sim, filename=None):
    """Given a simulation object, this function saves the current
    magnetisation, and some integrator metadata into a file. """

    #create metadata
    integrator_stats = sim.integrator.stats()
    datetimetuple = datetime.now()
    drivertype = 'cvode'  # we should deduce this from sim object XXX
    simtime = sim.t
    # fix filename
    if filename == None:
        filename = canonical_restart_filename(sim)

    create_backup_file_if_file_exists(filename)

    np.savez_compressed(filename,
        m=sim.integrator.llg.m,
        stats=integrator_stats,
        simtime=simtime,
        datetime=datetimetuple,
        simname=sim.name,
        driver=drivertype)
    log.debug("Have saved restart data at t=%g to %s "
              "(sim.name=%s)" % (sim.t, filename, sim.name))


def load_restart_data(filename_or_simulation):
    """Given a file name, load the restart data saved in that file
    name and return as dictionary. If object is simulation instance,
    use canonical name."""
    if isinstance(filename_or_simulation, finmag.Simulation):
        filename = canonical_restart_filename(filename_or_simulation)
    elif isinstance(filename_or_simulation, str):
        filename = filename_or_simulation
    else:
        ValueError("Can only deal with simulations or filenames, "
                   "but not '%s'" % type(filename_or_simulation))
    data = np.load(filename)

    # strip of arrays where we do not want them:
    data2 = {}
    for key in data.keys():
        # the 'tolist()' command returns dictionary and datetime objects
        # when wrapped up in numpy array
        if key in ['stats', 'datetime', 'simtime', 'simname', 'driver']:
            data2[key] = data[key].tolist()
        else:
            data2[key] = data[key]
    return data2


def run_normal_modes_computation(sim, params_relax=None, params_precess=None):
    """
    Run a normal modes computation consisting of two stages. During the first
    stage, the simulation is relaxed with the given parameters. During the
    second ('precessional') stage, the computation is run up to a certain
    time. The magnetisation can be saved at regular intervals (specified by
    the corresponding parameters in params_relax and params_precess), and this
    can be used for post-processing such as plotting normal modes, etc.

    XXX TODO: Document the parameter dictionaries params_relax, params_precess.

    """
    default_params_relax = {
        'alpha': 1.0,
        'H_ext': None,
        'save_ndt_every': None,
        'save_vtk_every': None,
        'save_npy_every': None,
        'save_relaxed_vtk': True,
        'save_relaxed_npy': True,
        'filename': None
        }

    default_params_precess = {
        'alpha': 0.0,
        'H_ext': None,
        'save_ndt_every': 1e-11,
        'save_vtk_every': None,
        'save_npy_every': None,
        't_end': 10e-9,
        'filename': None
        }

    #if params_precess == None:
    #    raise ValueError("No precession parameters given. Expected params_precess != None.")

    def set_simulation_parameters_and_schedule(sim, params, suffix=''):
        sim.alpha = params['alpha']
        sim.set_H_ext = params['H_ext']
        #if params['save_ndt_every'] != None: sim.schedule('save_ndt', filename=sim.name + '_precess.ndt', every=params['save_ndt_every'])
        if params['save_ndt_every'] != None: sim.schedule('save_ndt', every=params['save_ndt_every'])
        #if params['save_ndt_every'] != None: raise NotImplementedError("XXX FIXME: This is currently not implemented because we need a different .ndt filename but it cannot be changed at present.")
        if params['save_vtk_every'] != None: sim.schedule('save_vtk', filename=sim.name + suffix + '.pvd', every=params['save_vtk_every'])
        if params['save_npy_every'] != None: sim.schedule('save_field', 'm', filename=sim.name + suffix + '.npy', every=params['save_npy_every'])

    if params_relax == None:
        pass  # skip the relaxation phase
    else:
        params = default_params_relax
        params.update(params_relax)
        set_simulation_parameters_and_schedule(sim, params, suffix='_relax')
        sim.relax()
        if params['save_relaxed_vtk']:
            sim.save_vtk(filename=sim.name + '_relaxed.pvd')
        if params['save_relaxed_npy']:
            sim.save_field('m', filename=sim.name + '_relaxed.npy')

    sim.reset_time(0.0)
    sim.clear_schedule()
    params = default_params_precess
    if params_precess != None:
        params.update(params_precess)
    set_simulation_parameters_and_schedule(sim, params, suffix='_precess')
    sim.run_until(params['t_end'])
