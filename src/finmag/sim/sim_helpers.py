import finmag
import logging
import shutil
import os
import numpy as np
from datetime import datetime

log = logging.getLogger("finmag")


def save_ndt(sim):
    """Given the simulation object, saves one line to the ndt finalise
    (through the TableWriter object)."""
    if sim.driver == 'cvode':
        log.debug("Saving data to ndt file at t={} "
                  "(sim.name={}).".format(sim.t, sim.name))
    else:
        raise NotImplementedError("Only cvode driver known.")
    sim.tablewriter.save()


def save_field(sim, field_name, filename=None, overwrite=False):
    """
    Save the field data to a .npy file.

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

    *Returns*

    The output filename: either `filename` (if given) or the
    automatically generated name.

    """
    if filename is None:
        filename = '{}_{}.npy'.format(sim.name, field_name.lower())
    if not filename.endswith('.npy'):
        filename += '.npy'
    if os.path.exists(filename) and not overwrite == True:
        raise IOError("Could not save field '{}' to file '{}': "
                      "file already exists".format(field_name, filename))

    field = sim.get_field_as_dolfin_function(field_name)
    np.save(filename, field.vector().array())

    return filename


def create_backup_file_if_file_exists(filename, backupextension='.backup'):
    if os.path.exists(filename):
        backup_file_name = filename + backupextension
        shutil.copy(filename, backup_file_name)
        log.extremedebug("Creating backup %s of %s" % (backup_file_name, filename))


def canonical_restart_filename(sim):
    return sim.name + "-restart.npz"


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
