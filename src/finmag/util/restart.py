import datetime
import shutil
import logging
log = logging.getLogger('finmag')
import numpy as np
import finmag
import os.path


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
    datetimetuple = datetime.datetime.now()
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
    log.debug("Have saved restart data at t=%g to %s (sim.name=%s)" % (sim.t, filename, sim.name))


def load_restart_data(filename_or_simulation):
    """Given a file name, load the restart data saved in that file
    name and return as dictionary. If object is simulation instance,
    use canonical name."""
    if isinstance(filename_or_simulation, finmag.Simulation):
        filename = canonical_restart_filename(filename_or_simulation)
    elif isinstance(filename_or_simulation, str):
        filename = filename_or_simulation
    else:
        ValueError("Can only deal with simulations or filenames, but not '%s'" % \
            type(filename_or_simulation))
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
