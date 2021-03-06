import time
import finmag
import logging
import shutil
import os
import types
import dolfin as df
import numpy as np
from datetime import datetime, timedelta
from finmag.util.consts import ONE_DEGREE_PER_NS
from finmag.util.meshes import nodal_volume

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
    sim.save_field(
        'm', filename=filename, incremental=incremental, overwrite=overwrite)


def create_backup_file_if_file_exists(filename, backupextension='.backup'):
    if os.path.exists(filename):
        backup_file_name = filename + backupextension
        shutil.copy(filename, backup_file_name)
        log.extremedebug("Creating backup %s of %s" %
                         (backup_file_name, filename))


def canonical_restart_filename(sim):
    return sim.sanitized_name + "-restart.npz"


def create_non_existing_parent_directories(filename):
    """
    Create parent directories of the given file if they do not already exist.

    """
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        log.debug(
            "Creating non-existing parent directory: '{}'".format(dirname))
        os.makedirs(dirname)


def save_restart_data(sim, filename=None):
    """Given a simulation object, this function saves the current
    magnetisation, and some integrator metadata into a file. """

    # create metadata
    integrator_stats = sim.integrator.stats()
    datetimetuple = datetime.now()
    drivertype = 'cvode'  # we should deduce this from sim object XXX
    simtime = sim.t
    # fix filename
    if filename == None:
        filename = canonical_restart_filename(sim)

    create_backup_file_if_file_exists(filename)
    create_non_existing_parent_directories(filename)

    np.savez_compressed(filename,
                        m=sim.integrator.llg._m_field.get_numpy_array_debug(),
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
    elif isinstance(filename_or_simulation, types.StringTypes):
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


def get_submesh(sim, region=None):
    """
    Return the submesh associated with the given region.

    """
    if region == None:
        submesh = sim.mesh
    else:
        region_id = sim._get_region_id(region)
        submesh = df.SubMesh(sim.mesh, sim.region_markers, region_id)
    return submesh


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

    # if params_precess == None:
    #    raise ValueError("No precession parameters given. Expected params_precess != None.")

    def set_simulation_parameters_and_schedule(sim, params, suffix=''):
        sim.alpha = params['alpha']
        sim.set_H_ext = params['H_ext']
        #if params['save_ndt_every'] != None: sim.schedule('save_ndt', filename=sim.name + '_precess.ndt', every=params['save_ndt_every'])
        if params['save_ndt_every'] != None:
            sim.schedule('save_ndt', every=params['save_ndt_every'])
        #if params['save_ndt_every'] != None: raise NotImplementedError("XXX FIXME: This is currently not implemented because we need a different .ndt filename but it cannot be changed at present.")
        if params['save_vtk_every'] != None:
            sim.schedule('save_vtk', filename=sim.name +
                         suffix + '.pvd', every=params['save_vtk_every'])
        if params['save_npy_every'] != None:
            sim.schedule('save_field', 'm', filename=sim.name +
                         suffix + '.npy', every=params['save_npy_every'])

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


def eta(sim, when_started):
    """
    Estimated time of simulation completion.
    Only works in conjunction with run_until.

    """
    elapsed_real_time = time.time() - when_started
    simulation_speed = sim.t / elapsed_real_time
    if simulation_speed > 0 and sim.t < sim.t_max:
        remaining_simulation_time = sim.t_max - sim.t
        remaining_real_time = remaining_simulation_time / simulation_speed
        # split up remaining_real_time in hours, minutes
        # and seconds for nicer formatting
        hours, remainder = divmod(remaining_real_time, 60)
        minutes, seconds = divmod(remainder, 60)
        log.info("Integrated up to t = {:.4} ns. "
                 "Predicted end in {:0>2}:{:0>2}:{:0>2}.".format(
                     sim.t * 1e9, int(hours), int(minutes), int(seconds)))


def plot_relaxation(sim, filename="relaxation.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t, max_dmdt_norms = np.array(zip(* sim.relaxation['dmdts']))
    ax.semilogy(t * 1e9, max_dmdt_norms / ONE_DEGREE_PER_NS, "ro")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("maximum dm/dt (1/ns)")
    threshold = sim.relaxation['stopping_dmdt'] / ONE_DEGREE_PER_NS
    ax.axhline(y=threshold, xmin=0.5, color="red", linestyle="--")
    ax.annotate("threshold", xy=(0.6, 1.1 * threshold), color="r")
    plt.savefig(filename)
    plt.close()


def skyrmion_number(self):
    """
    This function returns the skyrmion number calculated from the spin
    texture in this simulation instance.

    If the sim object is 3D, the skyrmion number is calculated from the
    magnetisation of the top surface (since the skyrmion number formula
    is only defined for 2D).
    
    Details about computing functionals over subsets of the mesh (the 
    method used in this function to calculate the skyrmion number) can be
    found at:
    
    https://bitbucket.org/fenics-project/dolfin/src/master/demo/undocumented/lift-drag/python/demo_lift-drag.py
    """
    
    mesh = self.mesh

    # Get m field and define skyrmion number integrand
    m = self.m_field.f
    integrand = -0.25 / np.pi * df.dot(m, df.cross(df.Dx(m, 0),
                                                   df.Dx(m, 1)))

    # determine if 3D system or not. Act accordingly.
    if self.mesh.topology().dim() == 3:
        mesh_coords = mesh.coordinates()
        z_max = max(mesh_coords[:, 2])

        # Define a subdomain which consists of the top surface of the geometry
        class Top(df.SubDomain):
            def inside(self, pt, on_boundary):
                x, y, z = pt
                return z >= z_max - df.DOLFIN_EPS and on_boundary

        # Define a set of Markers from a MeshFunction (which is a function that
        # can be evaluated at a set of mesh entities). Set all the marks to Zero.
        markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        markers.set_all(0)
        top = Top()
        # Redefine the marks on the top surface to 1.
        top.mark(markers, 1)
        # Define ds so that it only computes where the marker=1 (top surface)
        ds = df.ds[markers]
        
        # Assemble the integrand on the the original domain, but only over the
        # marked region (top surface).
        S = df.assemble(form = integrand * ds(1, domain=mesh))

    else:
        S = df.assemble(integrand * df.dx)
    
    return S


def skyrmion_number_density_function(self):
    """
    This function returns the skyrmion number density function calculated
    from the spin texture in this simulation instance. This function can be
    probed to determine the local nature of the skyrmion number.
    """

    integrand = -0.25 / np.pi * df.dot(self.m_field.f,
                                       df.cross(df.Dx(self.m_field.f, 0),
                                                df.Dx(self.m_field.f, 1)))

    # Build the function space of the mesh nodes.
    dofmap = self.S3.dofmap()
    S1 = df.FunctionSpace(self.S3.mesh(), "Lagrange", 1,
                          constrained_domain=dofmap.constrained_domain)

    # Find the skyrmion density at the mesh nodes using the above function
    # space.
    nodalSkx = df.dot(integrand, df.TestFunction(S1)) * df.dx
    nodalVolumeS1 = nodal_volume(S1, self.unit_length)
    skDensity = df.assemble(nodalSkx).array() * self.unit_length\
        ** self.S3.mesh().topology().dim() / nodalVolumeS1

    # Build the skyrmion number density dolfin function from the skDensity
    # array.
    skDensityFunc = df.Function(S1)
    skDensityFunc.vector()[:] = skDensity
    return skDensityFunc
