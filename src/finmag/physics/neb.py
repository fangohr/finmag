import os
import dolfin as df
import numpy as np
import inspect
from aeon import default_timer
import finmag.util.consts as consts

from finmag.util import helpers
from finmag.physics.effective_field import EffectiveField
from finmag.util.vtk_saver import VTKSaver
from finmag import Simulation
# from finmag.field import Field  # Change sim._m to new field class
                                  # in line 184
from finmag.native import sundials
import finmag.native.neb as native_neb

from finmag.util.fileio import Tablewriter, Tablereader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import colorConverter
from matplotlib.collections import PolyCollection, LineCollection

import logging
log = logging.getLogger(name="finmag")


def cartesian2spherical(xyz):
    """
    Transform cartesian coordinates (x, y, z)
    in spherical coordinates. The function only returns
    the (theta, phi) pair since
    the magnetisation is fixed at zero Temperature
    (the r-component is constant) and is
    fully characterised by two degrees of freedom.
    (We use this to specifically transform M coordinates)
    Are we asuming? that xyz is:
    [x1, x2, .... , y1, y2, ... , z1, z2 ...]
    """
    # Transform to a 3 x -- array
    xyz.shape = (3, -1)
    r_xy = np.sqrt(xyz[0, :] ** 2 + xyz[1, :] ** 2)
    theta = np.arctan2(r_xy, xyz[2, :])
    phi = np.arctan2(xyz[1, :], xyz[0, :])
    xyz.shape = (-1,)

    theta_phi = np.concatenate((theta, phi))
    # Return [theta1, theta2, ... , phi1, phi2, ... ]
    return theta_phi


def spherical2cartesian(theta_phi):
    """
    Returns the (x, y, z) cartesian components
    from spherical coordinates (theta, phi)
    for a r-component equal to 1
    (thus, (x,y,z) is normalised)
    We are assuming that theta_phi has the form

    [theta1, theta2, ... , phi1, phi2, ...]

    as in the previous function
    """
    theta_phi.shape = (2, -1)
    theta = theta_phi[0]
    phi = theta_phi[1]
    mxyz = np.zeros(3 * len(theta))
    mxyz.shape = (3, -1)
    mxyz[0, :] = np.sin(theta) * np.cos(phi)
    mxyz[1, :] = np.sin(theta) * np.sin(phi)
    mxyz[2, :] = np.cos(theta)
    mxyz.shape = (-1, )
    theta_phi.shape = (-1, )
    return mxyz


def check_boundary(theta_phi):
    """
    Rescale the theta, phi angles between
    two vectors when they are
    larger than specific values:

    * theta is undefined when is smaller than zero
      or larger than pi. Here we redefine to zero or pi
      in those cases

    * If phi is larger than pi: substract 2*pi to get
      the shorter arc that separate two vectors
      Similar if phi is smaller than -pi

    This makes the phi angle differences to be rescaled in the same
    range than the theta angle:  | Delta-phi | < pi

    """

    theta_phi.shape = (2, -1)
    theta = theta_phi[0]
    phi = theta_phi[1]

    theta[theta > np.pi] = np.pi
    theta[theta < 0] = 0

    phi[phi > np.pi] -= 2*np.pi
    phi[phi < -np.pi] += 2*np.pi
    theta_phi.shape = (-1,)


def cartesian2spherical_field(field_c, theta_phi):
    """
    Transform the cartesian (x, y, z) field coordinates to
    spherical coordinates (r, t, p) using the unit vectors
    transformation matrix:

    | sin t cos p  | sin t sin p | cos t  | | hx |   | h_r |
    | cos t cos p  | cos t sin p | -sin t | | hy | = | h_t |
    | -sin p       | cos p       |   0    | | hz |   | h_p |

    The function only returns the (t, p) = (theta, phi)
    coordinates of h since we asume that the r component
    is fixed

    """
    theta_phi.shape = (2, -1)
    theta = theta_phi[0]
    phi = theta_phi[1]

    field_s = np.zeros(theta_phi.shape)

    field_c.shape = (3, -1)
    field_s.shape = (2, -1)

    hx = field_c[0]
    hy = field_c[1]
    hz = field_c[2]

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    field_s[0] = (hx*cos_p + hy*sin_p)*cos_t - hz*sin_t
    field_s[1] = (-hx*sin_p + hy*cos_p)*sin_t  # sin t ???

    field_c.shape = (-1,)
    field_s.shape = (-1,)
    theta_phi.shape = (-1,)

    return field_s


def linear_interpolation_two(m0, m1, n):
    """
    Define a linear interpolation between
    two states of the energy band (m0, m1) to get
    an initial state. The interpolation is
    done in the magnetic moments that constitute the
    magnetic system. To achieve this,
    we use spherical coordinates and the
    angle difference between m0 and m1, is divided
    'n' times. Thus, we obtain (n - 2) interpolations:

    m0 = [theta0_1, theta0_1, ..., phi0_1, phi0_2, ...]

         [theta1_1 + dt1, theta1_2 + dt2, ..., phi1_1 + df1, phi1_2 + df2, ...]

         [theta2_1 + 2*dt1, theta2_2 + 2*dt2, ...,
            phi2_1 + 2*df1, phi2_2 + 2*df2, ...]

    ...

    m1 = [thetan_1 + n*dt1, thetan_2 + n*dt2, ...,
            phin_1 + n*df1, phin_2 + n*df2, ...]


    where [dt1, dt2, ..., df1, df2, ...] = (m0 - m1) / n
    and m0, m1 are in spherical coordinates, as specified before.

    The function return an array of arrays, with the interpolations
    [thetai_1, thetai_2, ..., phii_1, phii_2, ...] as the entries of
    the main array (thetai_1 = thetai + i*dt1, and so on)
    """

    theta_phi0 = cartesian2spherical(m0)
    theta_phi1 = cartesian2spherical(m1)

    # The differences
    dtheta = (theta_phi1 - theta_phi0) / (n + 1)

    coords = []
    for i in range(n):
        theta = theta_phi0 + (i + 1) * dtheta
        coords.append(theta)

    return coords


def normalise_m(a):
    """
    Normalise the magnetisation array.
    We asume:
    a = [mx1, mx2, ..., my1, my2, ..., mz1, mz2, ...]
    to transform this into

    [ [mx1, mx2, ...],
      [my1, my2, ...],
      [mz1, mz2, ...]
    ]
    normalise the matrix, and return again a  1 x -- array
    """
    # Transform to matrix
    a.shape = (3, -1)
    # Compute the array 'a' length
    lengths = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    # Normalise all the entries
    a[:] /= lengths
    # Return to original shape
    a.shape = (-1, )


def linear_interpolation(theta_phi0, theta_phi1):
    """
    Compute a linear interpolation between two states: theta_phi0 and
    theta_phi1, which are given in cartersian coordinates.
    The output is a normalised linear interpolation between both states.
    """

    m0 = spherical2cartesian(theta_phi0)
    m1 = spherical2cartesian(theta_phi1)

    # suppose m0 and m1 are quite close
    m2 = (m0 + m1) / 2.0
    normalise_m(m2)

    return cartesian2spherical(m2)


def compute_dm(m0, m1):

    dm = m0-m1
    length = len(dm)

    x = dm > np.pi
    dm[x] = 2*np.pi-dm[x]
    x = dm < -np.pi
    dm[x] += 2*np.pi

    dm = np.sqrt(np.sum(dm**2))/length
    return dm


class NEB_Sundials(object):
    """
    Nudged elastic band method by solving the differential equation using Sundials.
    """

    def __init__(self, sim, initial_images, interpolations=None, spring=5e5, name='unnamed'):
        """
          *Arguments*

              sim: the Simulation class

              initial_images: a list contain the initial value, which can have
              any of the forms accepted by the function 'finmag.util.helpers.
              vector_valued_function', for example,

                  initial_images = [(0,0,1), (0,0,-1)]

              or with given defined function

                  def init_m(pos):
                      x=pos[0]
                      if x<10:
                          return (0,1,1)
                      return (-1,0,0)

                  initial_images = [(0,0,1), (0,0,-1), init_m ]

              are accepted forms.

              interpolations : a list only contain integers and the length of
              this list should equal to the length of the initial_images minus 1,
              i.e., len(interpolations) = len(initial_images) - 1

              spring: the spring constant, a float value

              disable_tangent: this is an experimental option, by disabling the
              tangent, we can get a rough feeling about the local energy minima quickly.

        """

        self.sim = sim
        self.name = name
        self.spring = spring

        # Dolfin function of the new _m_field (instead of _m)
        self._m = sim.llg._m_field.f
        self.effective_field = sim.llg.effective_field

        if interpolations is None:
            interpolations = [0 for i in range(len(initial_images)-1)]

        self.initial_images = initial_images
        self.interpolations = interpolations

        if len(interpolations) != len(initial_images)-1:
            raise RuntimeError("""The length of interpolations should be equal to
                the length of the initial_images array minus 1, i.e.,
                len(interpolations) = len(initial_images) - 1""")

        if len(initial_images) < 2:
            raise RuntimeError("""At least two images must be provided
                               to create the energy band""")

        # the total image number including two ends
        self.total_image_num = len(initial_images) + sum(interpolations)
        self.image_num = self.total_image_num - 2

        self.nxyz = len(self._m.vector())/3

        self.coords = np.zeros(2*self.nxyz*self.total_image_num)
        self.last_m = np.zeros(self.coords.shape)

        self.Heff = np.zeros(2*self.nxyz*self.image_num)
        self.Heff.shape = (self.image_num, -1)

        self.tangents = np.zeros(self.Heff.shape)
        self.energy = np.zeros(self.total_image_num)

        self.springs = np.zeros(self.image_num)

        self.t = 0
        self.step = 0
        self.ode_count = 1
        self.integrator = None

        self.initial_image_coordinates()
        self.create_tablewriter()

    def create_tablewriter(self):
        entities_energy = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'energy': {'unit': '<J>',
                       'get': lambda sim: sim.energy,
                       'header': ['image_%d'%i for i in range(self.image_num+2)]}
            }

        self.tablewriter = Tablewriter('%s_energy.ndt'%(self.name), self, override=True, entities=entities_energy)

        entities_dm = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'dms': {'unit': '<1>',
                       'get': lambda sim: sim.distances,
                       'header': ['image_%d_%d'%(i, i+1) for i in range(self.image_num+1)]}
            }
        self.tablewriter_dm = Tablewriter('%s_dms.ndt'%(self.name), self, override=True, entities=entities_dm)

    def initial_image_coordinates(self):
        """
        generate the coordinates linearly.
        """

        image_id = 0
        self.coords.shape = (self.total_image_num, -1)
        for i in range(len(self.interpolations)):

            n = self.interpolations[i]

            self.sim.set_m(self.initial_images[i])
            m0 = self.sim.m

            self.coords[image_id][:] = cartesian2spherical(m0)
            image_id = image_id + 1

            self.sim.set_m(self.initial_images[i+1])
            m1 = self.sim.m

            coords = linear_interpolation_two(m0, m1, n)

            for coord in coords:
                self.coords[image_id][:] = coord[:]
                image_id = image_id + 1

        self.sim.set_m(self.initial_images[-1])
        m2 = self.sim.m
        self.coords[image_id][:] = cartesian2spherical(m2)

        for i in range(self.total_image_num):
            self._m.vector().set_local(spherical2cartesian(self.coords[i]))
            self.effective_field.update()
            self.energy[i] = self.effective_field.total_energy()

        self.coords.shape = (-1,)

    def save_vtks(self):
        """
        Save vtk files in different folders, according to the
        simulation name and step.
        Files are saved as simname_simstep/image_00000x.vtu
        """

        # Create the directory
        directory = 'vtks/%s_%d' % (self.name, self.step)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the vtk files with the finmag function
        # The last separator '_' is to distinguish the image
        # from its corresponding number, e.g. image_000001.pvd
        vtk_saver = VTKSaver('%s/image_.pvd' % (directory),
                             overwrite=True)

        self.coords.shape = (self.total_image_num, -1)

        for i in range(self.total_image_num):
            self._m.vector().set_local(spherical2cartesian(self.coords[i, :]))
            # set t =0, it seems that the parameter time is only
            # for the interface?
            vtk_saver.save_field(self._m, 0)

        self.coords.shape = (-1, )

    def save_npys(self):
        """
        Save npy files in different folders according to
        the simulation name and step
        Files are saved as: simname_simstep/image_x.npy
        """
        # Create directory as simname_simstep
        directory = 'npys/%s_%d' % (self.name, self.step)

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the images with the format: 'image_{}.npy'
        # where {} is the image number, starting from 0
        self.coords.shape = (self.total_image_num, -1)
        for i in range(self.total_image_num):
            name = os.path.join(directory, 'image_%d.npy' % i)
            np.save(name, spherical2cartesian(self.coords[i, :]))

        self.coords.shape = (-1, )

    def create_integrator(self, reltol=1e-6, abstol=1e-6, nsteps=10000):

        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.coords)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)

        self.integrator = integrator

    def compute_effective_field(self, y):

        y.shape = (self.total_image_num, -1)

        for i in range(self.image_num):
            # Redefine the angles if phi is larger than pi
            # (see the corresponding function)
            check_boundary(y[i + 1])
            # Transform the input 'y'  to cartesian to compute the fields
            self._m.vector().set_local(spherical2cartesian(y[i + 1]))
            #
            self.effective_field.update()
            # Compute effective field, which is the gradient of
            # the energy in the NEB method (derivative with respect to
            # the generalised coordinates)
            h = self.effective_field.H_eff
            # Transform to spherical coordinates
            self.Heff[i, :] = cartesian2spherical_field(h, y[i + 1])
            # Compute the total energy
            self.energy[i + 1] = self.effective_field.total_energy()

            # Compute the 'distance' or difference between neighbouring states
            # around y[i+1]. This is used to compute the spring force
            #
            dm1 = compute_dm(y[i], y[i + 1])
            dm2 = compute_dm(y[i + 1], y[i + 2])
            self.springs[i] = self.spring * (dm2 - dm1)

        # Use the native NEB to compute the tangents according to the
        # improved NEB method, developed by Henkelman and Jonsson
        # at: Henkelman et al., Journal of Chemical Physics 113, 22 (2000)
        native_neb.compute_tangents(y, self.energy, self.tangents)
        # native_neb.compute_springs(y,self.springs,self.spring)

        y.shape = (-1, )

    def sundials_rhs(self, time, y, ydot):

        self.ode_count += 1
        default_timer.start("sundials_rhs", self.__class__.__name__)

        self.compute_effective_field(y)

        ydot.shape = (self.total_image_num, -1)

        for i in range(self.image_num):
            h = self.Heff[i]
            t = self.tangents[i]
            sf = self.springs[i]

            h3 = h - np.dot(h, t)*t + sf*t

            ydot[i+1, :] = h3[:]

        ydot[0, :] = 0
        ydot[-1, :] = 0

        ydot.shape = (-1,)

        default_timer.stop("sundials_rhs", self.__class__.__name__)

        return 0

    def compute_distance(self):

        distance = []

        ys = self.coords
        ys.shape = (self.total_image_num, -1)
        for i in range(self.total_image_num-1):
            dm = compute_dm(ys[i], ys[i+1])
            distance.append(dm)

        ys.shape = (-1, )
        self.distances = np.array(distance)

    def run_until(self, t):

        if t <= self.t:
            return

        self.integrator.advance_time(t, self.coords)

        m = self.coords
        y = self.last_m

        m.shape = (self.total_image_num, -1)
        y.shape = (self.total_image_num, -1)
        max_dmdt = 0
        for i in range(1, self.image_num+1):
            dmdt = compute_dm(y[i], m[i]) / (t-self.t)
            if dmdt > max_dmdt:
                max_dmdt = dmdt

        m.shape = (-1,)
        y.shape = (-1,)
        self.last_m[:] = m[:]
        self.t = t

        return max_dmdt

    def relax(self, dt=1e-8, stopping_dmdt=1e4,
              max_steps=1000, save_npy_steps=100,
              save_vtk_steps=100):

        if self.integrator is None:
            self.create_integrator()

        log.debug("Relaxation parameters: "
                  "stopping_dmdt={} (degrees per nanosecond), "
                  "time_step={} s, max_steps={}.".format(stopping_dmdt,
                                                         dt, max_steps))

        for i in range(max_steps):

            if i % save_vtk_steps == 0:
                self.save_vtks()

            if i % save_npy_steps == 0:
                self.save_npys()

            self.step += 1

            cvode_dt = self.integrator.get_current_step()

            increment_dt = dt

            if cvode_dt > dt:
                increment_dt = cvode_dt

            dmdt = self.run_until(self.t+increment_dt)

            self.compute_distance()
            self.tablewriter.save()
            self.tablewriter_dm.save()

            log.debug("step: {:.3g}, step_size: {:.3g}"
                      " and max_dmdt: {:.3g}.".format(self.step,
                                                      increment_dt,
                                                      dmdt))

            if dmdt < stopping_dmdt:
                break

        log.info("Relaxation finished at time step = {:.4g}, "
                 "t = {:.2g}, call rhs = {:.4g} "
                 "and max_dmdt = {:.3g}".format(self.step,
                                                self.t,
                                                self.ode_count,
                                                dmdt))

        self.save_vtks()
        self.save_npys()

    def __adjust_coords_once(self):

        self.compute_effective_field(self.coords)
        self.compute_distance()

        average_dm = np.mean(self.distances)
        # What about a local minimum?
        energy_barrier = np.max(self.energy) - np.min(self.energy)

        dm_threshold = average_dm / 5.0
        energy_threshold = energy_barrier / 5.0
        to_be_remove_id = -1
        for i in range(self.image_num):
            e1 = self.energy[i+1] - self.energy[i]
            e2 = self.energy[i+2] - self.energy[i+1]
            if self.distances[i] < dm_threshold and \
                    self.distances[i+1] < dm_threshold \
                    and e1*e2 > 0 \
                    and abs(e1) < energy_threshold \
                    and abs(e2) < energy_threshold:
                to_be_remove_id = i+1
                break

        if to_be_remove_id < 0:
            return -1

        self.coords.shape = (self.total_image_num, -1)

        coords_list = []
        for i in range(self.total_image_num):
            coords_list.append(self.coords[i].copy())

        energy_diff = []
        for i in range(self.total_image_num-1):
            de = abs(self.energy[i]-self.energy[i+1])
            energy_diff.append(de)

        # if there is a saddle point, increase the weight
        # of the energy difference
        factor1 = 2.0
        for i in range(1, self.total_image_num-1):
            de1 = self.energy[i]-self.energy[i-1]
            de2 = self.energy[i+1]-self.energy[i]
            if de1*de2 < 0:
                energy_diff[i-1] *= factor1
                energy_diff[i] *= factor1

        factor2 = 2.0
        for i in range(2, self.total_image_num-2):
            de1 = self.energy[i-1]-self.energy[i-2]
            de2 = self.energy[i]-self.energy[i-1]
            de3 = self.energy[i+1]-self.energy[i]
            de4 = self.energy[i+2]-self.energy[i+1]
            if de1*de2 > 0 and de3*de4 > 0 and de2*de3 < 0:
                energy_diff[i-1] *= factor2
                energy_diff[i] *= factor2

        max_i = np.argmax(energy_diff)
        theta_phi = linear_interpolation(coords_list[max_i],
                                         coords_list[max_i+1])

        if to_be_remove_id < max_i:
            coords_list.insert(max_i+1, theta_phi)
            coords_list.pop(to_be_remove_id)
        else:
            coords_list.pop(to_be_remove_id)
            coords_list.insert(max_i+1, theta_phi)

        for i in range(self.total_image_num):
            m = coords_list[i]
            self.coords[i, :] = m[:]

        # print to_be_remove_id, max_i

        self.coords.shape = (-1, )

        return 0

    def adjust_coordinates(self):
        """
        Adjust the coordinates automatically.
        """

        for i in range(self.total_image_num/2):
            if self.__adjust_coords_once() < 0:
                break

        """
        self.compute_effective_field(self.coords)
        self.compute_distance()
        self.tablewriter.save()
        self.tablewriter_dm.save()

        self.step += 1
        self.tablewriter.save()
        self.tablewriter_dm.save()
        """

        log.info("Adjust the coordinates at step = {:.4g}, t = {:.6g},".format(self.step, self.t))


def plot_energy_2d(name, step=-1):
    """
    Plot the energy path at given step.

    name is the simulation name.
    """

    data = np.loadtxt('%s_energy.ndt' % name)
    dms = np.loadtxt('%s_dms.ndt' % name)

    if data.ndim == 1:
        data.shape = (1, -1)
        dms.shape = (1, -1)

    if step < 0:
        step = data[-1, 0]
        id = -1
    else:
        steps = abs(data[:, 0] - step)
        id = np.argmin(steps)
        step = data[id, 0]

    fig = plt.figure()
    xs = range(1, len(data[0, :]))

    for i in range(len(xs)):
        xs[i] = sum(dms[id, 1:i+1])

    plt.plot(xs, data[id, 1:], '.-')

    plt.legend()
    plt.grid()
    plt.ylabel('Energy (J)')
    plt.xlabel('Position in path (a.u.)')

    fig.savefig('energy_%d.pdf' % step)


def plot_energy_3d(name, key_steps=50, filename=None):

    data = np.loadtxt('%s_energy.ndt' % name)

    if data.ndim == 1:
        data.shape = (1, -1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # image index
    xs = range(1, data.shape[1])

    steps = data[:, 0]

    each_n_step = int(len(steps)/key_steps)

    if each_n_step < 1:
        each_n_step = 1

    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
    colors = [cc('r'), cc('g'), cc('b'), cc('y')]
    facecolors = []
    line_data = []
    energy_min = np.min(data[:, 1:])

    zs = []
    index = 0
    for i in range(0, len(steps), each_n_step):
        line_data.append(list(zip(xs, data[i, 1:]-energy_min)))
        facecolors.append(colors[index % 4])
        zs.append(data[i, 0])
        index += 1

    poly = PolyCollection(line_data, facecolors=facecolors, closed=False)
    poly.set_alpha(0.7)

    ax.add_collection3d(poly, zs=zs, zdir='x')

    ax.set_xlabel('Steps')
    ax.set_ylabel('images')
    ax.set_zlabel('Energy (J)')

    ax.set_ylim3d(0, len(xs)+1)
    ax.set_xlim3d(0, int(data[-1, 0])+1)
    ax.set_zlim3d(0, np.max(data[:, 1:]-energy_min))

    if filename is None:
        filename = '%s_energy_3d.pdf' % name

    fig.savefig(filename)

if __name__ == '__main__':

    import finmag

    sim = finmag.example.barmini()

    init_images = [(0, 0, -1), (1, 1, 0), (0, 0, 1)]
    interpolations = [15, 14]

    neb = NEB_Sundials(sim, init_images, interpolations)

    neb.relax(stopping_dmdt=1e2)
    plot_energy_3d('unnamed_energy.ndt')

