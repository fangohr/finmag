import os
import numpy as np
import dolfin as df
# import inspect
from aeon import default_timer
# import finmag.util.consts as consts


# from finmag.util import helpers
# from finmag.physics.effective_field import EffectiveField
from finmag.util.vtk_saver import VTKSaver
from finmag import Simulation
# from finmag.field import Field  # Change sim._m to new field class
# in line 184
import finmag.native.cvode_petsc as cvode

from finmag.util.fileio import Tablewriter, Tablereader

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import colorConverter
# from matplotlib.collections import PolyCollection, LineCollection

import logging
log = logging.getLogger(name="finmag")

from neb import cartesian2spherical
from neb import spherical2cartesian


def linear_interpolation_two(m0, m1, n):
    """
    Define a linear interpolation between
    two states of the energy band (m0, m1) to get
    an initial state. The interpolation is
    done in the magnetic moments that constitute the
    magnetic system.
    """
    
    
    
    theta_phi0 = cartesian2spherical(m0)
    theta_phi1 = cartesian2spherical(m1)
    
    # The differences with the number of interps + 1
    dtheta = (theta_phi1 - theta_phi0) / (n + 1)

    coords = []
    for i in range(n):
        theta = theta_phi0 + (i + 1) * dtheta
        coords.append(spherical2cartesian(theta))

    return coords


def compute_dm(m0, m1):

    dm = m0 - m1
    length = len(dm)
    dm = np.sqrt(np.sum(dm ** 2)) / length
    return dm


class NEB_Sundials(object):

    """

    Nudged elastic band method by solving the differential equation using
    Sundials.

    """

    def __init__(self, sim,
                 initial_images,
                 climbing_image=None,
                 interpolations=None,
                 spring=5e5, name='unnamed'):
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


              climbing_image : An integer with the index (from 1 to the total
              number of images minus two; it doesn't have any sense to use the
              extreme images) of the image with the largest energy, which will
              be updated in the NEB algorithm using the Climbing Image NEB
              method (no spring force and "with the component along the elastic
              band inverted" [*]). See: [*] Henkelman et al., The Journal of
              Chemical Physics 113, 9901 (2000)

              interpolations : a list only contain integers and the length of
              this list should equal to the length of the initial_images minus
              1, i.e., len(interpolations) = len(initial_images) - 1 ** THIS IS
              not well defined in CARTESIAN coordinates**

              spring: the spring constant, a float value

              disable_tangent: this is an experimental option, by disabling the
              tangent, we can get a rough feeling about the local energy minima
              quickly.

        """

        self.sim = sim
        self.name = name
        self.spring = spring

        # We set a minus one because the *sundials_rhs* function
        # only uses an array without counting the extreme images,
        # whose length is self.image_num (see below)
        if climbing_image is not None:
            self.climbing_image = climbing_image - 1
        else:
            self.climbing_image = climbing_image

        # Dolfin function of the new _m_field (instead of _m)
        self._m = sim.llg._m_field.f
        self.effective_field = sim.llg.effective_field

        if interpolations is None:
            interpolations = [0 for i in range(len(initial_images) - 1)]

        self.initial_images = initial_images
        self.interpolations = interpolations

        if len(interpolations) != len(initial_images) - 1:
            raise RuntimeError("""The length of interpolations should be equal to
                the length of the initial_images array minus 1, i.e.,
                len(interpolations) = len(initial_images) - 1""")

        if len(initial_images) < 2:
            raise RuntimeError("""At least two images must be provided
                               to create the energy band""")

        # the total image number including two ends
        self.total_image_num = len(initial_images) + sum(interpolations)
        self.image_num = self.total_image_num - 2

        
        S3 = sim.S3
        Vs = []
        for i in range(self.image_num):
            Vs.append(S3)
        ME = df.MixedFunctionSpace(Vs)

        self.images_fun = df.Function(ME)
        #all the degree of freedom, which is a petsc vector
        self.coords = df.as_backend_type(self.images_fun.vector()).vec()
        

        self.t = 0
        self.step = 0
        self.ode_count = 1
        self.integrator = None

        self.initial_image_coordinates()
        self.create_tablewriter()

    def linear_interpolation_two(self, image0, image1, n):
        """
        Define a linear interpolation between
        two states of the energy band (m0, m1) to get
        an initial state. The interpolation is
        done in the magnetic moments that constitute the
        magnetic system.
        """

        # Get the spherical coordinates dolfin functions
        # for the m0 and m1 magnetisation vector fields
        self.sim.set_m(self.initial_images[image0])
        theta0, phi0 = self.sim._m_field.get_spherical()

        self.sim.set_m(self.initial_images[image1])
        theta1, phi1 = self.sim._m_field.get_spherical()

        # To not depend on numpy arrays, we will assemble every
        # interpolation into dolfin functions and assign their
        # values to the subdomains of the MixedFunctionSpace of images

        # Create a scalar Function Space
        S1 = df.FunctionSpace(self.sim.m_field.functionspace.mesh(), 'CG', 1)

        # Define a variable to use as vector in all the assemble instances
        assemble_vector = None
 
        # Define the interpolations step for theta
        assemble_vector = df.assemble(df.dot((theta1 - theta0) / (n + 1))
                                      * df.dP,
                                      tensor=assemble_vector)
        dtheta = df.Function(S1)
        dtheta.vector().axpy(1, assemble_vector)
 
        # The same for Phi
        assemble_vector = df.assemble(df.dot((theta1 - theta0) / (n + 1))
                                      * df.dP,
                                      tensor=assemble_vector)
        dphi = df.Function(S1)
        dphi.vector().axpy(1, assemble_vector)

        # Now loop for every interpolation and assign it to
        # the MixedFunctionSpace
        for i in xrange(n):
 
            # Create a dolfin function from the FS, for the interpolation
            interpolation_theta = df.Function(S1)
            interpolation_phi = df.Function(S1)
            # Compute the radius using the assemble method with dolfin dP
            # (like a dirac delta to get values on every node of the mesh)
            # This returns a dolfin vector
 
            # Theta
            assemble_vector = df.assemble(df.dot(theta0 + (i + 1) * dtheta,
                                          #
                                          df.TestFunction(S1)) * df.dP,
                                          tensor=assemble_vector  
                                          )
            # Set the vector values to the dolfin function
            interpolation_theta.vector().axpy(1, assemble_vector)    
 
            # Phi
            assemble_vector = df.assemble(df.dot(phi0 + (i + 1) * dphi,
                                          #
                                          df.TestFunction(S1)) * df.dP,
                                          tensor=assemble_vector  
                                          )
            # Set the vector values to the dolfin function
            interpolation_phi.vector().axpy(1, assemble_vector)

            # Now set this interpolation to the corresponding image

            # Set a vector function space for the simulation
            # magnetisation vector field
            interpolation = df.VectorFunction(self.sim.S3)
            interpolation = df.assemble(df.dot(df.as_vector((df.sin(interpolation_theta) * df.cos(interpolation_phi),
                                                             df.sin(interpolation_theta) * df.sin(interpolation_phi),
                                                             df.cos(interpolation_theta)  
                                                             )),
                                                            df.TestFunction(self.sim.S3)) * df.dP
                                                            )
            interpolation.vector().axpy(1, interpolation)

            # Now assign the interpolation vector function values to the corresponding
            # image in the MixedFunctionSpace
            df.assign(self.images_fun.sub(i), interpolation)

        # coords = []
        # for i in range(n):
        #     theta_phi_interp = theta_phi0 + (i + 1) * d_theta_phi
        #     coords.append(spherical2cartesian(theta_phi_interp))

        # return coords


    def create_tablewriter(self):
        entities_energy = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'energy': {'unit': '<J>',
                       'get': lambda sim: sim.energy,
                       'header': ['image_%d' % i for i in range(self.image_num + 2)]}
        }

        self.tablewriter = Tablewriter(
            '%s_energy.ndt' % (self.name), self, override=True, entities=entities_energy)

        entities_dm = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'dms': {'unit': '<1>',
                    'get': lambda sim: sim.distances,
                    'header': ['image_%d_%d' % (i, i + 1) for i in range(self.image_num + 1)]}
        }
        self.tablewriter_dm = Tablewriter(
            '%s_dms.ndt' % (self.name), self, override=True, entities=entities_dm)

    def initial_image_coordinates(self):
        """

        Generate the coordinates linearly according to the number of
        interpolations provided.

        Example: Imagine we have 4 images and we want 3 interpolations
        between every neighbouring pair, i.e  interpolations = [3, 3, 3]

        1. Imagine the initial states with the interpolation numbers
           and choose the first and second state

            0          1           2          3
            X -------- X --------- X -------- X
                  3          3           3

            2. Counter image_id is set to 0

            3. Set the image 0 magnetisation vector as m0 and append the
               values to self.coords[0]. Update the counter: image_id = 1 now

            4. Set the image 1 magnetisation values as m1 and interpolate
               the values between m0 and m1, generating 3 arrays
               with the magnetisation values of every interpolation image.
               For every array, append the values to self.coords[i]
               with i = 1, 2 and 3 ; updating the counter every time, so
               image_id = 4 now

            5. Append the value of m1 (image 1) in self.coords[4]
               Update counter (image_id = 5 now)

            6. Move to the next pair of images, now set the 1-th image
               magnetisation values as m0 and append to self.coords[5]

            7. Interpolate to get self.coords[i], for i = 6, 7, 8
               ...
            8. Repeat as before until move to the pair of images: 2 - 3

            9. Finally append the magnetisation of the last image
               (self.initial_images[-1]). In this case, the 3rd image

        Then, for every magnetisation vector values array (self.coords[i])
        append the value to the simulation and store the energies
        corresponding to every i-th image to the self.energy[i] arrays

        Finally, flatten the self.coords matrix (containing the magnetisation
        values of every image in different rows)


        ** Our generalised coordinates in the NEBM are the magnetisation values

        """

        # Initiate the counter
        image_id = 0

        # For every interpolation between images (zero if no interpolations
        # were specified)
        for i in range(len(self.interpolations)):
            # Store the number
            n = self.interpolations[i]
            
            # Save on the first image of a pair (step 1, 6, ...)
            self.sim.set_m(self.initial_images[i])
            m0 = self.sim._m_field.get_ordered_numpy_array_xxx()
            df.assign(self.images_fun.sub(image_id),self.sim._m_field.f)
            
            image_id = image_id + 1

            # Set the second image in the pair as m1 and interpolate
            # (step 4 and 7), saving in corresponding self.coords entries
            
            self.sim.set_m(self.initial_images[i + 1])
            m1 = self.sim._m_field.get_ordered_numpy_array_xxx()
            # Interpolations (arrays with magnetisation values)
            coords = linear_interpolation_two(m0, m1, n)

            for coord in coords:
                self.sim.set_m(coord)
                df.assign(self.images_fun.sub(image_id), self.sim._m_field.f)
                self.coords[image_id][:] = coord[:]
                image_id = image_id + 1

            # Continue to the next pair of images

        # Append the magnetisation of the last image
        self.sim.set_m(self.initial_images[-1])
        df.assign(self.images_fun.sub(image_id), self.sim._m_field.f)
        
        """
        # Save the energies
        for i in range(self.total_image_num):
            self._m.vector().set_local(self.coords[i])
            self.effective_field.update()
            self.energy[i] = self.effective_field.total_energy()
        """
        

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
            self._m.vector().set_local(self.coords[i, :])
            # set t =0, it seems that the parameter time is only
            # for the interface?
            vtk_saver.save_field(self._m, 0)

        self.coords.shape = (-1, )


    def create_integrator(self, rtol=1e-6, atol=1e-6):

        integrator = cvode.CvodeSolver(self.sundials_rhs, 0, self.coords, rtol, atol)
        
        self.integrator = integrator

    def compute_effective_field(self, y):

        y.shape = (self.total_image_num, -1)
        """
        for i in range(self.image_num):

            self._m.vector().set_local(y[i + 1])
            #
            self.effective_field.update()
            # Compute effective field, which is the gradient of
            # the energy in the NEB method (derivative with respect to
            # the generalised coordinates)
            h = self.effective_field.H_eff
            #
            self.Heff[i + 1, :] = h[:]
            # Compute the total energy
            self.energy[i + 1] = self.effective_field.total_energy()

            # Compute the 'distance' or difference between neighbouring states
            # around y[i+1]. This is used to compute the spring force
            #
            dm1 = compute_dm(y[i], y[i + 1])
            dm2 = compute_dm(y[i + 1], y[i + 2])
            self.springs[i] = self.spring * (dm2 - dm1)

        # Use the native NEB (C++ code) to compute the tangents according
        # to the improved NEB method, developed by Henkelman and Jonsson
        # at: Henkelman et al., Journal of Chemical Physics 113, 22 (2000)
        native_neb.compute_tangents(y, self.energy, self.tangents)
        # native_neb.compute_springs(y,self.springs,self.spring)
        """
        y.shape = (-1, )
        

    def sundials_rhs(self, time, y, ydot):
        """

        Right hand side of the optimization scheme used to find the minimum
        energy path. In our case, we use a LLG kind of equation:

            d Y / dt = Y x Y x D

            D = -( nabla E + [nabla E * t] t ) + F_spring

        where Y is an image: Y = (M_0, ... , M_N) and t is the tangent vector
        defined according to the energy of the neighbouring images (see
        Henkelman et al publication)

        If a climbing_image index is specified, the corresponding image
        will be iterated without the spring force and with an inversed
        component along the tangent

        """
        # Update the ODE solver
        self.ode_count += 1
        default_timer.start("sundials_rhs", self.__class__.__name__)

        # Compute the eff field H for every image, H = -nabla E
        # (derived with respect to M)
        self.compute_effective_field(y)

        # Reshape y and ydot in a matrix of total_image_num rows
        y.shape = (self.total_image_num, -1)
        ydot.shape = (self.total_image_num, -1)

        # Compute the total force for every image (not the extremes)
        # Rememeber that self.image_num = self.total_image_num - 2
        # The total force is:
        #   D = - (-nabla E + [nabla E * t] t) + F_spring
        # This value is different is a climbing image is specified:
        #   D_climb = -nabla E + 2 * [nabla E * t] t
        for i in range(self.image_num):
            h = self.Heff[i + 1]
            t = self.tangents[i]
            sf = self.springs[i]

            if not (self.climbing_image and i == self.climbing_image):
                h3 = h - np.dot(h, t) * t + sf * t
            else:
                h3 = h - 2 * np.dot(h, t) * t

            h[:] = h3[:]

            #ydot[i+1, :] = h3[:]

        # Update the step with the optimisation algorithm, in this
        # case we use: dY /dt = Y x Y x D
        # (check the C++ code in finmag/native/src/)
        #native_neb.compute_dm_dt(y, self.Heff, ydot)

        ydot[0, :] = 0
        ydot[-1, :] = 0

        y.shape = (-1,)
        ydot.shape = (-1,)

        default_timer.stop("sundials_rhs", self.__class__.__name__)

        return 0

    def compute_distance(self):

        distance = []

        ys = self.coords
        ys.shape = (self.total_image_num, -1)
        for i in range(self.total_image_num - 1):
            dm = compute_dm(ys[i], ys[i + 1])
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
        for i in range(1, self.image_num + 1):
            dmdt = compute_dm(y[i], m[i]) / (t - self.t)
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
        # Save the initial state i=0
        self.compute_distance()
        self.tablewriter.save()
        self.tablewriter_dm.save()

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

            dmdt = self.run_until(self.t + increment_dt)

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

