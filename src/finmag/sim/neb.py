import os
import dolfin as df
import numpy as np
import inspect
from aeon import default_timer
import finmag.util.consts as consts

from finmag.util import helpers
from finmag.energies.effective_field import EffectiveField
from finmag.util.vtk_saver import VTKSaver
from finmag import Simulation

from finmag.native import sundials
from finmag.util.fileio import Tablewriter, Tablereader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import colorConverter
from matplotlib.collections import PolyCollection, LineCollection

import logging
log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


def normalise(a):
    """
    normalise the n dimensional vector a
    """
    length = 1.0/np.linalg.norm(a)
    a[:] *= length

def normalise_m(a):
    """
    normalise the magnetisation length.
    """
    a.shape=(3, -1)
    lengths = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    a[:] /= lengths
    a.shape=(-1, )

def cartesian2spherical(xyz):
    xyz.shape=(3,-1)
    r_xy = np.sqrt(xyz[0,:]**2 + xyz[1,:]**2)
    theta =  np.arctan2(r_xy, xyz[2,:])
    phi = np.arctan2(xyz[1,:], xyz[0,:])
    xyz.shape=(-1,)
    return theta,phi

def spherical2cartesian(theta, phi):
    mxyz = np.zeros(3*len(theta))
    mxyz.shape=(3,-1)
    mxyz[0,:] = np.sin(theta)*np.cos(phi)
    mxyz[1,:] = np.sin(theta)*np.sin(phi)
    mxyz[2,:] = np.cos(theta)
    mxyz.shape=(-1,)
    return mxyz

def linear_interpolation_two(m0, m1, n):
    theta0, phi0 = cartesian2spherical(m0)
    theta1, phi1 = cartesian2spherical(m1)
    dtheta = (theta1-theta0)/(n+1)
    dphi = (phi1-phi0)/(n+1)
        
    coords=[]
    for i in range(n):
        theta = theta0+(i+1)*dtheta
        phi = phi0+(i+1)*dphi
        coords.append(spherical2cartesian(theta,phi))
    
    return coords


def compute_dm(m0, m1):
    dm = m0-m1
    length = len(dm)
    dm = np.sqrt(np.sum(dm**2))/length
    return dm


class NEB_Image(object):
    """
    An Image is an object that has its coordinate (magnetisation), and 
    the object moves in the presence of the force (effective field).
    
    For Steepest descents (SD), given a short time dt, the new coordinate 
    coord_new is computed by,
        
        coord_new = coord_old + F * dt
    
    where F is the force.
    
    
    For Quick-min, 
    
        coord_new = coord_old + V_old * dt
        V_new = V_old + F * dt
    
    where V is the velocity in the direction of the force (u_F),
    
        V = (V * u_F) u_F

    and V = 0 if V * u_F < 0.
    
    """
    def __init__(self, sim, spring=5e5, left_image=None, right_image=None, disable_tangent=False):
        self._m = sim.llg._m
        self.effective_field = sim.llg.effective_field
        
        self.coordinate = np.zeros(self._m.vector().size())
        self.force = np.zeros(self._m.vector().size())
        #self.velocity = self.force.copy()
        #self.force_unit = self.force.copy()
        self.tangent = self.force.copy()
        self.H_eff = self.force.copy()
        
        self.energy = 0
        self.left_image = left_image
        self.right_image = right_image
        self.spring = spring
        self.disable_tangent=disable_tangent
    
    def update_effective_field(self):
        self._m.vector().set_local(self.coordinate)
        self.effective_field.update()
        
        self.H_eff[:] = self.effective_field.H_eff[:]
        self.energy = self.effective_field.total_energy()
    
    def __compute_tangent(self):
        if self.left_image is None or self.right_image is None:
            return 
        
        if self.disable_tangent:
            self.tangent[:] = 0
            return
        img_a = self.left_image
        img_b = self.right_image
        energy = self.energy
        
        t1 = self.coordinate - img_a.coordinate
        t2 = img_b.coordinate - self.coordinate
        
        if img_a.energy < energy and energy < img_b.energy:
            self.tangent[:] = t2
        elif img_a.energy > energy and energy > img_b.energy:
            self.tangent[:] = t1
        else:
            e1 = img_a.energy - self.energy
            e2 = img_b.energy - self.energy
            max_e = max(abs(e1), abs(e2))
            min_e = min(abs(e1), abs(e2)) 
            
            normalise(t1)
            normalise(t2)
            
            if img_b.energy > img_a.energy:
                self.tangent[:] = t1*min_e + t2*max_e
            else:
                self.tangent[:] = t1*max_e + t2*min_e
        
        normalise(self.tangent)
            
        
    def compute_force(self):
        if self.left_image is None or self.right_image is None:
            return 
        
        self.__compute_tangent()
        
        h = self.H_eff
        t = self.tangent
                
        self.force = self.H_eff-np.dot(h,t)*t
        
        if self.spring!=0:
            m_a = self.left_image.coordinate
            m_b = self.coordinate
            m_c = self.right_image.coordinate
                        
            R = self.spring*(m_c+m_a-2*m_b)
            
            self.force += np.dot(R,t)*t
            
    def move(self, dt):
        if self.left_image is None or self.right_image is None:
            return 0
        
        # Actually, what we used here something like Euler method
        # also called Steepest descents (SD), converge slowly in stiff systems
        self.coordinate += self.force*dt

        normalise_m(self.coordinate)
        
        self.update_effective_field()
        
        dm = (self.force*dt).reshape((3, -1))
        max_dm = np.max(np.sqrt(np.sum(dm**2, axis=0))) # max of L2-norm
        max_dmdt = max_dm/dt
        return max_dmdt


    
class NEB(object):
    """
    Nudged elastic band methods.
    """
    
    def __init__(self, sim, initial_images, interpolations=None, spring=5e5, disable_tangent=False, name='unnamed'):
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
        
        self._m = sim.llg._m
                
        if interpolations is None:
            interpolations=[0 for i in range(len(initial_images)-1)]
        
        self.initial_images = initial_images
        self.interpolations = interpolations
        
        if len(interpolations)!=len(initial_images)-1:
            raise RuntimeError("""the length of interpolations should equal to 
                the length of the initial_images minus 1, i.e., 
                len(interpolations) = len(initial_images) -1""")
            
        if len(initial_images)<2:
            raise RuntimeError("""At least two images are needed to be provided.""")
        
        self.image_num = len(initial_images) + sum(interpolations)
        self.image_list = []
        
        for i in range(self.image_num):
            self.image_list.append(NEB_Image(sim, spring=spring, disable_tangent=disable_tangent))
        
        
        for i,image in enumerate(self.image_list):
            if i>0 and i<self.image_num-1:
                image.left_image=self.image_list[i-1]
                image.right_image=self.image_list[i+1]
            
        self.initial_image_coordinates()
        
        self.step = 0
        
        self.tablewriter = Tablewriter('%s_energy.ndt'%name, self, override=True)
        self.tablewriter.entities = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'energy': {'unit': '<J>',
                       'get': lambda sim: sim.energy,
                       'header': ['image_%d'%i for i in range(self.image_num)]}
            }
        keys = self.tablewriter.entities.keys()
        keys.remove('step')
        self.tablewriter.entity_order = ['step'] + sorted(keys)
        
        
        self.tablewriter_dm = Tablewriter('%s_dms.ndt'%name, self, override=True)
        self.tablewriter_dm.entities = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'dms': {'unit': '<1>',
                       'get': lambda sim: sim.distances,
                       'header': ['image_%d_%d'%(i, i+1) for i in range(self.image_num-1)]}
            }
        keys = self.tablewriter_dm.entities.keys()
        keys.remove('step')
        self.tablewriter_dm.entity_order = ['step'] + sorted(keys)
        
        
    def initial_image_coordinates(self):
        
        image_id = 0
        
        for i in range(len(self.interpolations)):
            
            n = self.interpolations[i]
            
            self.sim.set_m(self.initial_images[i])
            m0 = self.sim.m
            
            self.image_list[image_id].coordinate[:]=m0[:]
            image_id = image_id + 1
            
            self.sim.set_m(self.initial_images[i+1])
            m1 = self.sim.m
            
            coords = linear_interpolation_two(m0,m1,n)
            
            for coord in coords:
                self.image_list[image_id].coordinate[:]=coord[:]
                image_id = image_id + 1
                
            if image_id == len(self.image_list)-1:
                self.image_list[image_id].coordinate[:]=m1[:]
        
        energy=[]
        for image in self.image_list:
            image.update_effective_field()
            energy.append(image.energy)
            
        self.energy = np.array(energy)      
            
    def compute_distance(self):
        distance = []
        for i,image in enumerate(self.image_list):
            if i<self.image_num-1:
                m_a = image.coordinate
                m_b = self.image_list[i+1].coordinate
                dm = compute_dm(m_a, m_b)
                distance.append(dm)
        
        self.distances=np.array(distance)

    def save_vtks(self):
        
        vtk_saver = VTKSaver('vtks_%s/step_%d.pvd'%(self.name, self.step), overwrite=True)
        
        for image in self.image_list:
            
            self._m.vector().set_local(image.coordinate)
            
            vtk_saver.save_field(self._m, 0)
    
    def save_npys(self):
        directory='npys_%s_%d'%(self.name,self.step)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i,image in enumerate(self.image_list):
            name=os.path.join(directory,'image_%d.npy'%i)
            np.save(name,image.coordinate)
    
    def step_relax(self, dt):
        
        dm_dts=[]
        energy = []
                
        for image in self.image_list:
            # we already updated the effective field after moving
            #image.update_effective_field() 
            image.compute_force()
        
        for image in self.image_list:
            dm_dt=image.move(dt)
            dm_dts.append(dm_dt)
            energy.append(image.energy)
            
        self.energy = np.array(energy)
        
        self.step += 1
        return np.max(dm_dts)
        
    
    def relax(self, dt=1e-8, save_ndt_steps=10, save_vtk_steps=100, max_steps=500):

        log.debug("Relaxation parameters "
                  "time_step={} s, max_steps={}.".format(dt, max_steps))
        
                
        for i in range(max_steps):
                
            max_dmdt=self.step_relax(dt)
            
            if i%save_ndt_steps==0:
                self.compute_distance()
                self.tablewriter.save()
                self.tablewriter_dm.save()
                
            if i%save_vtk_steps==0:
                self.save_vtks()
            log.info("step: {:.3g}, max_dmdt: {:.3g}.".format(self.step,max_dmdt))
        
        #log.info("Relaxation finished at time t = {:.2g}, ".format(self.t, self.step, self.ode_count))
        self.save_npys()
        self.save_vtks()


class NEB_Sundials(object):
    """
    Nudged elastic band method by solving the differential equation using Sundials.
    """
    
    def __init__(self, sim, initial_images, interpolations=None, spring=5e5, name='unnamed'):
        
        self.sim = sim
        self.name = name
        self.spring = spring
        
        self._m = sim.llg._m
        self.effective_field = sim.llg.effective_field
                
        if interpolations is None:
            interpolations=[0 for i in range(len(initial_images)-1)]
        
        self.initial_images = initial_images
        self.interpolations = interpolations
        
        if len(interpolations)!=len(initial_images)-1:
            raise RuntimeError("""the length of interpolations should equal to 
                the length of the initial_images minus 1, i.e., 
                len(interpolations) = len(initial_images) -1""")
            
        if len(initial_images)<2:
            raise RuntimeError("""At least two images are needed to be provided.""")
        
        self.image_num = len(initial_images) + sum(interpolations) - 2
        
        self.nxyz = len(self._m.vector())
        
        self.all_m = np.zeros(self.nxyz*self.image_num)
        self.Heff = np.zeros(self.all_m.shape)
        self.tangents = np.zeros(self.all_m.shape)
        self.images_energy = np.zeros(self.image_num+2)
        self.last_m = np.zeros(self.all_m.shape)
        self.spring_force = np.zeros(self.all_m.shape)
        
        self.t = 0
        self.step = 1
        self.integrator=None
        self.ode_count=1
        
        self.initial_image_coordinates()
        
        self.tablewriter = Tablewriter('%s_energy.ndt'%name, self, override=True)
        self.tablewriter.entities = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'energy': {'unit': '<J>',
                       'get': lambda sim: sim.images_energy,
                       'header': ['image_%d'%i for i in range(self.image_num+2)]}
            }
        keys = self.tablewriter.entities.keys()
        keys.remove('step')
        self.tablewriter.entity_order = ['step'] + sorted(keys)
        
        
        self.tablewriter_dm = Tablewriter('%s_dms.ndt'%name, self, override=True)
        self.tablewriter_dm.entities = {
            'step': {'unit': '<1>',
                     'get': lambda sim: sim.step,
                     'header': 'steps'},
            'dms': {'unit': '<1>',
                       'get': lambda sim: sim.distances,
                       'header': ['image_%d_%d'%(i, i+1) for i in range(self.image_num+1)]}
            }
        keys = self.tablewriter_dm.entities.keys()
        keys.remove('step')
        self.tablewriter_dm.entity_order = ['step'] + sorted(keys)
        
        
    def initial_image_coordinates(self):
        
        image_id = 0
        self.all_m.shape=(self.image_num,-1)
        for i in range(len(self.interpolations)):
            
            n = self.interpolations[i]
            
            self.sim.set_m(self.initial_images[i])
            m0 = self.sim.m
            
            if image_id!=0:
                self.all_m[image_id][:]=m0[:]
                image_id = image_id + 1
            
            self.sim.set_m(self.initial_images[i+1])
            m1 = self.sim.m
            
            coords = linear_interpolation_two(m0,m1,n)
            
            for coord in coords:
                self.all_m[image_id][:]=coord[:]
                image_id = image_id + 1
                
        
        self.sim.set_m(self.initial_images[0])
        self.m_init = self.sim.m.copy()
        self.effective_field.update()
        self.images_energy[0]=self.effective_field.total_energy()
        
        self.sim.set_m(self.initial_images[-1])
        self.m_final = self.sim.m.copy()
        self.effective_field.update()
        self.images_energy[-1]=self.effective_field.total_energy()
        
        for i in range(self.image_num):
            self._m.vector().set_local(self.all_m[i])
            self.effective_field.update()
            self.images_energy[i+1]=self.effective_field.total_energy()
            
            
        self.all_m.shape=(-1,)

    def save_vtks(self):
        
        vtk_saver = VTKSaver('vtks/%s_%d.pvd'%(self.name, self.step), overwrite=True)
        
        self.all_m.shape=(self.image_num,-1)
        
        self._m.vector().set_local(self.m_init)
        vtk_saver.save_field(self._m, 0)
        
        for i in range(self.image_num):
            self._m.vector().set_local(self.all_m[i, :])
            # set t =0, it seems that the parameter time is only for the interface? 
            vtk_saver.save_field(self._m, 0)
            
        self._m.vector().set_local(self.m_final)
        vtk_saver.save_field(self._m, 0)
        self.all_m.shape=(-1,)
        
    def save_npys(self):
        directory='npys_%s_%d'%(self.name,self.step)
        if not os.path.exists(directory):
            os.makedirs(directory)

        name=os.path.join(directory,'image_0.npy')
        np.save(name,self.m_init)
        
        img_id = 1
        self.all_m.shape=(self.image_num,-1)
        for i in range(self.image_num):
            name=os.path.join(directory,'image_%d.npy'%img_id)
            np.save(name,self.all_m[i, :])
            img_id += 1

        name=os.path.join(directory,'image_%d.npy'%img_id)
        np.save(name,self.m_final)
        
    
    def create_integrator(self, reltol=1e-6, abstol=1e-6, nsteps=10000):

        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.all_m)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)
        
        self.integrator = integrator
    
    
    def compute_effective_field(self, y):
        
        y.shape=(self.image_num, -1)
        self.Heff.shape = (self.image_num,-1)
        
        for i in range(self.image_num):
            # normalise y first
            normalise_m(y[i])
            self._m.vector().set_local(y[i])
            self.effective_field.update()
            self.Heff[i,:] = self.effective_field.H_eff[:]
            self.images_energy[i+1] = self.effective_field.total_energy()
        
        y.shape=(-1,)
        self.Heff.shape=(-1,)
    
    
    def compute_tangents(self, y):
        
        y.shape=(self.image_num, -1)
        self.tangents.shape = (self.image_num,-1)
        self.spring_force.shape = (self.image_num,-1)
        
        for i in range(self.image_num):
            
            if i==0:
                m_a = self.m_init
            else:
                m_a = y[i-1]
                
            if i==self.image_num-1:
                m_b = self.m_final
            else:
                m_b = y[i+1]
            
            energy_a = self.images_energy[i]
            energy = self.images_energy[i+1]
            energy_b = self.images_energy[i+2]
            
            t1 = y[i] - m_a
            t2 = m_b - y[i]
            normalise(t1)
            normalise(t2)
            
            # actually this method quite stable
            tangent = t1 + t2
            
            """
            tangent = m_b - m_a
            
            if energy_a<energy and energy<energy_b:
                tangent = t2
            elif energy_a>energy and energy>energy_b:
                tangent = t1
            else:
                
                e1 = energy_a - energy
                e2 = energy_b - energy
                
                if abs(e1)>abs(e2):
                    max_e=abs(e1)
                    min_e=abs(e2)
                else:
                    max_e=abs(e2)
                    min_e=abs(e1)
            
                normalise(t1)
                normalise(t2)
            
                if energy_b > energy_a:
                    tangent = t1*min_e + t2*max_e
                else:
                    tangent = t1*max_e + t2*min_e
            """
            normalise(tangent)
            
            self.tangents[i,:]=tangent[:]
            
            # this 'old' method is much better than the improved method,
            # i.e., eq (5) is better than eq.(12) in J. Chem. Phys.,Vol. 113, 9978 
            self.spring_force[i][:]=self.spring*(m_a+m_b-2*y[i])
            
        y.shape=(-1,)
        
    
    def sundials_rhs(self, t, y, ydot):
        
        self.ode_count+=1
        default_timer.start("sundials_rhs", self.__class__.__name__)
        
        self.compute_effective_field(y)
        self.compute_tangents(y)

        y.shape=(self.image_num, -1)
        self.Heff.shape = (self.image_num,-1)
        
        for i in range(self.image_num):
            m = y[i]
            h = self.Heff[i]
            t = self.tangents[i]
            sf = self.spring_force[i]
            
            # BUG: h = h - np.dot(h,t)*t is not safe???
            h2 = h - np.dot(h-sf,t)*t
            
            self.Heff[i][:] = h2

        y.shape = (-1,)
        self.Heff.shape=(-1,)
        
        ydot[:]=self.Heff[:]

        default_timer.stop("sundials_rhs", self.__class__.__name__)
        
        return 0
    
    def compute_distance(self):
        distance = []
        
        y = self.all_m
        y.shape=(self.image_num, -1)
        for i in range(self.image_num):
            if i==0:
                m_a = self.m_init
            else:
                m_a = y[i-1]
                
            dm = compute_dm(y[i], m_a)
            distance.append(dm)
        
        dm = compute_dm(y[-1], self.m_final)
        distance.append(dm)
        
        self.distances=np.array(distance)
    
    
    def run_until(self, t):
        
        if t <= self.t:
            return

        self.integrator.advance_time(t, self.all_m)
        
        m = self.all_m
        y = self.last_m
        
        m.shape=(self.image_num,-1)
        y.shape=(self.image_num,-1)
        max_dmdt=0
        for i in range(self.image_num):
            dmdt=helpers.compute_dmdt(self.t,y[i],t,m[i])
            if dmdt>max_dmdt:
                max_dmdt = dmdt
        
        self.last_m[:] = m[:]
        self.t=t
        
        return max_dmdt
    
    def relax(self, dt=1e-8, stopping_dmdt=1e4, max_steps=1000, save_ndt_steps=1, save_vtk_steps=100):
        
        if self.integrator is None:
            self.create_integrator()
        
        log.debug("Relaxation parameters: stopping_dmdt={} (degrees per nanosecond), "
                  "time_step={} s, max_steps={}.".format(stopping_dmdt, dt, max_steps))
         
        for i in range(max_steps):
            
            cvode_dt = self.integrator.get_current_step()
            
            increment_dt = dt
            
            if cvode_dt > dt:
                increment_dt = cvode_dt

            dmdt = self.run_until(self.t+increment_dt)
            
            if i%save_ndt_steps==0:
                self.compute_distance()
                self.tablewriter.save()
                self.tablewriter_dm.save()
                
            if i%save_vtk_steps==0:
                self.save_vtks()
            log.debug("step: {:.3g}, step_size: {:.3g} and max_dmdt: {:.3g}.".format(self.step,increment_dt,dmdt))
            
            if dmdt<stopping_dmdt:
                break
            self.step+=1
        
        log.info("Relaxation finished at time step = {:.4g}, t = {:.2g}, call rhs = {:.4g} and max_dmdt = {:.3g}".format(self.step, self.t, self.ode_count, dmdt))
        
        self.save_vtks()
        self.save_npys()


def plot_energy_3d(ndt_name, key_steps=50, filename=None):
    
    data = np.loadtxt(ndt_name)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    #image index
    xs = range(1, len(data[0,:]))
    
    steps = data[:,0]
    
    if key_steps>len(steps)-1:
        key_steps = len(steps)-1
    
    each_n_step=int(len(steps)/key_steps)
    
    if each_n_step<1:
        each_n_step = 1
    
    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
    colors = [cc('r'), cc('g'), cc('b'),cc('y')]
    facecolors = []
    line_data = []
    energy_min=np.min(data[:,1:])
    
    for i in range(0,len(steps),each_n_step):
        line_data.append(list(zip(xs, data[i,1:]-energy_min)))
        facecolors.append(colors[i%4])
        
    poly = PolyCollection(line_data, facecolors=facecolors)
    poly.set_alpha(0.7)
    
    ax.add_collection3d(poly,zs=data[:,0], zdir='x')
    
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('images')
    ax.set_zlabel('Energy (J)')
    
    ax.set_ylim3d(0, len(xs)+1)
    ax.set_xlim3d(0, key_steps+1)
    ax.set_zlim3d(0, np.max(data[:,1:]-energy_min))
    
    if filename is None:
        filename = '%s_energy_3d.pdf'%ndt_name[:-4]

    fig.savefig(filename)



if __name__ == '__main__':
    
    import finmag
    
    sim = finmag.example.barmini()
    
    init_images=[(0,0,-1),(1,1,0),(0,0,1)]
    interpolations = [5,4]
    
    neb = NEB(sim, init_images, interpolations)
    
    neb.relax()
    