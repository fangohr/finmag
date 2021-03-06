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

from finmag.native import sundials
from finmag.util.fileio import Tablewriter, Tablereader

import finmag.native.neb as native_neb

import logging
log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s

    
def linear_interpolation_two_direct(m0, m1, n):
    m0 = np.array(m0)
    m1 = np.array(m1)
    
    dm = 1.0*(m1 - m0)/(n+1)
        
    coords=[]
    for i in range(n):
        m = m0+(i+1)*dm
        coords.append(m)
    
    return coords


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
    m0 = np.array(m0)
    m1 = np.array(m1)
    
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

def normalise_m(a):
    """
    normalise the magnetisation length.
    """
    a.shape=(3, -1)
    lengths = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    a[:] /= lengths
    a.shape=(-1, )

class NEB_Sundials(object):
    """
    Nudged elastic band method by solving the differential equation using Sundials.
    """
    
    def __init__(self, sim, initial_images, interpolations=None, spring=0, name='unnamed', normalise=False):
        
        self.sim = sim
        self.name = name
        self.spring = spring
        self.normalise = normalise
                
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
        
        self.nxyz = len(initial_images[0])
        
        self.all_m = np.zeros(self.nxyz*self.image_num)
        self.Heff = np.zeros(self.nxyz*(self.image_num-2))
        self.Heff.shape=(self.image_num-2, -1)
        
        self.tangents = np.zeros(self.Heff.shape)
        self.images_energy = np.zeros(self.image_num)
        self.last_m = np.zeros(self.all_m.shape)
        self.spring_force = np.zeros(self.image_num-2)
        
        
        
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
            m0 = self.initial_images[i]

            self.all_m[image_id][:]=m0[:]
            image_id = image_id + 1
                
            m1 = self.initial_images[i+1]
            
            if self.normalise:
                coords = linear_interpolation_two(m0,m1,n)
            else:
                coords = linear_interpolation_two_direct(m0,m1,n)
            
            for coord in coords:
                self.all_m[image_id][:]=coord[:]
                image_id = image_id + 1
        
        m2 = self.initial_images[-1]
        self.all_m[image_id][:] = m2[:]
        
        for i in range(self.image_num):
            self.images_energy[i]=self.sim.energy(self.all_m[i])
        
        self.all_m.shape=(-1,)
        print self.all_m
        
        self.create_integrator()

    def save_npys(self):
        directory='npys_%s_%d'%(self.name,self.step)
        if not os.path.exists(directory):
            os.makedirs(directory)

        img_id = 1
        self.all_m.shape=(self.image_num,-1)
        for i in range(self.image_num):
            name=os.path.join(directory,'image_%d.npy'%img_id)
            np.save(name,self.all_m[i, :])
            img_id += 1

        self.all_m.shape=(-1,)
        
    
    def create_integrator(self, reltol=1e-6, abstol=1e-6, nsteps=10000):

        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.all_m)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)
        
        self.integrator = integrator
    
    
    def compute_effective_field(self, y):
        
        y.shape=(self.image_num, -1)
        
        for i in range(1,self.image_num-1):
            if self.normalise:
                normalise_m(y[i])
            self.Heff[i-1,:] = self.sim.gradient(y[i])
            self.images_energy[i] = self.sim.energy(y[i])
        
        y.shape=(-1,)
    
    
    def compute_tangents(self, ys):
        ys.shape=(self.image_num, -1)
        native_neb.compute_tangents(ys,self.images_energy, self.tangents)

        for i in range(1, self.image_num-1):
            dm1 = compute_dm(ys[i-1], ys[i])
            dm2 = compute_dm(ys[i+1], ys[i])
            self.spring_force[i-1] = self.spring*(dm2-dm1)

        ys.shape=(-1, )
        
        
    
    def sundials_rhs(self, time, y, ydot):
        
        self.ode_count+=1
        
        default_timer.start("sundials_rhs", self.__class__.__name__)
        
        self.compute_effective_field(y)
        self.compute_tangents(y)

        y.shape=(self.image_num, -1)
        
        ydot.shape=(self.image_num, -1)
        
        for i in range(1, self.image_num-1):
            
            h = self.Heff[i-1]
            t = self.tangents[i-1]
            sf = self.spring_force[i-1]
            
            h3 = h - np.dot(h,t)*t + sf*t
            #h4 = h - np.dot(h,t)*t + sf*t
            
            ydot[i,:] = h3[:]
        
        #it turns out that the following two lines are very important
        ydot[0,:] = 0
        ydot[-1,:] = 0
        
        y.shape = (-1,)
        ydot.shape=(-1,)
        

        default_timer.stop("sundials_rhs", self.__class__.__name__)
        
        return 0
    
    def compute_distance(self):
        distance = []
        
        y = self.all_m
        y.shape=(self.image_num, -1)
        for i in range(self.image_num-1):                
            dm = compute_dm(y[i], y[i+1])
            distance.append(dm)
        
        y.shape =(-1, )
        
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
            dmdt = compute_dm(y[i],m[i])/(t-self.t)
            if dmdt>max_dmdt:
                max_dmdt = dmdt
                        
        m.shape=(-1,)
        y.shape=(-1,)
        
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
            log.debug("step: {:.3g}, step_size: {:.3g} and max_dmdt: {:.3g}.".format(self.step,increment_dt,dmdt))
            
            if dmdt<stopping_dmdt:
                break
            self.step+=1
        
        log.info("Relaxation finished at time step = {:.4g}, t = {:.2g}, call rhs = {:.4g} and max_dmdt = {:.3g}".format(self.step, self.t, self.ode_count, dmdt))
        
        self.save_npys()
        print self.all_m
        
    
