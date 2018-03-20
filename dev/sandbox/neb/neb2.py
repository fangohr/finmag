import os
import dolfin as df
import numpy as np
import inspect
from aeon import default_timer


from finmag.native import sundials


import logging
log = logging.getLogger(name="finmag")



def normalise(a):
    """
    normalise the n dimensional vector a
    """
    x = a>np.pi
    a[x] = 2*np.pi-a[x]
    x = a<-np.pi
    a[x] += 2*np.pi
    
    length = np.linalg.norm(a)
    
    if length>0:
        length=1.0/length
        
    a[:] *= length
    
def linear_interpolation_two_direct(m0, m1, n):
    m0 = np.array(m0)
    m1 = np.array(m1)
    
    dm = 1.0*(m1 - m0)/(n+1)
        
    coords=[]
    for i in range(n):
        m = m0+(i+1)*dm
        coords.append(m)
    
    return coords


def compute_dm(m0, m1):
    dm = m0-m1
    length = len(dm)
    dm = np.sqrt(np.sum(dm**2))/length
    return dm


def compute_tangents(y, images_energy, m_init, m_final, image_num):
        
        y.shape = (image_num, -1)
        tangents = np.zeros(y.shape)
        
        for i in range(image_num):
            
            if i==0:
                m_a = m_init
            else:
                m_a = y[i-1]
                
            if i == image_num-1:
                m_b = m_final
            else:
                m_b = y[i+1]
            
            energy_a = images_energy[i]
            energy = images_energy[i+1]
            energy_b = images_energy[i+2]
            
            t1 = y[i] - m_a
            t2 = m_b - y[i]
            
                        
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
                
            normalise(tangent)
            
            tangents[i,:]=tangent[:]
            
            
        y.shape=(-1,)
        tangents.shape=(-1,)
        return tangents
    

class NEB_Sundials(object):
    """
    Nudged elastic band method by solving the differential equation using Sundials.
    """
    
    def __init__(self, sim, initial_images, interpolations=None, spring=0, name='unnamed'):
        
        self.sim = sim
        self.name = name
        self.spring = spring
                
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
        
        self.nxyz = len(initial_images[0])
        
        self.all_m = np.zeros(self.nxyz*self.image_num)
        self.Heff = np.zeros(self.all_m.shape)
        self.images_energy = np.zeros(self.image_num+2)
        self.last_m = np.zeros(self.all_m.shape)
        
        
        self.t = 0
        self.step = 1
        self.integrator=None
        self.ode_count=1
        
        self.initial_image_coordinates()
        
       
        
        
    def initial_image_coordinates(self):
        
        image_id = 0
        self.all_m.shape=(self.image_num,-1)
        for i in range(len(self.interpolations)):
            
            n = self.interpolations[i]
            m0 = self.initial_images[i]
            
            if i!=0:
                self.all_m[image_id][:]=m0[:]
                image_id = image_id + 1
                
            m1 = self.initial_images[i+1]
            
            coords = linear_interpolation_two_direct(m0,m1,n)
            
            for coord in coords:
                self.all_m[image_id][:]=coord[:]
                image_id = image_id + 1
        

        self.m_init = self.initial_images[0]
        self.images_energy[0] = self.sim.energy(self.m_init)
        
        self.m_final = self.initial_images[-1]
        self.images_energy[-1] = self.sim.energy(self.m_final)
        
        for i in range(self.image_num):
            self.images_energy[i+1]=self.sim.energy(self.all_m[i])
            
        self.all_m.shape=(-1,)
        print self.all_m

    
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
            self.Heff[i,:] = self.sim.gradient(y[i])[:]
            self.images_energy[i+1] = self.sim.energy(y[i])
        
        y.shape=(-1,)
        self.Heff.shape=(-1,)
    

        
    def sundials_rhs(self, t, y, ydot):
        
        if self.ode_count<3:
            print '%0.20g'%t,y
        
        self.ode_count+=1
        default_timer.start("sundials_rhs", self.__class__.__name__)
        
        self.compute_effective_field(y)
        
        tangents = compute_tangents(y, self.images_energy, self.m_init, self.m_final, self.image_num)
        

        y.shape=(self.image_num, -1)
        ydot.shape=(self.image_num, -1)
        self.Heff.shape = (self.image_num,-1)
        tangents.shape = (self.image_num,-1)
        
        for i in range(self.image_num):
            
            h = self.Heff[i,:]
            t = tangents[i,:]
            
            h3 = h - np.dot(h,t)*t
            ydot[i,:] = h3[:]
            
            """
            The magic thing is that I will get different simulation result 
            if I uncomment the following line even it is isolated.
            (good news is that it doesn't matter in alo)
            """
            #never_used_variable_h3 = h - np.dot(h,t)*t
            
            

        y.shape = (-1,)
        self.Heff.shape=(-1,)
        
        ydot.shape=(-1,)

        default_timer.stop("sundials_rhs", self.__class__.__name__)
        
        return 0
    

    
    
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
        
        m.shape = (-1,)
        y.shape = (-1,)
        
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
            
                
            log.debug("step: {:.3g}, step_size: {:.3g} and max_dmdt: {:.3g}.".format(self.step,increment_dt,dmdt))
            
            if dmdt<stopping_dmdt:
                break
            self.step+=1
        
        log.info("Relaxation finished at time step = {:.4g}, t = {:.2g}, call rhs = {:.4g} and max_dmdt = {:.3g}".format(self.step, self.t, self.ode_count, dmdt))
        
        print self.all_m

    