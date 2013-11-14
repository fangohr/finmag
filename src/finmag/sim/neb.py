import dolfin as df
import numpy as np
import inspect
from aeon import default_timer
import finmag.util.consts as consts

from finmag.util import helpers
from finmag.energies.effective_field import EffectiveField
from finmag.util.meshes import nodal_volume
from finmag.native import llg as native_llg
from finmag.util.vtk_saver import VTKSaver
#from finmag.native import sundials
from scipy.integrate import ode as scipy_ode

from finmag.util.pbc2d import PeriodicBoundary1D, PeriodicBoundary2D

import logging
log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


class NEB(object):
    """
    Nudged elastic band methods.
        
    The method implemented in this class is similar, but using an extra parameter time t in 
    the path-finding scheme, without spring force and without reparametrization.
    
    If this method doesn't work well, we will implement other methods.
    
    """
    
    def __init__(self, mesh, images_num=10, Ms=8e5, unit_length=1, pbc=None, name='unnamed'):
        """
          *Arguments*
          
              mesh: a dolfin mesh
              
              images_num: images number used in the NEB method (initial and final states are not included)
              
              Ms: Saturation magnetisation  (in A/m)
          
        """
        
        self.mesh = mesh
        self.unit_length = unit_length
        self.images_num = images_num
        
        self.Ms = Ms
        
        self.pbc = pbc
        if pbc == '2d':
            self.pbc = PeriodicBoundary2D(mesh)
        elif pbc == '1d':
            self.pbc = PeriodicBoundary1D(mesh)
            
        self.name = name
        
        self.m_init = None
        self.m_final = None
        self.integrator = None
        self.generate_images = False 
        
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=self.pbc)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3, constrained_domain=self.pbc)
        
        self._m = df.Function(self.S3)
        self.effective_field = EffectiveField(self.S3)
        
        self.nxyz = len(self._m.vector())
        
        self.all_m = np.zeros(self.nxyz*images_num)
        self.Heff = np.zeros(self.all_m.shape)
        self.tangents = np.zeros(self.all_m.shape)
        self.images_energy = np.zeros(images_num)
        
        #print self.all_m
        
        #self.set_default_values()

        self.step = 1
        self.t = 0
        self.ode_count=1
        
    
    def add(self, interaction):
        
        if interaction.name in [i.name for i in self.effective_field.interactions]:
            raise ValueError("Interaction names must be unique, but an "
                             "interaction with the same name already "
                             "exists: {}".format(interaction.name))

        log.debug("Adding interaction %s to simulation '%s'" % (str(interaction),self.name))
        interaction.setup(self.S3, self._m, self.Ms, self.unit_length)
        self.effective_field.add(interaction)
    
        
    def set_m_init(self, value):
        """
        Set the initial magnetisation.

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        """
        self.m_init = helpers.vector_valued_function(value, self.S3, normalise=True).vector().array()
        self._m.vector().set_local(self.m_init)
        self.effective_field.update()
        self.init_energy = self.effective_field.total_energy()
        
        
    def set_m_final(self, value):
        """
        Set the final magnetisation.

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        """
        self.m_final = helpers.vector_valued_function(value, self.S3, normalise=True).vector().array()
        self._m.vector().set_local(self.m_final)
        self.effective_field.update()
        self.final_energy = self.effective_field.total_energy()
        
    def normalise(self, a):
        """
        normalise the array a.
        """
        a.shape=(3, -1)
        lengths = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
        a[:] /= lengths
        a.shape=(-1, )
    
    def cartesian2spherical(self, xyz):
        """
        suppose magnetisation length is normalised
        """
        xyz.shape=(3,-1)
        r_xy = np.sqrt(xyz[0,:]**2 + xyz[1,:]**2)
        theta =  np.arctan2(r_xy, xyz[2,:])
        phi = np.arctan2(xyz[1,:], xyz[0,:])
        xyz.shape=(-1,)
        return theta,phi
    
    
    def spherical2cartesian(self, theta, phi):
        mxyz = np.zeros(self.nxyz)
        mxyz.shape=(3,-1)
        mxyz[0,:] = np.sin(theta)*np.cos(phi)
        mxyz[1,:] = np.sin(theta)*np.sin(phi)
        mxyz[2,:] = np.cos(theta)
        mxyz.shape=(-1,)
        return mxyz
        
    
    def generate_init_images(self):
        if self.m_init is None: 
            raise RuntimeError("Please set the initial magnetisation!")
        if self.m_final is None: 
            raise RuntimeError("Please set the final magnetisation!")
        
        theta0, phi0 = self.cartesian2spherical(self.m_init)
        theta1, phi1 = self.cartesian2spherical(self.m_final)
        dtheta = (theta1-theta0)/(self.images_num+1)
        dphi = (phi1-phi0)/(self.images_num+1)
        
        self.all_m.shape=(self.images_num,-1)
        for i in range(self.images_num):
            theta = theta0+(i+1)*dtheta
            phi = phi0+(i+1)*dphi
            self.all_m[i, :] = self.spherical2cartesian(theta,phi)
        
        self.all_m.shape=(-1,)

    def save_vtks(self):
        vtk_saver = VTKSaver('vtk/%s_%d.pvd'%(self.name, self.step), overwrite=True)
        
        self.all_m.shape=(self.images_num,-1)
        
        self._m.vector().set_local(self.m_init)
        vtk_saver.save_field(self._m, 0)
        
        for i in range(self.images_num):
            self._m.vector().set_local(self.all_m[i, :])
            # set t =0, it seems that the parameter time is only for the interface? 
            vtk_saver.save_field(self._m, 0)
            
        self._m.vector().set_local(self.m_final)
        vtk_saver.save_field(self._m, 0)
        self.all_m.shape=(-1,)
    
    def create_integrator(self, reltol=1e-6, abstol=1e-6, nsteps=10000):

        integrator = scipy_ode(self.rhs_callback)
        # the normalise doesn't have any effect if 'vode' is used
        integrator.set_integrator('dopri5') 
        integrator.set_initial_value(self.all_m, 0)
        
        self.integrator = integrator
    
    
    def compute_effective_field(self, y):
        y.shape=(self.images_num, -1)
        self.Heff.shape = (self.images_num,-1)
        
        for i in range(self.images_num):
            # normalise y first
            self.normalise(y[i])
            self._m.vector().set_local(y[i])
            self.effective_field.update()
            self.Heff[i,:] = self.effective_field.H_eff
            self.images_energy[i] = self.effective_field.total_energy()
        
        y.shape=(-1,)
        #self.Heff.shape=(-1,)
        
    def compute_tangents(self, y):
        y.shape=(self.images_num, -1)
        self.tangents.shape = (self.images_num,-1)
                
        for i in range(self.images_num):
            
            if i==0:
                energy_a = self.init_energy
                m_a = self.m_init
            else:
                energy_a = self.images_energy[i-1]
                m_a = y[i-1]
                
            if i==self.images_num-1:
                energy_b = self.final_energy
                m_b = self.m_final
            else:
                energy_b = self.images_energy[i+1]
                m_b = y[i+1]
            
            energy=self.images_energy[i]
            if energy_a<energy and energy<energy_b:
                tangent = m_b-y[i]
            elif energy_a>energy and energy>energy_b:
                tangent = m_a-y[i]
            else:
                tangent = m_b - m_a
        
            self.tangents[i,:]=helpers.fnormalise(tangent)
        
        y.shape=(-1,)
        
    
    def rhs_callback(self, t, y):
        
        self.ode_count+=1

        default_timer.start("sundials_rhs", self.__class__.__name__)
        self.compute_effective_field(y)
        self.compute_tangents(y)

        self.Heff.shape = (self.images_num,-1)

        for i in range(self.images_num):
            h = self.Heff[i]
            t = self.tangents[i]
            h.shape=(3,-1)
            t.shape=(3,-1)
            ht = h[0]*t[0] + h[1]*t[1] + h[2]*t[2]
            h[0] = h[0] - ht*t[0]
            h[1] = h[1] - ht*t[0]
            h[2] = h[2] - ht*t[0]

            h.shape=(-1,)
            t.shape=(-1,)
            
        self.Heff.shape=(-1,)

        default_timer.stop("sundials_rhs", self.__class__.__name__)
        
        return self.Heff
    
    def run_until(self, t):
        r = self.integrator
        
        while r.successful() and r.t < t:
            r.integrate(t)
            
        m = self.all_m
        y = r.y
        
        m.shape=(self.images_num,-1)
        y.shape=(self.images_num,-1)
        max_dmdt=0
        for i in range(self.images_num):
            dmdt=helpers.compute_dmdt(self.t,m[i],t,y[i])
            if dmdt>max_dmdt:
                max_dmdt = dmdt
        
        m.shape=(-1,)
        y.shape=(-1,)
        
        self.all_m[:] = r.y[:]
        self.t=t
        
        log.info("max_dmdt at t={} s is {:.3g}.".format(self.t,max_dmdt))
        
        return max_dmdt
        
    
    def relax(self, dt=1e-7, stopping_dmdt=1e4, max_steps=1000):
        if not self.generate_images:
            self.generate_init_images()
            self.generate_images=True
            
        if self.integrator is None:
            self.create_integrator()
            
        log.debug("Relaxation parameters: stopping_dmdt={} (degrees per nanosecond), "
                  "time_step={} s, max_steps={}.".format(stopping_dmdt, dt, max_steps))
         
        for i in range(max_steps):
            dmdt=self.run_until((i+1)*dt)
            if dmdt<stopping_dmdt:
                break
            self.step+=1
        
        log.info("Relaxation finished at time t = {:.2g}, steps={}.".format(self.t, self.step))
        
        self.save_vtks()
        
        



if __name__ == '__main__':
    from finmag.energies import Exchange, DMI, Demag, Zeeman, UniaxialAnisotropy
    mesh = df.RectangleMesh(0,0,10,10,1,1)
    
    
    neb = NEB(mesh, Ms=8.6e5, images_num=15, unit_length=1e-9)
    
    neb.set_m_init((-1,0,0))
    neb.set_m_final((1,0,0))
    
    neb.add(UniaxialAnisotropy(-1e5, (0, 1, 1), name='Kp'))
    neb.add(UniaxialAnisotropy(1e4, (1, 0, 0), name='Kx'))
    #neb.add(UniaxialAnisotropy(1e4, (0, 1, 0), name='Ky'))
    
    #neb.save_vtks()

    neb.relax()
    
    #df.plot(mesh)
    #df.interactive()
    
