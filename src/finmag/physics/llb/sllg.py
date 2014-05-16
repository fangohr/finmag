import time
import numpy as np
import dolfin as df
import finmag.util.consts as consts

import finmag.native.llb as native_llb
from finmag.util import helpers
from finmag.physics.effective_field import EffectiveField
from finmag.util.meshes import mesh_volume, nodal_volume
from finmag.util.fileio import Tablewriter

from finmag.energies import Zeeman
from finmag.energies import Exchange
from finmag.energies import Demag
from finmag.util.pbc2d import PeriodicBoundary2D

import logging
log = logging.getLogger(name="finmag")

class SLLG(object):
    def __init__(self, S1, S3, method='RK2b', checking_length=False, unit_length=1):

        self.S1 = S1
        self.S3 = S3
        self.mesh=S1.mesh()

        self._t=0
        self.time_scale=1e-9

        self._m = df.Function(self.S3)
        self._m.rename("m", "magnetisation")
        self.nxyz = self._m.vector().size()/3

        self._T = np.zeros(self.nxyz)
        self._alpha = np.zeros(self.nxyz)
        self.m=np.zeros(3*self.nxyz)
        self.field=np.zeros(3*self.nxyz)
        self.grad_m=np.zeros(3*self.nxyz)
        self.dm_dt=np.zeros(3*self.nxyz)
        self._Ms = np.zeros(3*self.nxyz) #Note: nxyz for Ms length is more suitable?

        self.pin_fun=None
        self.method = method
        self.checking_length = checking_length
        self.unit_length = unit_length
        self.effective_field = EffectiveField(self.S3, self._m, self.Ms, self.unit_length)
        
        self.zhangli_stt=False

        self.set_default_values()

    def set_default_values(self):
        #self.Ms = 8.6e5  # A/m saturation magnetisation
        self._pins = np.array([], dtype="int")
        self.volumes = df.assemble(df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx).array()
        self.Volume = mesh_volume(self.mesh)
        self.real_volumes = self.volumes*self.unit_length**3

        self.m_pred = np.zeros(self.m.shape)
        
        self.integrator = native_llb.StochasticSLLGIntegrator(
                                    self.m,
                                    self.m_pred,
                                    self._Ms,
                                    self._T,
                                    self.real_volumes,
                                    self._alpha,
                                    self.stochastic_update_field,
                                    self.method)
        
        self.alpha = 0.1
        self._gamma = consts.gamma
        self._seed = np.random.random_integers(4294967295)
        self.dt = 1e-13
        self.T = 0


    @property
    def cur_t(self):
        return self._t*self.time_scale

    @cur_t.setter
    def cur_t(self,value):
        self._t=value/self.time_scale

    @property
    def dt(self):
        return self._dt*self.time_scale

    @property
    def gamma(self):
        return self._gamma

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed=value
        self.setup_parameters()

    @gamma.setter
    def gamma(self, value):
        self._gamma=value
        self.setup_parameters()

    @dt.setter
    def dt(self, value):
        self._dt=value/self.time_scale
        self.setup_parameters()

    def setup_parameters(self):
        #print 'seed:', self.seed
        self.integrator.set_parameters(self.dt,self.gamma,self.seed,self.checking_length)
        log.info("seed=%d."%self.seed)
        log.info("dt=%g."%self.dt)
        log.info("gamma=%g."%self.gamma)
        #log.info("checking_length: "+str(self.checking_length))

    def set_m(self,value,normalise=True):
        m_tmp = helpers.vector_valued_function(value, self.S3, normalise=normalise).vector().array()
        self._m.vector().set_local(m_tmp)
        self.m[:]=self._m.vector().array()[:]


    def advance_time(self,t):
        tp=t/self.time_scale

        if tp <= self._t:
            return
        try:
            while tp-self._t>1e-12:
                if self.zhangli_stt:
                    self.integrator.run_step(self.field, self.grad_m)
                else:
                    self.integrator.run_step(self.field)
                    
                self._m.vector().set_local(self.m)

                self._t+=self._dt
                
        except Exception,error:
            log.info(error)
            raise Exception(error)

        if abs(tp-self._t)<1e-12:
            self._t = tp

    def stochastic_update_field(self,y):

        self._m.vector().set_local(y)

        self.field[:] = self.effective_field.compute(self.cur_t)[:]
        
        if self.zhangli_stt:
            self.grad_m[:] = self.compute_gradient_field()[:]
        
    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T[:]=helpers.scalar_valued_function(value,self.S1).vector().array()[:]
        log.info('Temperature  : %g',self._T[0])

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha[:]=helpers.scalar_valued_function(value,self.S1).vector().array()[:]

    def set_alpha(self, value):
        """ for compability reasons with LLG """
        self.alpha = value

    @property
    def Ms(self):
        return self._Ms

    @Ms.setter
    def Ms(self, value):
        self._Ms_dg=helpers.scalar_valued_dg_function(value,self.mesh)

        tmp = df.assemble(self._Ms_dg*df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx)
        tmp = tmp/self.volumes
        self._Ms[:]=tmp[:]

    def m_average_fun(self,dx=df.dx):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`

        """

        mx = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([1, 0, 0])) * dx)
        my = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([0, 1, 0])) * dx)
        mz = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([0, 0, 1])) * dx)
        volume = df.assemble(self._Ms_dg*dx, mesh=self.mesh)

        return np.array([mx, my, mz]) / volume
    m_average=property(m_average_fun)
    
    
    def compute_gradient_matrix(self):
        """
        compute (J nabla) m , we hope we can use a matrix M such that M*m = (J nabla)m.

        """
        tau = df.TrialFunction(self.S3)
        sigma = df.TestFunction(self.S3)
        
        self.nodal_volume_S3 = nodal_volume(self.S3)*self.unit_length
        
        dim = self.S3.mesh().topology().dim()
        
        ty = tz = 0
        
        tx = self._J[0]*df.dot(df.grad(tau)[:,0],sigma)
        
        if dim >= 2:
            ty = self._J[1]*df.dot(df.grad(tau)[:,1],sigma)
        
        if dim >= 3:
            tz = self._J[2]*df.dot(df.grad(tau)[:,2],sigma)
        
        self.gradM = df.assemble((tx+ty+tz)*df.dx)

        #self.gradM = df.assemble(df.dot(df.dot(self._J, df.nabla_grad(tau)),sigma)*df.dx)
        
    
    def compute_gradient_field(self):

        self.gradM.mult(self._m.vector(), self.H_gradm)
        
        return self.H_gradm.array()/self.nodal_volume_S3
    
    
    def use_zhangli(self, J_profile=(1e10,0,0), P=0.5, beta=0.01, using_u0=False):
        
        self.zhangli_stt = True
        
        self.P = P
        self.beta = beta
        
        self._J = helpers.vector_valued_function(J_profile, self.S3)
        self.J = self._J.vector().array()
        self.compute_gradient_matrix()
        self.H_gradm = df.PETScVector()
        
        self.integrator = native_llb.StochasticLLGIntegratorSTT(
                                    self.m,
                                    self.m_pred,
                                    self._Ms,
                                    self._T,
                                    self.real_volumes,
                                    self._alpha,
                                    self.P,
                                    self.beta,
                                    self.stochastic_update_field,
                                    self.method)
        
        #seems that in the presence of current, the time step have to very small
        self.dt = 1e-14
        
        #TODO: fix the using_u0 here.
        


if __name__ == "__main__":
    mesh = df.Box(0, 0, 0, 5, 5, 5, 1, 1, 1)
    sim = SLLG(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.1
    sim.set_m((1, 0, 0))
    ts = np.linspace(0, 1e-9, 11)
    print sim.Ms
    sim.T=2000
    sim.dt=1e-14

    H0 = 1e6
    sim.add(Zeeman((0, 0, H0)))

    A=helpers.scalar_valued_dg_function(13.0e-12,mesh)
    exchange = Exchange(A)
    sim.add(exchange)

    demag=Demag(solver='FK')
    sim.add(demag)

    print exchange.Ms.vector().array()

    for t in ts:
        sim.run_until(t)
        print sim.m_average

