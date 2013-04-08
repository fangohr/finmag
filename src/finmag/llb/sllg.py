import time
import numpy as np
import dolfin as df
import finmag.util.consts as consts

import finmag.native.llb as native_llb
from finmag.util import helpers
from finmag.energies.effective_field import EffectiveField
from finmag.util.meshes import mesh_volume
from finmag.util.fileio import Tablewriter

from finmag.energies import Zeeman
from finmag.energies import Exchange
from finmag.energies import Demag
from finmag.util.pbc2d import PeriodicBoundary2D

import logging
log = logging.getLogger(name="finmag")

class SLLG(object):
    def __init__(self,mesh,Ms=8.6e5,unit_length=1.0,name='unnamed',auto_save_data=True,method='RK2b',checking_length=False, pbc=None):
        self._t=0
        self.time_scale=1e-9
        self.mesh=mesh
        self.domains =  df.CellFunction("uint", mesh)
        self.domains.set_all(0)
        self.dx = df.Measure("dx")[self.domains]
        self.region_id=0
        
        self.pbc = pbc
        if pbc == '2d':
            self.pbc = PeriodicBoundary2D(mesh)
        
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=self.pbc)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3, constrained_domain=self.pbc)
        self._m = df.Function(self.S3)
        self.nxyz=mesh.num_vertices()
        
        self.unit_length=unit_length
        
        self._T = np.zeros(self.nxyz)
        self._alpha = np.zeros(self.nxyz)
        self.m=np.zeros(3*self.nxyz)
        self.field=np.zeros(3*self.nxyz)
        self.dm_dt=np.zeros(3*self.nxyz)
        self._Ms = np.zeros(3*self.nxyz) #Note: nxyz for Ms length is more suitable? 
        self.effective_field = EffectiveField(self.S3)
        

        self.pin_fun=None
        self.method=method
        self.checking_length=checking_length

        self.set_default_values()
        self.auto_save_data=auto_save_data
        self.name = name
        self.sanitized_name = helpers.clean_filename(name)
        self.logfilename = self.sanitized_name + '.log'
        helpers.start_logging_to_file(self.logfilename, mode='w')
        log.info("Creating Sim object '{}' (rank={}/{}) [{}].".format(
            self.name, df.MPI.process_number(),
            df.MPI.num_processes(), time.asctime()))
        log.info(mesh)
        
        
        
        self.Ms=Ms
        
        if self.auto_save_data:
            self.ndtfilename = self.sanitized_name + '.ndt'
            self.tablewriter = Tablewriter(self.ndtfilename, self, override=True)

    def set_default_values(self):
        #self.Ms = 8.6e5  # A/m saturation magnetisation
        
        self._pins = np.array([], dtype="int")
        self.volumes = df.assemble(df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx).array()
        self.Volume = mesh_volume(self.mesh)
        self.real_volumes=self.volumes*self.unit_length**3
                
        M_pred=np.zeros(self.m.shape)
        
        self.integrator = native_llb.StochasticSLLGIntegrator(
                                    self.m,
                                    M_pred,
                                    self._Ms,
                                    self._T,
                                    self.real_volumes,
                                    self._alpha,
                                    self.stochastic_update_field,
                                    self.method)
        
        self.alpha = 0.5  
        self._gamma = consts.gamma
        self._seed=np.random.random_integers(4294967295)
        self.dt=1e-13
        self.T=0
        
  
    @property
    def t(self):
        return self._t*self.time_scale
    
    @t.setter
    def t(self,value):
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
        log.info("checking_length: "+str(self.checking_length))
                
    def set_m(self,value):
        self._m.vector().set_local(helpers.vector_valued_function(value, self.S3, normalise=True).vector().array())
        self.m[:]=self._m.vector().array()[:]

    def add(self,interaction,with_time_update=None):
        log.debug("Adding interaction %s to simulation '%s'" % (str(interaction),self.name))
        interaction.setup(self.S3, self._m, self._Ms_dg, self.unit_length)
        self.effective_field.add(interaction, with_time_update)
        
        if interaction.__class__.__name__=='Zeeman':
            self.zeeman_interation=interaction
            self.tablewriter.entities['zeeman']={
                        'unit': '<A/m>',
                        'get': lambda sim: sim.zeeman_interation.average_field(),
                        'header': ('h_x', 'h_y', 'h_z')}
        
            self.tablewriter.update_entity_order()


    def run_until(self,t):
        tp=t/self.time_scale
        
        if tp <= self._t:
            return
        try:
            while tp-self._t>1e-12:
                self.integrator.run_step(self.field)
                self._m.vector().set_local(self.m)

                self._t+=self._dt
            
            
        except Exception,error:
            log.info(error)
            raise Exception(error)
            
            
        if abs(tp-self._t)<1e-12:
            self._t=tp
        log.debug("Integrating dynamics up to t = %g" % t)
            
        if self.auto_save_data:
            self.save_data()

    def stochastic_update_field(self,y):
                
        self._m.vector().set_local(y)
        
        self.field[:] = self.effective_field.compute(self.t)[:]
        
        #print self.field[:]

    
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
        
    @property
    def Ms(self):
        return self._Ms
    
    @Ms.setter
    def Ms(self, value):
        self._Ms_dg=helpers.scalar_valued_dg_function(value,self.mesh)

        tmp = df.assemble(self._Ms_dg*df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx)
        tmp=tmp/self.volumes
        self._Ms[:]=tmp[:]
        
    def m_average_fun(self,dx=df.dx):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`

        """ 
        
        mx = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([1, 0, 0])) * dx)
        my = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([0, 1, 0])) * dx)
        mz = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([0, 0, 1])) * dx)
        volume = df.assemble(self._Ms_dg*dx,mesh=self.mesh)
                        
        return np.array([mx, my, mz]) / volume
    m_average=property(m_average_fun)
    
        
    def save_m_in_region(self,region,name='unnamed'):
        
        self.region_id+=1
        helpers.mark_subdomain_by_function(region, self.mesh, self.region_id,self.domains)
        self.dx = df.Measure("dx")[self.domains]
        
        if name=='unnamed':
            name='region_'+str(self.region_id)
        
        region_id=self.region_id
        self.tablewriter.entities[name]={
                        'unit': '<>',
                        'get': lambda sim: sim.m_average_fun(dx=self.dx(region_id)),
                        'header': (name+'_m_x', name+'_m_y', name+'_m_z')}
        
        self.tablewriter.update_entity_order()
    
    def save_data(self):
        log.debug("Saving average field values for simulation '{}'.".format(self.name))
        self.tablewriter.save()
        
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

