import numpy as np
import dolfin as df
import finmag.util.consts as consts

from finmag.native.sllgc import RK2S
from finmag.util import helpers
from finmag.energies.effective_field import EffectiveField
from finmag.util.meshes import mesh_volume

class SLLG(object):
    def __init__(self,mesh,Ms=8.6e5,unit_length=1.0):
        self._t=0
        self.time_scale=1e-9
        self.mesh=mesh
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1,dim=3)
        self._m = df.Function(self.S3)
        self.nxyz=mesh.num_vertices()
        
        self.unit_length=unit_length
        self.Ms=Ms
        
        self._T = np.zeros(self.nxyz)
        self._alpha = np.zeros(self.nxyz)
        self.m=np.zeros(3*self.nxyz)
        self.field=np.zeros(3*self.nxyz)
        self.dm_dt=np.zeros(3*self.nxyz)
        self.effective_field = EffectiveField(mesh)
        

        self.pin_fun=None

        self.set_default_values()

    def set_default_values(self):
        #self.Ms = 8.6e5  # A/m saturation magnetisation
        
        self._pins = np.array([], dtype="int")
        self.volumes = df.assemble(df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx).array()
        self.Volume = mesh_volume(self.mesh)
        self.volumes*=self.unit_length**3
        
        self.integrator=RK2S(self.nxyz,
                                self.m,
                                self._alpha,
                                self._T,
                                self.volumes,
                                self.stochastic_update_field)
        
        self.alpha = 0.5  
        self.gamma = consts.gamma
        self.c = 1e11  # 1/s numerical scaling correction 
        self.dt=1e-14
        self.T=0
  
    @property
    def t(self):
        return self._t*self.time_scale       
    @property
    def dt(self):
        return self._dt*self.time_scale 
    
    @dt.setter
    def dt(self, value):
        self._dt=value/self.time_scale 
        self.integrator.setup_parameters(self.gamma,self.Ms,self.dt,self.c)
                
    def set_m(self,value):
        self._m = helpers.vector_valued_function(value, self.S3, normalise=False)
        self.m[:]=self._m.vector().array()[:]

    def add(self,interaction,with_time_update=None):
        interaction.setup(self.S3, self._m, self.Ms, self.unit_length)
        self.effective_field.add(interaction, with_time_update)


    def run_until(self,t):
        tp=t/self.time_scale
        
        if tp <= self._t:
            return
        
        while tp-self._t>1e-12:
            self.integrator.run_step(self.field)
            self._t+=self._dt
        
        if abs(tp-self._t)<1e-12:
            self._t=tp

    def stochastic_update_field(self,y):
                
        self._m.vector().set_local(y)
        
        self.field[:] = self.effective_field.compute(self.t)[:]

    
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, value):
        self._T[:]=helpers.scale_valued_function(value,self.S1).vector().array()[:]
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha[:]=helpers.scale_valued_function(value,self.S1).vector().array()[:]
            
    
    @property
    def m_average(self):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`

        """
        #Compute volume if not done before
            
        mx = df.assemble(df.dot(self._m, df.Constant([1, 0, 0])) * df.dx)
        my = df.assemble(df.dot(self._m, df.Constant([0, 1, 0])) * df.dx)
        mz = df.assemble(df.dot(self._m, df.Constant([0, 0, 1])) * df.dx)
        return np.array([mx, my, mz]) / self.Volume

