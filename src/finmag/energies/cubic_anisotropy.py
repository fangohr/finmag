import logging
import numpy as np
import dolfin as df
from aeon import mtimed
from energy_base import EnergyBase
from finmag.util import helpers
from finmag.util.consts import mu0
from finmag.native import llg as native_llg

logger = logging.getLogger('finmag')



class CubicAnisotropy(EnergyBase):
    """
    Compute the cubic anisotropy field.

    *Arguments*
        K1, K2, K3
            The anisotropy constants.
        u1, u2, u3
            The anisotropy axes. Should be unit vectors.

    *Example of Usage*
            Refer to the UniaxialAnisotropy class.

    """

    def __init__(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy', assemble=False):
        """
        Define a cubic anisotropy with (fourth/six/eigth order) anisotropy
        constants `K1`, `K2`, `K3` (in J/m^3) and corresponding axes
        `u1`, `u2` and `u3`.
        
        if assemble = True, the box-assemble will be used, seems that box-assemble 
        method has introduced extra error!!!

        The constants and axes can be passed as df.Constant or df.Function or
        a general function, It is possible to specify spatially
        varying anisotropy by using df.Functions.

        """

        u3 = np.cross(u1,u2)
        
        self.u1 = df.Constant(u1)
        self.u2 = df.Constant(u2)
        self.u3 = df.Constant(u3)
        
        self.K1_input = K1
        self.K2_input = K2
        self.K3_input = K3
        
        self.uv = 1.0*np.array([u1,u2,u3])
        self.uv.shape = (-1,)
        
        self.name = name
        super(CubicAnisotropy, self).__init__("box-assemble", in_jacobian=True)
        
        self.assemble=assemble
        

    @mtimed
    def setup(self, S3, m, Ms, unit_length=1):
        dofmap = S3.dofmap()
        S1 = df.FunctionSpace(S3.mesh(), "Lagrange", 1, constrained_domain=dofmap.constrained_domain)
        
        self.K1_dg = helpers.scalar_valued_dg_function(self.K1_input, S1)
        self.K1_dg.rename('K1', 'fourth order anisotropy constant')
        self.K2_dg = helpers.scalar_valued_dg_function(self.K2_input, S1)
        self.K2_dg.rename('K2', 'sixth order anisotropy constant')
        self.K3_dg = helpers.scalar_valued_dg_function(self.K3_input, S1)
        self.K3_dg.rename('K3', 'eigth order anisotropy constant')
        
        self.volumes = df.assemble(df.TestFunction(S1) * df.dx)
        self.K1 = df.assemble(self.K1_dg*df.TestFunction(S1)* df.dx).array()/self.volumes
        self.K2 = df.assemble(self.K2_dg*df.TestFunction(S1)* df.dx).array()/self.volumes
        self.K3 = df.assemble(self.K3_dg*df.TestFunction(S1)* df.dx).array()/self.volumes
        

        u1msq = df.dot(self.u1, m) ** 2
        u2msq = df.dot(self.u2, m) ** 2
        u3msq = df.dot(self.u3, m) ** 2

        E_term1 = self.K1_dg * (u1msq * u2msq + u2msq * u3msq + u3msq * u1msq)
        E_term2 = self.K2_dg * (u1msq * u2msq * u3msq)
        E_term3 = self.K3_dg * (u1msq ** 2 * u2msq ** 2 + u2msq ** 2 * u3msq ** 2 + u3msq ** 2 * u1msq ** 2)

        E_integrand = E_term1
        
        if self.K2_input!=0:
            E_integrand += E_term2
        
        if self.K3_input!=0:
            E_integrand += E_term3

        super(CubicAnisotropy, self).setup(E_integrand, S3, m, Ms, unit_length)
        
        if not self.assemble:
            self.H = self.m.vector().array()
            self.Ms = self.Ms.vector().array()
            self.compute_field = self.__compute_field_directly
            self.compute_energy = self.__compute_energy
        
    def __compute_field_directly(self):
        
        m = self.m.vector().array()
        
        m.shape=(3,-1)
        self.H.shape=(3,-1)
        native_llg.compute_cubic_field(m,self.Ms, self.H, self.uv, self.K1,self.K2,self.K3)
        m.shape=(-1,)
        self.H.shape=(-1,)
        
        return self.H
    
    def __compute_energy(self):
        m = self.m.vector().array()
        mh = m*self.H
        mh.shape=(3,-1)
        Ei = np.sum(mh, axis=0)*mu0*self.Ms*self.volumes
        E = -0.5*np.sum(Ei) * self.unit_length ** self.dim
        return E
        
        
        
