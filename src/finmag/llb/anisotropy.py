import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from finmag.util.consts import mu0
from finmag.util.helpers import fnormalise
from finmag.energies.energy_base import EnergyBase
logger = logging.getLogger('finmag')


class LLBAnisotropy(EnergyBase):
    """
    Compute the anisotropy field for LLB case
    
    H = -(m_x e_x + m_y e_y)/chi_perp
    
    ==>
    
    E = 0.5 * (m_x^2 + m_y^2)/chi_perp 

   

    """
    
    def __init__(self, inv_chi_perp, method="box-matrix-petsc"):
        self.e_x = df.Constant([1, 0, 0]) 
        self.e_y = df.Constant([0, 1, 0]) 
        self.inv_chi_perp = inv_chi_perp
        super(LLBAnisotropy, self).__init__(method, in_jacobian=True)

    def setup(self, S3, m, Ms0, unit_length=1):
        timings.start('LLBAnisotropy-setup')

        #self._m_normed = df.Function(S3)
        self.m = m

        # Testfunction
        self.v = df.TestFunction(S3)

        # Anisotropy energy
        E = 0.5 * ((df.dot(self.e_x, self.m)) ** 2 + df.dot(self.e_y, self.m) ** 2) * df.dx


        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
       
        nodal_E = 0.5 * df.dot(((df.dot(self.e_x, self.m)) ** 2 + df.dot(self.e_y, self.m) ** 2), w) * df.dx

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(S1)

        EnergyBase.setup(self,
                E=E,
                nodal_E=nodal_E,
                S3=S3,
                M=self.m,
                Ms=Ms0,
                unit_length=unit_length)

        timings.stop('LLBAnisotropy-setup')

    def compute_field(self):
        """
        Compute the field associated with the energy.

         *Returns*
            numpy.ndarray
                The coefficients of the dolfin-function in a numpy array.

        """

        
        Han = super(LLBAnisotropy, self).compute_field()
                
        return Han * self.inv_chi_perp

    

if __name__ == "__main__":
    from dolfin import *
    m = 1e-8
    Ms = 0.8e6
    n = 5
    mesh = Box(0, m, 0, m, 0, m, n, n, n)

    S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
    C = 1.3e-11  # J/m exchange constant
    M = project(Constant((Ms, 0, 0)), S3)  # Initial magnetisation
    #Need to update this example in the main program here:
    uniax = UniaxialAnisotropy(K=1e11, a=[1, 0, 0])

    uniax.setup(S3, M, Ms)

    _ = uniax.compute_field()
    _ = uniax.compute_energy()
    _ = uniax.energy_density()

    print uniax.name
    print timings.report_str()
