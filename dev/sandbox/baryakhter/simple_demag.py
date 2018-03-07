import logging
import numpy as np
import dolfin as df
from finmag.util import helpers

logger = logging.getLogger('finmag')


class SimpleDemag(object):
    """
    Demagnetising field for thin films in the i-direction.
    Hj = Hk = 0 and Hi = - Mi.

    """
    def __init__(self, Ms, Nx=0, Ny=0.5, Nz=0.5, in_jacobian=False, name='SimpleDemag'):
        """
        field_strength is Ms by default

        """
        self.Ms = Ms
        
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        
        self.name = name
        
        self.in_jacobian = in_jacobian
        in_jacobian_msg = "in Jacobian" if in_jacobian else "not in Jacobian"
        logger.debug("Creating {} object, {}.".format(
            self.__class__.__name__, in_jacobian_msg))

    def setup(self, S3, m, Ms, unit_length):
        self.m = m
        self.H = np.zeros((3, S3.mesh().num_vertices()))
        self.Ms_array = np.zeros(3*S3.mesh().num_vertices())
        if isinstance(self.Ms, (int,float)):
            self.Ms_array[:] = self.Ms
        else:
            self.Ms_array[:] = self.Ms[:]
        
        self.Ms_array.shape=(3,-1)


    def compute_field(self):
        m = self.m.vector().array().view().reshape((3, -1))
        self.H[0][:] = -self.Nx*m[0][:]*self.Ms_array[0]
        self.H[1][:] = -self.Ny*m[1][:]*self.Ms_array[1]
        self.H[2][:] = -self.Nz*m[2][:]*self.Ms_array[2]
        return self.H.ravel()

    def compute_energy(self):
        return 0
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())
