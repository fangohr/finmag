import logging
import numpy as np
import dolfin as df

logger = logging.getLogger('finmag')


class SimpleDemag(object):
    """
    Demagnetising field for thin films in the i-direction.
    Hj = Hk = 0 and Hi = - Mi.

    """
    def __init__(self, Ms, Nx=0, Ny=0.5, Nz=0.5, in_jacobian=False):
        """
        field_strength is Ms by default

        """
        self.Ms = Ms
        
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        
        self.in_jacobian = in_jacobian
        in_jacobian_msg = "in Jacobian" if in_jacobian else "not in Jacobian"
        logger.debug("Creating {} object, {}.".format(
            self.__class__.__name__, in_jacobian_msg))

    def setup(self, S3, m, Ms, unit_length):
        self.m = m
        self.H = np.zeros((3, S3.mesh().num_vertices()))


    def compute_field(self):
        m = self.m.vector().array().view().reshape((3, -1))
        self.H[0][:] = -self.Nx*m[0][:]*self.Ms
        self.H[1][:] = -self.Ny*m[1][:]*self.Ms
        self.H[2][:] = -self.Nz*m[2][:]*self.Ms
        return self.H.ravel()

    def compute_energy(self):
        return 0
