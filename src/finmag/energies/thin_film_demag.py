import logging
import numpy as np
from energy_base import AbstractEnergy

logger = logging.getLogger('finmag')

class ThinFilmDemag(AbstractEnergy):
    """
    Demagnetising field for thin films in the i-direction.
    Hj = Hk = 0 and Hi = - Mi.

    """
    def __init__(self, direction="z", in_jacobian=False):
        assert direction in ["x", "y", "z"]
        self.direction = ord(direction) - 120 # converts x,y,z to 0,1,2
        self.in_jacobian = in_jacobian
        in_jacobian_msg = "in Jacobian" if in_jacobian else "not in Jacobian"
        logger.debug("Creating {} object, {}.".format(
            self.__class__.__name__, in_jacobian_msg))

    def setup(self, S3, m, Ms, unit_length):
        self.m = m
        self.Ms = Ms
        self.H = np.zeros((3, S3.mesh().num_vertices()))

    def compute_field(self):
        m = self.m.vector().array().view().reshape((3, -1))
        self.H[self.direction][:] = m[self.direction]
        return - self.Ms * self.H.ravel()

    def compute_energy(self):
        return 0
