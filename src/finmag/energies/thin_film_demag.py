import logging
import numpy as np
from energy_base import AbstractEnergy

logger = logging.getLogger('finmag')

class ThinFilmDemag(AbstractEnergy):
    """
    Demagnetising field for thin films in the z-direction.

    Hx = Hy = 0 and Hz = - Mz.
    Computed by multiplying m by Ms and a matrix with negative ones one the
    last third of the main diagonal, and zeros everywhere else.

    """
    def __init__(self, in_jacobian=False):
        self.in_jacobian = in_jacobian
        in_jacobian_msg = "in Jacobian" if in_jacobian else "not in Jacobian"
        logger.debug("Creating {} object, {}.".format(
            self.__class__.__name__, in_jacobian_msg))

    def setup(self, S3, m, Ms, unit_length):
        nodes = m.vector().array().shape[0] / 3
        diagonal = np.zeros((3, nodes))
        diagonal[2] = -1
        self.g = np.diag(diagonal.ravel())
        self.m = m
        self.Ms = Ms

    def compute_field(self):
        return self.Ms * np.dot(self.g, self.m.vector().array())

    def compute_energy(self):
        return 0
