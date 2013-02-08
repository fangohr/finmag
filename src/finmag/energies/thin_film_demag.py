import logging
import numpy as np
import dolfin as df

logger = logging.getLogger('finmag')


class ThinFilmDemag(object):
    """
    Demagnetising field for thin films in the i-direction.
    Hj = Hk = 0 and Hi = - Mi.

    """
    def __init__(self, direction="z", field_strength=None, in_jacobian=False):
        """
        field_strength is Ms by default

        """
        assert direction in ["x", "y", "z"]
        self.direction = ord(direction) - 120 # converts x,y,z to 0,1,2
        self.strength = field_strength
        self.in_jacobian = in_jacobian
        in_jacobian_msg = "in Jacobian" if in_jacobian else "not in Jacobian"
        logger.debug("Creating {} object, {}.".format(
            self.__class__.__name__, in_jacobian_msg))

    def setup(self, S3, m, Ms, unit_length):
        self.m = m
        self.H = np.zeros((3, S3.mesh().num_vertices()))

        if self.strength == None:
            self.S1 = df.FunctionSpace(S3.mesh(), "Lagrange", 1)
            self.volumes = df.assemble(df.TestFunction(self.S1) * df.dx)
            Ms = df.assemble(Ms * df.TestFunction(self.S1) * df.dx).array() / self.volumes
            self.strength = Ms

    def compute_field(self):
        m = self.m.vector().array().view().reshape((3, -1))
        self.H[self.direction][:] = - self.strength * m[self.direction]
        return self.H.ravel()

    def compute_energy(self):
        return 0
