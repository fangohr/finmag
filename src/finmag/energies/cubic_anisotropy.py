import logging
import dolfin as df
from aeon import mtimed
from energy_base import EnergyBase
from finmag.util import helpers

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

    def __init__(self, K1, u1, K2, u2, K3, u3, name='CubicAnisotropy'):
        """
        Define a cubic anisotropy with (fourth/six/eigth order) anisotropy
        constants `K1`, `K2`, `K3` (in J/m^3) and corresponding axes
        `u1`, `u2` and `u3`.

        Both the constants and axes can be passed as df.Constant or df.Function,
        although automatic convertion will be attempted from float for K1 and a
        sequence type for axis. It is possible to specify spatially
        varying anisotropy by using df.Functions.

        """
        self.constants = (K1, K2, K3)
        self.axes = (u1, u2, u3)
        self.name = name
        super(CubicAnisotropy, self).__init__("box-matrix-petsc", in_jacobian=True)

    @mtimed
    def setup(self, S3, m, Ms, unit_length=1):
        dofmap = S3.dofmap()
        S1 = df.FunctionSpace(S3.mesh(), "Lagrange", 1, constrained_domain=dofmap.constrained_domain)

        self.K1 = helpers.scalar_valued_function(self.constants[0], S1)
        self.K1.rename('K1', 'fourth order anisotropy constant')
        self.K2 = helpers.scalar_valued_function(self.constants[1], S1)
        self.K2.rename('K2', 'sixth order anisotropy constant')
        self.K3 = helpers.scalar_valued_function(self.constants[2], S1)
        self.K3.rename('K2', 'eigth order anisotropy constant')

        self.u1 = helpers.vector_valued_function(self.axes[0], S3, normalise=True)
        self.u1.rename('u1', 'K1 anisotropy axis')
        self.u2 = helpers.vector_valued_function(self.axes[1], S3, normalise=True)
        self.u2.rename('u2', 'K2 anisotropy axis')
        self.u3 = helpers.vector_valued_function(self.axes[2], S3, normalise=True)
        self.u3.rename('u3', 'K3 anisotropy axis')

        u1msq = df.dot(self.u1, m) ** 2
        u2msq = df.dot(self.u2, m) ** 2
        u3msq = df.dot(self.u3, m) ** 2

        E_integrand = self.K1 * (u1msq * u2msq + u2msq * u3msq + u3msq * u1msq) \
                    + self.K2 * (u1msq * u2msq * u3msq) \
                    + self.K3 * (
                           u1msq ** 2 * u2msq ** 2
                         + u2msq ** 2 * u3msq ** 2
                         + u3msq ** 2 * u1msq ** 2)
        super(CubicAnisotropy, self).setup(E_integrand, S3, m, Ms, unit_length)
