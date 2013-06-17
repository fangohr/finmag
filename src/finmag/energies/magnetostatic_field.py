import numpy as np
from finmag.util.consts import mu0

EPS = 1e-15


class MagnetostaticField(object):
    """
    Magnetostatic field for uniform magnetisation.

    The magnetostatic energy density for a uniform magnetisation can
    be written as:

    .. math::

        U_{\\text{ms}} =
            \\frac{1}{2} \\mu_0 M_\\text{s}^2
            \\left( N_{xx}m_x^2 + N_{yy}m_y^2 + N_{zz}m_z^2 \\right)

    where :math:`N_{xx}, N_{yy}, N_{zz}` are the shape-dependent demagnetising
    factors along the x, y, and z directions. The demagnetising factors sum
    to 1 and are smallest for directions along the longest dimensions of the
    magnetic element.

    The magnetostatic field is obtained by differentiating the energy with
    respect to the magnetisation:

    .. math::

        \\vec{H}_{\\text{ms}} =
            - M_\\text{s}
            \\left( N_{xx}m_x \\hat{x} + N_{yy}m_y \\hat{y} + N_{zz}m_z \\hat{z} \\right)

    The magnetostatic field along a particular axis is proportional to the
    magnetisation along that axis and points opposite to the direction of
    the magnetisation.

    Since this class is more likely to be used in a macrospin model it
    doesn't try to fit in with the code written with the finite element method
    in mind.

    """
    def __init__(self, Ms, Nxx, Nyy, Nzz):
        """
        Initialise a UniformDemag object.

        Ms is the saturation magnetisation in A/m and the demagnetising
        factors should be given as floats that sum to 1.

        """
        assert abs(sum((Nxx, Nyy, Nzz)) - 1) < EPS, "Demagnetising factors do not sum to 1."
        self.Ms = Ms
        self.N = np.array((Nxx, Nyy, Nzz))

    def compute_energy(self, m):
        """
        Compute the magnetostatic energy density.
        The unit magnetisation m should be given as some kind of iterable of length 3.

        """
        return mu0 * pow(self.Ms, 2) * np.dot(self.N, np.square(m)) / 2

    def compute_field(self, m):
        """
        Compute the magnetostatic field.
        The unit magnetisation m should be given as some kind of iterable of length 3.

        """
        return - self.Ms * (self.N * m)  # element-wise multiplication
