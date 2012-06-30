import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from finmag.util.consts import mu0
from finmag.sim.helpers import fnormalise
from energy_base import EnergyBase
logger = logging.getLogger('finmag')


class UniaxialAnisotropy(EnergyBase):
    """
    Compute the exchange field.

    .. math::

        E_{\\text{exch}} = \\int_\\Omega A (\\nabla M)^2  dx

    *Arguments*
        K
            The anisotropy constant
        a
            The easy axis (use dolfin.Constant for now).
            Should be a unit vector.
        Ms
            The saturation magnetisation.
        method
            The method used to compute the anisotropy field.
            For alternatives and explanation, see EnergyBase class.

    *Example of Usage*
        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)

            S3 = VectorFunctionSpace(mesh, 'Lagrange', 1)
            K = 520e3 # For Co (J/m3)

            a = Constant((0, 0, 1)) # Easy axis in z-direction
            m = project(Constant((1, 0, 0)), V)  # Initial magnetisation
            Ms = 1e6

            anisotropy = Anisotropy(K, a, Ms)
            anisotropy.setup(S3, m)

            # Print energy
            print anisotropy.compute_energy()

            # Anisotropy field
            H_ani = anisotropy.compute_field()

    """
    
    def __init__(self, K, a, method="box-matrix-petsc"):
        """If K and a are dolfin-functions, then accept them as they are.
        Otherwise, assume they are dolfin constants.
        If they are not dolfin constants (but a float for K, or sequence
            for a), try to convert them to dolfin constants.

        Dolfin-functions are required for spatially varying anisotropy
        """
        if isinstance(K, df.Function):
            self.K = K
        else:
            # Make sure that K is dolfin.Constant
            if not 'dolfin' in str(type(K)):
                K = df.Constant(K)  # or convert to df constant
            self.K = K

        if isinstance(a, (df.Function, df.Constant)):
            logger.debug("Found Anisotropy direction a of type {}".format(a.__class__))
            self.a = a
        else:
            logger.debug("Found Anisotropy direction '{}' -> df.Constant".format(a))
            self.a = df.Constant(a)

        self.method = method
        EnergyBase.__init__(self,
            name='UniaxialAnisotropy',
            method=method,
            in_jacobian=True)
        logger.debug("Creating UniaxialAnisotropy object with method {}.".format(method))

    def setup(self, S3, M, Ms, unit_length=1):
        timings.start('UniaxialAnisotropy-setup')

        #self._m_normed = df.Function(S3)
        self.M = M

        # Testfunction
        self.v = df.TestFunction(S3)

        # Anisotropy energy
        E = self.K * (df.Constant(1) - (df.dot(self.a, self.M)) ** 2) * df.dx

        # HF's version inline with nmag, breaks comparison with analytical
        # solution in the energy density test for anisotropy, as this uses
        # the Scholz-Magpar method. Should anyway be a an easy fix when we
        # decide on method.
        #self.E = -K*(df.dot(a, self.m))**2*df.dx

        # Gradient
        self.dE_dM = df.Constant(-1.0
            / (Ms * mu0)) * df.derivative(E, self.M)

        # Volume
        self.vol = df.assemble(df.dot(self.v,
            df.Constant([1, 1, 1])) * df.dx).array()

        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        self.nodal_vol = df.assemble(w * df.dx, mesh=S3.mesh()).array()
        nodal_E = df.dot(self.K * (df.Constant(1)
                        - (df.dot(self.a, self.m)) ** 2), w) * df.dx

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(S1)

        EnergyBase.setup(self,
                E=E,
                nodal_E=nodal_E,
                S3=S3,
                M=M,
                Ms=Ms,
                unit_length=unit_length)

        timings.stop('UniaxialAnisotropy-setup')

    def normed_m(self):
        """
        There are three possibilities to get a magnetisation that is normalised
        for sure.

        1. Disrupt the whole application by normalising the shared
        magnetisation object that was given to us. Bad.
        2. Keep a reference to the original magnetisation, and create a
        local copy of it, that is always normalised. That is what
        is implemented.
        3. Normalise the magnetisation on the fly during the computation.
        That would be quite nice, however, we are dealing with dolfin
        functions (or vectors) and not simply arrays.

        Note that normalisation is disabled as of now, because some tests
        fail (not because of faults, but because normalising changes the
        test results slightly). To see which, uncomment the last two lines
        of this function and comment the first return.

        """
        return self.M  # old behaviour

        self._m_normed.vector()[:] = fnormalise(self._m.vector().array())
        return self._m_normed

    m = property(normed_m)


if __name__ == "__main__":
    from dolfin import *
    m = 1e-8
    Ms = 0.8e6
    n = 5
    mesh = Box(0, m, 0, m, 0, m, n, n, n)

    S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
    C = 1.3e-11  # J/m exchange constant
    M = project(Constant((Ms, 0, 0)), S3)  # Initial magnetisation
    uniax = UniaxialAnisotropy(K=1e11, a=[1, 0, 0])

    uniax.setup(S3, M, Ms)

    _ = uniax.compute_field()
    _ = uniax.compute_energy()
    _ = uniax.energy_density()

    print uniax.name
    print timings.report_str()
