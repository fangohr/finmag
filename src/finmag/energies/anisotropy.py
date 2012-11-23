import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from finmag.util.consts import mu0
from finmag.util.helpers import fnormalise
from energy_base import EnergyBase
logger = logging.getLogger('finmag')


class UniaxialAnisotropy(EnergyBase):
    """
    Compute the exchange field.

    .. math::

        E_{\\text{exch}} = \\int_\\Omega A (\\nabla M)^2  dx

    *Arguments*
        K1
            The anisotropy constant
        axis
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
    
    def __init__(self, K1, axis, method="box-matrix-petsc"):
        """
        Define a uniaxial anisotropy with (first) anisotropy constant `K1`
        (in J/m^3) and easy axis `axis`.

        If K1 and axis are dolfin-functions, then accept them as they are.
        Otherwise, assume they are dolfin constants.
        If they are not dolfin constants (but a float for K1, or sequence
        for axis), try to convert them to dolfin constants.

        Dolfin-functions are required for spatially varying anisotropy
        """
        if isinstance(K1, df.Function):
            self.K1 = K1
        else:
            # Make sure that K1 is dolfin.Constant
            if not 'dolfin' in str(type(K1)):
                K1 = df.Constant(K1)  # or convert to df constant
            self.K1 = K1

        if isinstance(axis, (df.Function, df.Constant)):
            logger.debug("Found Anisotropy direction of type {}".format(axis.__class__))
            self.axis = axis
        else:
            logger.debug("Found Anisotropy direction '{}' -> df.Constant".format(axis))
            self.axis = df.Constant(axis)

        super(UniaxialAnisotropy, self).__init__(method, in_jacobian=True)

    def setup(self, S3, m, Ms, unit_length=1):
        timings.start('UniaxialAnisotropy-setup')

        # Testfunction
        self.v = df.TestFunction(S3)

        # Anisotropy energy
        E_integrand = self.K1 * (df.Constant(1) - (df.dot(self.axis, m)) ** 2)

        # HF's version inline with nmag, breaks comparison with analytical
        # solution in the energy density test for anisotropy, as this uses
        # the Scholz-Magpar method. Should anyway be a an easy fix when we
        # decide on method.
        #self.E = -K*(df.dot(a, self.m))**2*df.dx

        # Volume
        self.vol = df.assemble(df.dot(self.v,
            df.Constant([1, 1, 1])) * df.dx).array()

        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        self.nodal_vol = df.assemble(w * df.dx, mesh=S3.mesh()).array()
        nodal_E = df.dot(self.K1 * (df.Constant(1)
                        - (df.dot(self.axis, m)) ** 2), w) * df.dx

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(S1)

        super(UniaxialAnisotropy, self).setup(
                E_integrand=E_integrand,
                nodal_E=nodal_E,
                S3=S3,
                m=m,
                Ms=Ms,
                unit_length=unit_length)

        timings.stop('UniaxialAnisotropy-setup')

    def normed_m(self):
        """
        Note that this code is not used at the moment, i.e. we rely on M being 
        normalised. As Weiwei is working on the Barhakhter model, we need to 
        address this properly with him anyway, and can then decide whether to 
        follow the model shown below in this function, or to do something else.

        Hans, 30 June 2012.
        

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
        return self.m  # old behaviour

        self._m_normed.vector()[:] = fnormalise(self._m.vector().array())
        return self._m_normed

if __name__ == "__main__":
    from dolfin import *
    m = 1e-8
    Ms = 0.8e6
    n = 5
    mesh = Box(0, m, 0, m, 0, m, n, n, n)

    S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
    C = 1.3e-11  # J/m exchange constant
    M = project(Constant((Ms, 0, 0)), S3)  # Initial magnetisation
    uniax = UniaxialAnisotropy(K1=1e11, axis=[1, 0, 0])

    uniax.setup(S3, M, Ms)

    _ = uniax.compute_field()
    _ = uniax.compute_energy()
    _ = uniax.energy_density()

    print uniax.name
    print timings.report_str()
