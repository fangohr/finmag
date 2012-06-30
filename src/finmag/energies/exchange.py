import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from energy_base import EnergyBaseExchange as EnergyBase
from finmag.util.consts import mu0

logger = logging.getLogger('finmag')


class Exchange(EnergyBase):
    """
    Compute the exchange field.

    .. math::
        
        E_{\\text{exch}} = \\int_\\Omega A (\\nabla M)^2  dx
        
    *Arguments*
        C 
            the exchange constant
        method
            possible methods are 
                * 'box-assemble' 
                * 'box-matrix-numpy' 
                * 'box-matrix-petsc' [Default]
                * 'project'

    At the moment, we think (all) 'box' methods work 
    (and the method is used in Magpar and Nmag).

    - 'box-assemble' is a slower version that assembles the H_ex for a given M in every
      iteration.

    - 'box-matrix-numpy' precomputes a matrix g, so that H_ex = g*M

    - 'box-matrix-petsc' is the same mathematical scheme as 'box-matrix-numpy',
      but uses a PETSc linear algebra backend that supports sparse
      matrices, to exploit the sparsity of g (default choice).

    - 'project': does not use the box method but 'properly projects' the exchange field
      into the function space. Should explore whether this works and/or makes any difference
      (other than being slow.) Untested.


    *Example of Usage*

        .. code-block:: python
            from dolfin import *
            Ms   = 0.8e6
            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)

            S3  = VectorFunctionSpace(mesh, "Lagrange", 1)
            C  = 1.3e-11 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), S3) # Initial magnetisation

            exchange = Exchange(C, Ms)
            exchange.setup(S3, M)

            # Print energy
            print exchange.compute_energy()

            # Exchange field
            H_exch = exchange.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            exchange_np = Exchange(V, M, C, Ms, method='box-matrix-numpy')
            H_exch_np = exchange_np.compute_field()

    """
    def __init__(self, C, method="box-matrix-petsc"):
        EnergyBase.__init__(self,
            name='exchange',
            method=method,
            in_jacobian=True)
        logger.debug("Creating Exchange object with method {}.".format(method))
        self.C = C

    def setup(self, S3, M, Ms, unit_length=1):
        timings.start('Exchange-setup')

        #expression for the energy
        exchange_factor = df.Constant(
            1 * self.C / (mu0 * Ms * unit_length ** 2))

        self.exchange_factor = exchange_factor  # XXX

        E = exchange_factor * mu0 * Ms \
            * df.inner(df.grad(M), df.grad(M)) * df.dx

        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)

        self.S3 = S3  # keep reference as migth need it again
        w = df.TestFunction(S1)
        nodal_E = Ms * mu0 * df.dot(self.exchange_factor \
                * df.inner(df.grad(M), df.grad(M)), w) * df.dx

        EnergyBase.setup(self,
                E=E,
                nodal_E=nodal_E,
                S3=S3,
                M=M,
                Ms=Ms,
                unit_length=unit_length)

        timings.stop('Exchange-setup')


if __name__ == "__main__":
    from dolfin import *
    m = 1e-8
    Ms = 0.8e6
    n = 5
    mesh = Box(0, m, 0, m, 0, m, n, n, n)

    S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
    C = 1.3e-11  # J/m exchange constant
    M = project(Constant((Ms, 0, 0)), S3)  # Initial magnetisation
    exchange = Exchange(1e-11)

    exchange.setup(S3, M, Ms)

    _ = exchange.compute_field()
    _ = exchange.compute_energy()
    _ = exchange.energy_density()

    print exchange.name
    print timings.report_str()

