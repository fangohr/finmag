import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from energy_base import EnergyBase
from finmag.util.consts import mu0, exchange_length

logger = logging.getLogger('finmag')


class Exchange(EnergyBase):
    """
    Compute the exchange field.

    .. math::

        E_{\\text{exch}} = \\int_\\Omega A (\\nabla M)^2  dx


    *Arguments*

        A
            the exchange constant

        method
            possible methods are
                * 'box-assemble'
                * 'box-matrix-numpy'
                * 'box-matrix-petsc' [Default]
                * 'project'        

            See documentation of EnergyBase class for details.


    *Example of Usage*

        .. code-block:: python

            from dolfin import *
            from finmag.energies.exchange import Exchange

            # Define a mesh representing a cube with edge length L
            L    = 1e-8
            n    = 5
            mesh = Box(0, L, 0, L, 0, L, n, n, n)

            S3  = VectorFunctionSpace(mesh, "Lagrange", 1)
            A  = 1.3e-11 # J/m exchange constant
            Ms   = 0.8e6
            M  = project(Constant((Ms, 0, 0)), S3) # Initial magnetisation

            exchange = Exchange(A)
            exchange.setup(S3, M, Ms)

            # Print energy
            print exchange.compute_energy()

            # Exchange field
            H_exch = exchange.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            exchange_np = Exchange(A, method='box-matrix-numpy')
            exchange_np.setup(S3, M, Ms)
            H_exch_np = exchange_np.compute_field()

    """
    def __init__(self, A, method="box-matrix-petsc"):
        super(Exchange, self).__init__(method, in_jacobian=True)
        self.A = A

    def exchange_length(self):
        return exchange_length(self.A, self.Ms)

    def setup(self, S3, m, Ms, unit_length=1):
        timings.start('Exchange-setup')

        #expression for the energy
        self.exchange_factor = df.Constant(self.A / unit_length ** 2)
        E_integrand = self.exchange_factor * df.inner(df.grad(m), df.grad(m))

        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        nodal_E = df.dot(self.exchange_factor \
                * df.inner(df.grad(m), df.grad(m)), w) * df.dx

        super(Exchange, self).setup(
                E_integrand=E_integrand,
                nodal_E=nodal_E,
                S3=S3,
                m=m,
                Ms=Ms,
                unit_length=unit_length)

        timings.stop('Exchange-setup')


if __name__ == "__main__":
    from dolfin import *
    L = 1e-8
    Ms = 0.8e6
    n = 5
    mesh = Box(0, L, 0, L, 0, L, n, n, n)

    S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
    A = 1.0e-11  # J/m exchange constant
    M = project(Constant((1, 0, 0)), S3)  # Initial magnetisation
    exchange = Exchange(A)

    exchange.setup(S3, M, Ms)

    _ = exchange.compute_field()
    _ = exchange.compute_energy()
    _ = exchange.energy_density()

    print timings.report_str()

