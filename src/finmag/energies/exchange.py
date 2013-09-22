import dolfin as df
import numpy as np
import logging
from finmag.util.timings import mtimed
from energy_base import EnergyBase
from finmag.util.consts import exchange_length
from finmag.util import helpers
from numbers import Number

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
            See documentation of EnergyBase class for details.


    *Example of Usage*

        .. code-block:: python

            import dolfin as df
            from finmag.energies.exchange import Exchange

            # Define a mesh representing a cube with edge length L
            L = 1e-8
            n = 5
            mesh = df.BoxMesh(0, L, 0, L, 0, L, n, n, n)

            S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
            A = 1.3e-11 # J/m exchange constant
            Ms = 0.8e6
            m = df.project(Constant((1, 0, 0)), S3) # Initial magnetisation

            exchange = Exchange(A)
            exchange.setup(S3, m, Ms)

            # Print energy
            print exchange.compute_energy()

            # Exchange field
            H_exch = exchange.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            exchange_np = Exchange(A, method='box-matrix-numpy')
            exchange_np.setup(S3, m, Ms)
            H_exch_np = exchange_np.compute_field()

    """
    def __init__(self, A, method="box-matrix-petsc", name='Exchange'):
        self.A_waiting_for_mesh = A
        self.name = name
        super(Exchange, self).__init__(method, in_jacobian=True)

    def exchange_length(self):
        def convert_to_number(x):
            if isinstance(x, Number):
                res = x
            else:
                # In this case we assume that x is a Constant,
                # Function, or some other object which supports
                # compute_vertex_values(). TODO: make sure that there
                # are no other cases.
                x_vec = x.compute_vertex_values(self.S3.mesh())
                if not np.allclose(x_vec, x_vec[0], atol=0):
                    raise ValueError(
                        "Exchange constant A and/or saturation magnetisation Ms "
                        "are spatially non-uniform, but they must be constant to "
                        "compute the exchange length.")
                res = x_vec[0]
            return res

        A = convert_to_number(self.A)
        Ms = convert_to_number(self.Ms)
        return exchange_length(A, Ms)

    @mtimed
    def setup(self, S3, m, Ms, unit_length=1):
        self.exchange_factor = df.Constant(1.0 / unit_length ** 2)
        self.S3 = S3
        self.A = helpers.scalar_valued_dg_function(self.A_waiting_for_mesh, self.S3.mesh())
        self.A.rename('A', 'exchange_constant')
        self.A_av = np.average(self.A.vector().array())
        del(self.A_waiting_for_mesh)
        E_integrand = self.exchange_factor * self.A * df.inner(df.grad(m), df.grad(m))

        super(Exchange, self).setup(E_integrand, S3, m, Ms, unit_length)
