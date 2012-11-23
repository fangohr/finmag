import dolfin as df
import logging
from finmag.util.timings import timings
from finmag.energies.energy_base import EnergyBase
from finmag.util.consts import mu0

logger = logging.getLogger('finmag')



class Exchange(EnergyBase):
    """
    Compute the exchange field for LLB case.
    
    Notes: This class just works for one material which means even 
    the magnetisation is not normalised, but a constant value m_e 
    everywhere is expected. 

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
            Ms   = 0.8e6
            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)

            S3  = VectorFunctionSpace(mesh, "Lagrange", 1)
            A  = 1.3e-11 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), S3) # Initial magnetisation

            exchange = Exchange(C, Ms)
            exchange.setup(S3, M)

            # Print energy
            print exchange.compute_energy()

            # Exchange field
            H_exch = exchange.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            exchange_np = Exchange(A, M, C, Ms, method='box-matrix-numpy')
            H_exch_np = exchange_np.compute_field()

    """
    def __init__(self, C, method="box-matrix-petsc"):
        super(Exchange, self).__init__(method, in_jacobian=True)
        self.C = C

    def setup(self, S3, m, Ms, Me, unit_length=1):
        timings.start('Exchange-setup')
        self.Me = Me

        #expression for the energy
        exchange_factor = df.Constant(
            1 * self.C / (mu0 * Ms * unit_length ** 2))

        self.exchange_factor = exchange_factor  # XXX

        E = exchange_factor * mu0 * Ms \
            * df.inner(df.grad(m), df.grad(m)) 
        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        nodal_E = Ms * mu0 * df.dot(self.exchange_factor \
                * df.inner(df.grad(m), df.grad(m)), w) * df.dx

        super(Exchange, self).setup(
                E_integrand=E,
                nodal_E=nodal_E,
                S3=S3,
                m=m,
                Ms=Ms,
                unit_length=unit_length)

        timings.stop('Exchange-setup')
    
    def compute_field(self):
        """
        Compute the field associated with the energy.

         *Returns*
            numpy.ndarray
                The coefficients of the dolfin-function in a numpy array.

        """

        
        Hex = super(Exchange, self).compute_field()
        
        return Hex / self.Me ** 2


if __name__ == "__main__":
    from dolfin import *
    m = 1e-8
    Ms = 0.8e6
    n = 5
    mesh = Box(0, m, 0, m, 0, m, n, n, n)

    S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
    C = 1.3e-11  # J/m exchange constant
    M = project(Constant((1, 0, 0)), S3)  # Initial magnetisation
    exchange = Exchange(1e-11)

    exchange.setup(S3, M, Ms, 1)

    _ = exchange.compute_field()
    _ = exchange.compute_energy()
    _ = exchange.energy_density()

    
    print timings.report_str()

