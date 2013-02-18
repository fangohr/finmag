import logging
import dolfin as df
from finmag.util.timings import mtimed
from finmag.energies.energy_base import EnergyBase
from finmag.util.consts import mu0
from finmag.llb.material import Material

logger = logging.getLogger('finmag')


class ExchangeStd(EnergyBase):
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
            mesh = BoxMesh(0, m, 0, m, 0, m, n, n, n)

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
        super(ExchangeStd, self).__init__(method, in_jacobian=True)
        self.C = C

    @mtimed
    def setup(self, S3, m, Ms, Me, unit_length=1):
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

        super(ExchangeStd, self).setup(
                E_integrand=E,
                nodal_E=nodal_E,
                S3=S3,
                m=m,
                Ms=Ms,
                unit_length=unit_length)
    
    def compute_field(self):
        """
        Compute the field associated with the energy.

         *Returns*
            numpy.ndarray
                The coefficients of the dolfin-function in a numpy array.

        """

        
        Hex = super(ExchangeStd, self).compute_field()
        
        return Hex / self.Me ** 2


class Exchange(object):
    def __init__(self, mat, in_jacobian=False):
        self.C = mat._A_dg
        self.me= mat._m_e
        self.in_jacobian=in_jacobian
   
    @mtimed
    def setup(self, S3, m, Ms0, unit_length=1.0): 
        self.S3 = S3
        self.m = m
        self.Ms0=Ms0
        self.unit_length = unit_length

        self.mu0 = mu0
        self.exchange_factor = 2.0 / (self.mu0 * Ms0 * self.unit_length**2)

        u3 = df.TrialFunction(S3)
        v3 = df.TestFunction(S3)
        self.K = df.PETScMatrix()
        df.assemble(self.C*df.inner(df.grad(u3),df.grad(v3))*df.dx, tensor=self.K)
        self.H = df.PETScVector()
        
        self.vol = df.assemble(df.dot(v3, df.Constant([1, 1, 1])) * df.dx).array()
        
        self.coeff=-self.exchange_factor/(self.vol*self.me**2)
    
    def compute_field(self):
        
        self.K.mult(self.m.vector(), self.H)
         
        return  self.coeff*self.H.array()


if __name__ == "__main__":

    mesh = df.BoxMesh(0, 0, 0, 10, 1, 1, 10, 1, 1)



    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    C = 1.3e-11  # J/m exchange constant
    expr = df.Expression(('4.0*sin(x[0])', '4*cos(x[0])','0'))
    m0 = df.interpolate(expr, S3)
    
    from finmag.llb.material import Material
    mat = Material(mesh, name='FePt')
    mat.set_m(expr)
    mat.T = 1
    mat.alpha=0.01
    
    exch = Exchange(mat)
    exch.setup(mat.S3, mat._m, mat.Ms0, unit_length=1e-9)
    
    #exch2 = ExchangeStd(mat)
    #exch2.setup(mat.S3, mat._m, mat.Ms0, unit_length=1e-9)
    
    #print max(exch2.compute_field()-exch.compute_field())
    
    print exch.compute_field()
    
    #print timings.report()

