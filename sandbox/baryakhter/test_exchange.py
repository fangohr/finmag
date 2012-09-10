import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from finmag.util.meshes import mesh_volume
from finmag.energies.energy_base import EnergyBase
from finmag.energies.exchange import Exchange

logger=logging.getLogger('finmag')



class BaryakhtarExchange(EnergyBase):
    """
    
    """
   
    def __init__(self, C, chi=1, method="box-matrix-numpy"):
        logger.debug("Creating Exchange object with method {}.".format(method))
        self.in_jacobian = True        
        self.C = C
        self.chi=chi
        self.method = method
      
    def setup(self, S3, m, Ms, unit_length=1): 
        timings.start('Exchange-setup')

        self.S3 = S3
        self.m = m
        self.Ms=Ms
        self.init_Ms=np.array(Ms.vector().array())
        self.unit_length = unit_length

        self.mu0 = 4 * np.pi * 10**-7 # Vs/(Am)
        self.exchange_factor = df.Constant(1 * self.C / (self.mu0  * self.unit_length**2))

        self.v = df.TestFunction(S3)
        self.E = self.exchange_factor * df.inner(df.grad(m), df.grad(m)) * df.dx
        self.dE_dM = -1 * df.derivative(self.E, m, self.v)
        self.vol = df.assemble(df.dot(self.v, df.Constant([1, 1, 1])) * df.dx).array()
        self.coeff=0.5/self.chi/self.Ms.vector().array()**2/self.vol
        self.dim = S3.mesh().topology().dim()

        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        self.nodal_vol = df.assemble(w * df.dx, mesh=S3.mesh()).array() \
                * unit_length**self.dim
        self.nodal_E = df.dot(self.exchange_factor \
                * df.inner(df.grad(m), df.grad(m)), w) * df.dx

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(S1)

        # Don't know if this is needed
        self.total_vol = mesh_volume(S3.mesh()) * unit_length**self.dim

        if self.method=='box-assemble':
            self.__compute_field = self.__compute_field_assemble
        elif self.method == 'box-matrix-numpy':
            self.__setup_field_numpy()
            self.__compute_field = self.__compute_field_numpy
        elif self.method == 'box-matrix-petsc':
            self.__setup_field_petsc()
            self.__compute_field = self.__compute_field_petsc
        elif self.method=='project':
            self.__setup_field_project()
            self.__compute_field = self.__compute_field_project
        else:
            print "Desired method was {}.".format(self.method)
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'box-assemble', 
                                    * 'box-matrix-numpy',
                                    * 'box-matrix-petsc'  
                                    * 'project'""")

        timings.stop('Exchange-setup')

    def compute_field(self):
        """
        Compute the exchange field.
        
         *Returns*
            numpy.ndarray
                The exchange field.       
        
        """
        timings.start('Exchange-computefield')
        Hex = self.__compute_field()
        timings.stop('Exchange-computefield')
        return Hex

    def compute_energy(self):
        """
        Return the exchange energy.

        *Returns*
            Float
                The exchange energy.

        """
        """
        timings.start('Exchange-energy')
        E = df.assemble(self.E) * self.unit_length**self.dim * self.Ms * self.mu0
        timings.stop('Exchange-energy')
        return E
        """

    def energy_density(self):
        """
        Compute the exchange energy density,

        .. math::

            \\frac{E_\\mathrm{exch}}{V},

        where V is the volume of each node.

        *Returns*
            numpy.ndarray
                The exchange energy density.

        """

        """
        nodal_E = df.assemble(self.nodal_E).array() \
                * self.unit_length**self.dim * self.Ms * self.mu0
        return nodal_E/self.nodal_vol
        """

    def energy_density_function(self):
        """
        Compute the exchange energy density the same way as the
        function above, but return a Function to allow probing.

        *Returns*
            dolfin.Function
                The exchange energy density.

        """
        """
        self.ED.vector()[:] = self.energy_density()
        return self.ED
        """

    def __setup_field_numpy(self):
        """
        Linearise dE_dM with respect to M. As we know this is
        linear ( should add reference to Werner Scholz paper and
        relevant equation for g), this creates the right matrix
        to compute dE_dM later as dE_dM=g*M.  We essentially
        compute a Taylor series of the energy in M, and know that
        the first two terms (dE_dM=Hex, and ddE_dMdM=g) are the
        only finite ones as we know the expression for the
        energy.

        """
        g_form = df.derivative(self.dE_dM, self.m)
        self.g = df.assemble(g_form).array() #store matrix as numpy array  


    def __compute_field_numpy(self):
        m=self.m.vector().array()
        Ms=self.Ms.vector().array()
        H_ex = np.dot(self.g, m)
        relax = Ms*m*self.coeff*(self.init_Ms**2-Ms**2)
        return H_ex/Ms/self.vol+relax

   

    def  __compute_field_petsc(self):
        
        self.g_petsc.mult(m, self.H_ex_petsc)
        return self.H_ex_petsc.array()/self.vol



if __name__=="__main__":
    mesh=df.Interval(2,0,1)

   
    S3  = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    C  = 1.3e-11
   
    Msc = 8.6e5
    expr=df.Expression(('sin(x[0])','cos(x[0])','0'))
    m  = df.project(expr, S3) # Initial magnetisation
    Ms = df.Function(df.FunctionSpace(mesh,'DG',0))
    Ms.vector()[0]=Msc
    Ms.vector()[1]=Msc
   
    Mu =df.Function(S3)
    tmp = df.assemble(Ms*df.dot(df.TestFunction(S3), df.Constant([1, 1, 1])) * df.dx)
    Mu.vector()[:]=tmp*m.vector()
    print 'm_init',m.vector().array()
    print 'Minit',Mu.vector().array()


    exch_tmp = ExchangeTmp(C)
    exch_tmp.setup(S3, Mu,1e-9)
    
    tmp_field=exch_tmp.compute_field()
    print 'tmp',tmp_field

    exch=Exchange(C)
    exch.setup(S3,m,Msc,1e-9)
    field=exch.compute_field()
    print 'diff',tmp_field-field
    
