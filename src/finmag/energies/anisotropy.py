import logging
import dolfin as df
import numpy as np
from aeon import mtimed
from energy_base import EnergyBase
from finmag.util import helpers
from finmag.util.consts import mu0
from finmag.native import llg as native_llg

logger = logging.getLogger('finmag')


class UniaxialAnisotropy(EnergyBase):
    """
    Compute the uniaxial anisotropy field.

    .. math::

        E_{\\text{anis}} = \\int_\\Omega K_1 - (1 - a \\cdot m)^2  dx

    *Arguments*
        K1
            The anisotropy constant
        axis
            The easy axis. Should be a unit vector.
        Ms
            The saturation magnetisation.
        method
            The method used to compute the anisotropy field.
            For alternatives and explanation, see EnergyBase class.

    *Example of Usage*
        .. code-block:: python

            import dolfin as df
            from finmag import UniaxialAnisotropy

            L = 1e-8; nL = 5;
            mesh = df.BoxMesh(0, L, 0, L, 0, L, nL, nL, nL)

            S3 = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
            K = 520e3 # For Co (J/m3)

            a = df.Constant((0, 0, 1)) # Easy axis in z-direction
            m = df.project(df.Constant((1, 0, 0)), V)  # Initial magnetisation
            Ms = 1e6

            anisotropy = UniaxialAnisotropy(K, a)
            anisotropy.setup(S3, m, Ms)

            # Print energy
            print anisotropy.compute_energy()

            # Assign anisotropy field
            H_ani = anisotropy.compute_field()

    """

    def __init__(self, K1, axis, K2=0, method="box-matrix-petsc", name='Anisotropy', assemble=True):
        """
        Define a uniaxial anisotropy with (first) anisotropy constant `K1`
        (in J/m^3) and easy axis `axis`.

        K1 and axis can be passed as df.Constant or df.Function, although
        automatic convertion will be attempted from float for K1 and a
        sequence type for axis. It is possible to specify spatially
        varying anisotropy by using df.Functions.

        """
        self.K1_waiting_for_mesh = K1
        self.axis_waiting_for_mesh = axis
        self.name = name
        super(UniaxialAnisotropy, self).__init__(method, in_jacobian=True)
        self.assemble = assemble
        
        self.K2_input = K2
        if K2!=0:
            self.assemble = False

    @mtimed
    def setup(self, S3, m, Ms, unit_length=1):
        # The following two lines are duplicated again in EnergyBase.setup().
        # I wonder why there is the distinction betwen the __init__() and the
        # setup() methods anyway? It feels a bit artifial to me.  -- Max, 23.9.2013
        dofmap = S3.dofmap()
        S1 = df.FunctionSpace(S3.mesh(), "Lagrange", 1, constrained_domain=dofmap.constrained_domain)

        # Anisotropy energy
        # HF's version inline with nmag, breaks comparison with analytical
        # solution in the energy density test for anisotropy, as this uses
        # the Scholz-Magpar method. Should anyway be a an easy fix when we
        # decide on method.
        # FIXME: we should use DG0 space here?
        self.K1 = helpers.scalar_valued_function(self.K1_waiting_for_mesh, S1)
        self.K1.rename('K1', 'uniaxial anisotropy constant')
        self.K2 = helpers.scalar_valued_function(self.K2_input, S1)
        self.axis = helpers.vector_valued_function(self.axis_waiting_for_mesh, S3, normalise=True)
        self.axis.rename('K1_axis', 'anisotropy axis')
        E_integrand = self.K1 * ( df.Constant(1) - (df.dot(self.axis, m)) ** 2)
        #FIXME: need to re-write the energy density function since the second energy term is not included
            
        super(UniaxialAnisotropy, self).setup(E_integrand, S3, m, Ms, unit_length)
        
        if not self.assemble:
            self.H = self.m.vector().array()
            self.Ms = self.Ms.vector().array()
            self.u = self.axis.vector().array()
            self.K1_arr = self.K1.vector().array()
            self.K2_arr = self.K2.vector().array()
            self.volumes = df.assemble(df.TestFunction(S1) * df.dx)
            self.compute_field = self.__compute_field_directly
            self.compute_energy = self.__compute_energy
    
    def __compute_field_directly(self):
        
        m = self.m.vector().array()
        
        m.shape=(3,-1)
        self.H.shape=(3,-1)
        self.u.shape=(3,-1)
        native_llg.compute_anisotropy_field(m, self.Ms, self.H, self.u, self.K1_arr, self.K2_arr)
        m.shape=(-1,)
        self.H.shape=(-1,)
        self.u.shape=(-1,)
        
        return self.H
    
    def __compute_energy(self):
        m = self.m.vector().array()
        mh = m*self.H
        mh.shape=(3,-1)
        Ei = np.sum(mh, axis=0)*mu0*self.Ms*self.volumes
        E = -0.5*np.sum(Ei) * self.unit_length ** self.dim
        return E
        
