import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from finmag.sim.helpers import fnormalise
logger=logging.getLogger('finmag')

class UniaxialAnisotropy(object):
    """
    Compute the anisotropy field.

    The magnetocrystalline anisotropy energy for uniaxial
    anisotropy is given by

    .. math::

        E_{\\text{ani}} = - \\int K ( \\vec a \\cdot \\vec m)^2 \\mathrm{d}x,

    where :math:`K` is the anisotropy constant,
    :math:`\\vec a` the easy axis and :math:`\\vec{m}=\\vec{M}/M_\mathrm{sat}`
    the discrete approximation of the magnetic polarization.

    *Arguments*
        V
            A Dolfin VectorFunctionSpace object.
        m
            The Dolfin object representing the (unit) magnetisation
        K
            The anisotropy constant
        a
            The easy axis (use dolfin.Constant for now).
            Should be a unit vector.
        Ms
            The saturation magnetisation.
        method
            The method used to compute the anisotropy field.
            For alternatives and explanation, see Exchange.

    *Example of Usage*
        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)

            V = VectorFunctionSpace(mesh, 'Lagrange', 1)
            K = 520e3 # For Co (J/m3)

            a = Constant((0, 0, 1)) # Easy axis in z-direction
            m = project(Constant((1, 0, 0)), V) # Initial magnetisation
            Ms = 1e6

            anisotropy = Anisotropy(V, m, K, a, Ms)

            # Print energy
            print anisotropy.compute_energy()

            # Anisotropy field
            H_ani = anisotropy.compute_field()

    """

    def __init__(self, V, m, K, a, Ms, method="box-matrix-petsc"):
        logger.info("Anisotropy() method = %s" % method)
        timings.start('Anisotropy-init')

        self._m_normed = df.Function(V)
        self._m = m

        # Testfunction
        self.v = df.TestFunction(V)

        # Make sure that K is dolfin.Constant
        if not 'dolfin' in str(type(K)):
            K = df.Constant(K)

        # Anisotropy energy
        self.E = K * (df.Constant(1) - (df.dot(a, self.m))**2)*df.dx

        # Gradient
        mu0 = 4*np.pi*1e-7
        self.dE_dM = df.Constant(-1.0/(Ms*mu0))*df.derivative(self.E, self.m)

        # Volume
        self.vol = df.assemble(df.dot(self.v,
            df.Constant([1,1,1])) * df.dx).array()

        # Needed for energy density
        FS = df.FunctionSpace(V.mesh(), "CG", 1)
        w = df.TestFunction(FS)
        self.nodal_vol = df.assemble(w*df.dx, mesh=V.mesh()).array()
        self.nodal_E = df.dot(K*(df.Constant(1) - (df.dot(a, self.m))**2), w)*df.dx

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(FS)

        # Store for later
        self.V = V
        self.method = method

        if method=='box-assemble':
            self.__compute_field = self.__compute_field_assemble
        elif method == 'box-matrix-numpy':
            self.__setup_field_numpy()
            self.__compute_field = self.__compute_field_numpy
        elif method == 'box-matrix-petsc':
            self.__setup_field_petsc()
            self.__compute_field = self.__compute_field_petsc
        elif method=='project':
            self.__setup_field_project()
            self.__compute_field = self.__compute_field_project
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'box-assemble',
                                    * 'box-matrix-numpy',
                                    * 'box-matrix-petsc'
                                    * 'project'""")

        timings.stop('Anisotropy-init')

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
        return self._m # old behaviour
        #self._m_normed.vector()[:] = fnormalise(self._m.vector().array())
        #return self._m_normed
    m = property(normed_m)

    def compute_field(self):
        """
        Compute the anisotropy field.

         *Returns*
            numpy.ndarray
                The anisotropy field.

        """
        timings.start('Anisotropy-computefield')
        H = self.__compute_field()
        timings.stop('Anisotropy-computefield')
        return H

    def compute_energy(self):
        """
        Compute the anisotropy energy.

        *Returns*
            Float
                The anisotropy energy.

        """
        timings.start('Anisotropy-energy')
        E=df.assemble(self.E)
        timings.stop('Anisotropy-energy')
        return E

    def energy_density(self):
        nodal_E = df.assemble(self.nodal_E).array()
        return nodal_E/self.nodal_vol

    def density_function(self):
        self.ED.vector()[:] = self.energy_density()
        return self.ED


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

    def __setup_field_petsc(self):
        """
        Same as __setup_field_numpy but with a petsc backend.

        """
        g_form = df.derivative(self.dE_dM, self.m)
        self.g_petsc = df.PETScMatrix()

        df.assemble(g_form,tensor=self.g_petsc)
        self.H_ani_petsc = df.PETScVector()

    def __setup_field_project(self):
        #Note that we could make this 'project' method faster by computing
        #the matrices that represent a and L, and only to solve the matrix
        #system in 'compute_field'(). IF this method is actually useful,
        #we can do that. HF 16 Feb 2012
        H_ani_trial = df.TrialFunction(self.V)
        self.a = df.dot(H_ani_trial, self.v) * df.dx
        self.L = self.dE_dM
        self.H_ani_project = df.Function(self.V)

    def __compute_field_assemble(self):
        H_ani=df.assemble(self.dE_dM).array() / self.vol
        return H_ani

    def __compute_field_numpy(self):
        Mvec = self.m.vector().array()
        H_ani = np.dot(self.g,Mvec)/self.vol
        return H_ani

    def __compute_field_petsc(self):
        self.g_petsc.mult(self.m.vector(), self.H_ani_petsc)
        H_ani = self.H_ani_petsc.array()/self.vol
        return H_ani

    def __compute_field_project(self):
        df.solve(self.a == self.L, self.H_ani_project)
        H_ani = self.H_ani_project.vector().array()
        raise NotImplementedError("This has never been tested and is not meant to work.")
        return H_ani
