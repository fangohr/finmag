import numpy as np
import dolfin as df

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
        M
            The Dolfin object representing the magnetisation
        K
            The anisotropy constant
        a
            The easy axis (use dolfin.Constant for now)
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

            anisotropy = Anisotropy(V, m, K, a)

            # Print energy
            print anisotropy.compute_energy()

            # Anisotropy field
            H_ani = anisotropy.compute_field()
            
    """   
    
    def __init__(self, V, m, K, a, method="box-matrix-petsc"):
        print "Anisotropy(): method = %s" % method

        # Testfunction
        self.v = df.TestFunction(V)
        
        # Make sure that K is dolfin.Constant
        if not 'dolfin' in str(type(K)):
            K = df.Constant(K)
        
        # Anisotropy energy
        self.E = K*(df.Constant(1) - (df.dot(a, m))**2)*df.dx

        self.E = -K * (df.dot(a, m)**2) *df.dx

        # Gradient
        self.dE_dM = -df.derivative(self.E, m)

        # Volume
        self.vol = df.assemble(df.dot(self.v, 
            df.Constant([1,1,1])) * df.dx).array()

        # Store for later
        self.V = V
        self.m = m
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

    def compute_field(self):
        """
        Compute the anisotropy field.
        
         *Returns*
            numpy.ndarray
                The anisotropy field.       
        
        """
        return self.__compute_field()
    
    def compute_energy(self):
        """
        Compute the anisotropy energy.

        *Returns*
            Float
                The anisotropy energy.

        """
        return df.assemble(self.E)

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
        return df.assemble(self.dE_dM).array() / self.vol

    def __compute_field_numpy(self):
        Mvec = self.m.vector().array()
        H_ani = np.dot(self.g,Mvec)
        return H_ani/self.vol

    def __compute_field_petsc(self):
        self.g_petsc.mult(self.m.vector(), self.H_ani_petsc)
        return self.H_ani_petsc.array()/self.vol

    def __compute_field_project(self):
        df.solve(self.a == self.L, self.H_ani_project)
        return self.H_ani_project.vector().array()
