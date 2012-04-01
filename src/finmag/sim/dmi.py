import numpy as np
import dolfin as df
import logging
logger=logging.getLogger('finmag')
from finmag.util.timings import timings

class DMI(object):
    """
    Compute the DMI field.

    .. math::
        
        E_{\\text{DMI}} = \\int_\\Omega D \\vec{M} \\cdot (\\nabla \\times \\vec{M})  dx
        
    *Arguments*
        V 
            a Dolfin VectorFunctionSpace object.
        M 
            the Dolfin object representing the magnetisation
        D
            the Dzyaloshinskii-Moriya constant
        Ms 
            the saturation field
        method
            possible methods are 
                * 'box-assemble' 
                * 'box-matrix-numpy' 
                * 'box-matrix-petsc' [Default]
                * 'project'

    At the moment, we think (all) 'box' methods work..

    - 'box-assemble' is a slower version that assembles the H_dmi for a given M in every
      iteration.

    - 'box-matrix-numpy' precomputes a matrix g, so that H_dmi = g*M, but uses a (dense)
      numpy array to store the matrix. Inefficient for larger meshes.

    - 'box-matrix-petsc' is the same mathematical scheme as 'box-matrix-numpy',
      but uses a PETSc linear algebra backend that supports sparse
      matrices, to exploit the sparsity of g (default choice).

    - 'project': does not use the box method but 'properly projects' the dmi field
      into the function space. Should explore whether this works and/or makes any difference
      (other than being slow.) Untested.


    The equation is chosen as in the publications  Yu-Onose2010, Li-Lin2011, Elhoja-Canals2002, Bode-Heide2007, Bak-Jensen1980.

    *Example of Usage*

        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)
            
            V  = VectorFunctionSpace(mesh, "Lagrange", 1)
            D  = 5e-3 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), V) # Initial magnetisation
           
            dmi = DMI(V, M, D, Ms)

            # Print energy
            print dmi.compute_energy()

            # DMI field 
            H_dmi = dmi.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            dmi_np = Exchange(V, M, D, Ms, method='box-matrix-numpy')
            H_dmi_np = dmi_np.compute_field()
            
    """

    def __init__(self, V, M, D, Ms, method="box-assemble-petsc"):
        timings.start("DMI-init")
        
        logger.info("DMI(): method = %s" % method)
        
        self.V = V
        self.M = M
        self.DMIconstant = df.Constant(D) #Dzyaloshinsky-Moriya Constant
        self.method = method
       
        self.v = df.TestFunction(V)
        #Equation is chosen from the folowing papers
        #Yu-Onose2010, Li-Lin2011, Elhoja-Canals2002, Bode-Heide2007, Bak-Jensen1980
        self.E = self.DMIconstant * df.inner(self.M, df.curl(self.M)) * df.dx
        #Muhbauer2011
        #self.E = self.E = self.DMIconstant * df.inner(self.M, df.curl(self.M)) * df.dx
        #Rossler-Bogdanov2006
        #self.E = self.DMIconstant * df.cross(self.M, df.curl(self.M)) * df.dx
        
        self.dE_dM = df.derivative(self.E, M, self.v)
        self.vol = df.assemble(df.dot(self.v, df.Constant([1,1,1]))*df.dx).array()
        
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

        timings.stop("DMI-init")

    def compute_field(self):
        """
        Compute the DMI field.
        
         *Returns*
            numpy.ndarray
                The DMI field.       
        
        """
        timings.start("DMI-computefield")
        H = self.__compute_field()
        timings.stop("DMI-computefield")
        return H

    
    def compute_energy(self):
        """
        Return the DMI energy.

        *Returns*
            Float
                The DMI energy.

        """
        timings.start("DMI-computenergy")
        E = df.assemble(self.E)
        timings.stop("DMI-computenergy")
        return E

    def __setup_field_numpy(self):
        """
        Linearise dE_dM with respect to M. As we know this is
        linear ( should add reference to Werner Scholz paper and
        relevant equation for g), this creates the right matrix
        to compute dE_dM later as dE_dM=g*M.  We essentially
        compute a Taylor series of the energy in M, and know that
        the first two terms (dE_dM=Hdmi, and ddE_dMdM=g) are the
        only finite ones as we know the expression for the
        energy.

        """
        g_form = df.derivative(self.dE_dM, self.M)
        self.g = df.assemble(g_form).array() #store matrix as numpy array  

    def __setup_field_petsc(self):
        """
        Same as __setup_field_numpy but with a petsc backend.

        """
        g_form = df.derivative(self.dE_dM, self.M)
        self.g_petsc = df.PETScMatrix()
        
        df.assemble(g_form,tensor=self.g_petsc)
        self.H_dmi_petsc = df.PETScVector()

    def __setup_field_project(self):
        #Note that we could make this 'project' method faster by computing the matrices
        #that represent a and L, and only to solve the matrix system in 'compute_field'().
        #IF this method is actually useful, we can do that. HF 16 Feb 2012
        H_dmi_trial = df.TrialFunction(self.V)
        self.a = df.dot(H_dmi_trial, self.v) * df.dx
        self.L = self.dE_dM
        self.H_dmi_project = df.Function(self.V)        

    def __compute_field_assemble(self):
        return df.assemble(self.dE_dM).array() / self.vol

    def __compute_field_numpy(self):
        Mvec = self.M.vector().array()
        H_dmi = np.dot(self.g,Mvec)
        return H_dmi/self.vol

    def __compute_field_petsc(self):
        self.g_petsc.mult(self.M.vector(), self.H_dmi_petsc)
        return self.H_dmi_petsc.array()/self.vol

    def __compute_field_project(self):
        df.solve(self.a == self.L, self.H_dmi_project)
        return self.H_dmi_project.vector().array()
