import numpy as np
import dolfin as df
from energy_base import EnergyBase
import logging
logger=logging.getLogger('finmag')
from finmag.util.timings import mtimed
from finmag.util.consts import mu0


def dmi_term3d_dolfin(M,v,c):
    E = c * df.inner(M, df.curl(M))
    nodal_E = df.dot(c * df.inner(M, df.curl(M)), v) * df.dx
    return E, nodal_E

def dmi_term3d(M,v,c,debug=False):
    """Input arguments:

       M a dolfin 3d-vector function on a 3d space
       c the DMI constant.

       Returns the form to compute the DMI energy:

         c * df.inner(M, df.curl(M)) * df.dx                      (1)

       Instead of using this equation, we  can spell out the curl M as:

         curlx = dMzdy - dMydz
         curly = dMxdz - dMzdx
         curlz = dMydx - dMxdy

       and do the scalar product with M manually:

         E = c*(Mx*curlx + My*curly + Mz*curlz)*df.dx

       and this is how we compute the rotation in this routine.

       The routine itself is not very useful, the dolfin command in (1)
       is more effective. However, once this works, we can
    """



    if debug:
        Mx, My, Mz = M.split()
        print "Mx=", df.assemble(Mx * df.dx)
        print "My=", df.assemble(My * df.dx)
        print "Mz=", df.assemble(Mz * df.dx)

    gradM=df.grad(M)

    dMxdx = gradM[0, 0]
    dMxdy = gradM[0, 1]
    dMxdz = gradM[0, 2]
    dMydx = gradM[1, 0]
    dMydy = gradM[1, 1]
    dMydz = gradM[1, 2]
    dMzdx = gradM[2, 0]
    dMzdy = gradM[2, 1]
    dMzdz = gradM[2, 2]

    if debug:
        for i in range(3):
            for j in range(3):
                print "i=%d, j=%d" % (i, j),
                print df.assemble(gradM[i, j] * df.dx)

    curlx = dMzdy - dMydz
    curly = dMxdz - dMzdx
    curlz = dMydx - dMxdy

    if debug:

        print "curlx=", df.assemble(curlx * df.dx)
        print "curly=", df.assemble(curly * df.dx)
        print "curlz=", df.assemble(curlz * df.dx)

    #original equation:
    #E = c * df.inner(M, df.curl(M)) * df.dx

    #our version:
    E = c * (M[0] * curlx + M[1] * curly + M[2] * curlz)
    nodal_E = df.dot(c * (M[0] * curlx + M[1] * curly + M[2] * curlz), v) * df.dx

    return E, nodal_E


def dmi_term2d(M, v, c, debug=False):
    """Input arguments:

       M a dolfin 3d-vector function on a 2d space,
       c the DMI constant.

       Returns the form to compute the DMI energy:

         c * df.inner(M, df.curl(M)) * df.dx                      (1)

       However, curl(M) cannot be computed on a 2d mesh. We thus do it manually.

       Instead of using (1) equation, we  can spell out the curl M as:

         curlx = dMzdy - dMydz
         curly = dMxdz - dMzdx
         curlz = dMydx - dMxdy

       and do the scalar product with M manually:

         E = c*(Mx*curlx + My*curly + Mz*curlz)*df.dx

       We set dMx/dz = 0, dMy/dz=0, dMz/dz=0, assuming that the 2d mesh
       lives in the x-y plane, and thus the physics cannot change as a function
       of z.

       (Fully analog to dmi_term3d(M,c), which in turn is equivalent to
        inner(M,curl(M))*dx).

    """

    if debug:
        Mx, My, Mz = M.split()
        print "Mx=", df.assemble(Mx * df.dx)
        print "My=", df.assemble(My * df.dx)
        print "Mz=", df.assemble(Mz * df.dx)

    gradM = df.grad(M)

    dMxdx = gradM[0, 0]
    dMxdy = gradM[0, 1]
    #dMxdz= gradM[0, 2]
    dMxdz = 0
    dMydx = gradM[1, 0]
    dMydy = gradM[1, 1]
    #dMydz= gradM[1, 2]
    dMydz = 0
    dMzdx = gradM[2, 0]
    dMzdy = gradM[2, 1]
    #dMzdz= gradM[2, 2]
    dMzdz = 0

    if debug:
        for i in range(3):
            for j in range(2):
                print "i=%d, j=%d" % (i, j),
                print df.assemble(gradM[i, j] * df.dx)

    curlx = dMzdy - dMydz
    curly = dMxdz - dMzdx
    curlz = dMydx - dMxdy

    if debug:
        print "curlx=", df.assemble(curlx * df.dx)
        print "curly=", df.assemble(curly * df.dx)
        print "curlz=", df.assemble(curlz * df.dx)

    #original equation:
    #E = c * df.inner(M, df.curl(M)) * df.dx

    #our version:
    E = c * (M[0] * curlx + M[1] * curly + M[2] * curlz)
    nodal_E = df.dot(c * (M[0] * curlx + M[1] * curly + M[2] * curlz), v) * df.dx

    return E, nodal_E


class DMI(EnergyBase):
    """
    Compute the DMI field.

    .. math::

        E_{\\text{DMI}} = mu_0 M_s \\int_\\Omega D \\vec{M} \\cdot (\\nabla \\times \\vec{M})  dx

    *Arguments*
        S3
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

            See EnergyBase for details on the method.

    The equation is chosen as in the publications  Yu-Onose2010, Li-Lin2011, Elhoja-Canals2002, Bode-Heide2007, Bak-Jensen1980.

    *Example of Usage*

        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = BoxMesh(0, m, 0, m, 0, m, n, n, n)

            S3  = VectorFunctionSpace(mesh, "Lagrange", 1)
            D  = 5e-3 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), V) # Initial magnetisation

            dmi = DMI(S3, M, D, Ms)

            # Print energy
            print dmi.compute_energy()

            # DMI field
            H_dmi = dmi.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            dmi_np = Exchange(S3, M, D, Ms, method='box-matrix-numpy')
            H_dmi_np = dmi_np.compute_field()
    """

    def __init__(self, D, method="box-matrix-petsc",pbc2d=None):
        super(DMI, self).__init__(method, in_jacobian=True,pbc2d=pbc2d)
        self.D = D

    @mtimed
    def setup(self, S3, M, Ms, unit_length=1):
        """Function to be called after the energy object has been constructed.

        *Arguments*

            S3
                Dolfin 3d VectorFunctionSpace on which M is defined

            M
                Magnetisation field (normally normalised)

            Ms
                Saturation magnetitsation (scalar, or scalar Dolfin function)

            unit_length
                unit_length of distances in mesh.

        """
        if not isinstance(Ms, (df.Function, df.Constant)):
            Ms = df.Constant(Ms)

        self.S3 = S3
        self.M = M

        # Marijan, I think we need to add mu0*Ms here for the energy.
        # When we compute the field, we need to divide by mu0 and Ms.
        # I put the mu0 * Ms in here for now. For Py, the product is approximately 1
        # anyway.
        # This might also solve the problem with the energy conservation we have
        # seen.

        
        # Dzyaloshinsky-Moriya Constant
        # change unit_length**2 to unit_length since scaling effect DMI by Delta
        self.DMIconstant = df.Constant(mu0 * self.D / unit_length ) * Ms

        self.v = df.TestFunction(S3)
        #Equation is chosen from the folowing papers
        #Yu-Onose2010, Li-Lin2011, Elhoja-Canals2002, Bode-Heide2007, Bak-Jensen1980
        #self.E = self.DMIconstant * df.inner(self.M, df.curl(self.M)) * df.dx

        #self.E = dmi_term3d_dolfin(self.M,self.DMIconstant)

        # Needed for energy density
        FS = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(FS)

        meshdim = S3.mesh().topology().dim()
        if meshdim == 1:  # 1d mesh
            NotImplementedError("Not implemented for 1d mesh yet -- should be easy though")
        elif meshdim == 2:  # 2d mesh
            E, nodal_E = \
                dmi_term2d(self.M, w, self.DMIconstant, debug=False)
        elif meshdim == 3:  # 3d mesh
            E, nodal_E = \
                dmi_term3d(self.M, w, self.DMIconstant, debug=False)

        #Muhbauer2011
        #self.E = self.E = self.DMIconstant * df.inner(self.M, df.curl(self.M)) * df.dx
        #Rossler-Bogdanov2006
        #self.E = self.DMIconstant * df.cross(self.M, df.curl(self.M)) * df.dx

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        super(DMI, self).setup(
                E_integrand=E,
                S3=S3,
                m=M,
                Ms=Ms,
                unit_length=unit_length)


class DMI_Old(EnergyBase):
    """
    Compute the DMI field.

    This is the original class, which used to be called DMI. As of today (30 June 2012),
    I have renamed the old DMI to DMI_Old, and created a new one that is called 
    DMI and derived from EnergyBase. We should move to the new DMI class to avoid
    code-duplication. Hans

    Update 26 December: The regression tests have not been ported systematically from
    the old code to new code. The idea was to have some tests simulating the same system
    with the DMI_Old and DMI class, and to check whether the answers are the same.

    The comment above still holds: we should get rid of the OLD DMI class, but I
    suggest to keep it until Marijan starts to work on this. Updating the tests
    and getting rid of this code will be a good starting point for him to work
    with finmag. (HF)

    .. math::
        
        E_{\\text{DMI}} = \\int_\\Omega D \\vec{M} \\cdot (\\nabla \\times \\vec{M})  dx
        
    *Arguments*
        S3 
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
            mesh = BoxMesh(0, m, 0, m, 0, m, n, n, n)
            
            S3  = VectorFunctionSpace(mesh, "Lagrange", 1)
            D  = 5e-3 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), V) # Initial magnetisation
           
            dmi = DMI(S3, M, D, Ms)

            # Print energy
            print dmi.compute_energy()

            # DMI field 
            H_dmi = dmi.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            dmi_np = Exchange(S3, M, D, Ms, method='box-matrix-numpy')
            H_dmi_np = dmi_np.compute_field()
            
    """
    @mtimed
    def __init__(self, D, method="box-matrix-petsc"):
        logger.debug("DMI(): method = %s" % method)

        self.D = D
        self.method = method
        self.in_jacobian = True

    @mtimed
    def setup(self, S3, m, Ms, unit_length=1):
        self.S3 = S3
        self.M = m
        self.DMIconstant = df.Constant(self.D / unit_length**2) #Dzyaloshinsky-Moriya Constant

        self.v = df.TestFunction(S3)        #Equation is chosen from the folowing papers
        #Yu-Onose2010, Li-Lin2011, Elhoja-Canals2002, Bode-Heide2007, Bak-Jensen1980
        #self.E = self.DMIconstant * df.inner(self.M, df.curl(self.M)) * df.dx

        #self.E = dmi_term3d_dolfin(self.M,self.DMIconstant)

        # Needed for energy density
        FS = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(FS)

        #mesh_shape = S3.mesh().coordinates().shape
        #meshdim = mesh_shape[1]
        meshdim = S3.mesh().topology().dim()
        if meshdim == 1: #2d mesh
            NotImplementedError("Not implemented for 1d mesh yet -- should be easy though")
        elif meshdim == 2: #2d mesh
            self.E, self.nodal_E = \
                    dmi_term2d(self.M, w, self.DMIconstant, debug=False)
        elif meshdim ==3: #3d mesh
            self.E, self.nodal_E = \
                    dmi_term3d(self.M, w, self.DMIconstant, debug=False)

        #Muhbauer2011
        #self.E = self.E = self.DMIconstant * df.inner(self.M, df.curl(self.M)) * df.dx
        #Rossler-Bogdanov2006
        #self.E = self.DMIconstant * df.cross(self.M, df.curl(self.M)) * df.dx

        self.dE_dM = df.derivative(self.E * df.dx, self.M, self.v)
        self.vol = df.assemble(df.dot(self.v, df.Constant([1,1,1]))*df.dx).array()
        self.nodal_vol = df.assemble(w*df.dx, mesh=S3.mesh()).array()

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(FS)

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
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'box-assemble',
                                    * 'box-matrix-numpy',
                                    * 'box-matrix-petsc'
                                    * 'project'""")

    @mtimed
    def compute_field(self):
        """
        Compute the DMI field.

         *Returns*
            numpy.ndarray
                The DMI field.

        """
        H = self.__compute_field()
        return H

    @mtimed
    def compute_energy(self):
        """
        Return the DMI energy.

        *Returns*
            Float
                The DMI energy.

        """
        E = df.assemble(self.E * df.dx)
        return E

    def energy_density(self):
        """
        Compute the DMI energy density,

        .. math::

            \\frac{E_\\mathrm{DMI}}{V},

        where V is the volume of each node.

        *Returns*
            numpy.ndarray
                The DMI energy density.

        """
        nodal_E = df.assemble(self.nodal_E).array()
        return nodal_E/self.nodal_vol

    def energy_density_function(self):
        """
        Compute the DMI energy density the same way as the
        function above, but return a Function to allow probing.

        *Returns*
            dolfin.Function
                The DMI energy density.

        """
        self.ED.vector()[:] = self.energy_density()
        return self.ED

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
        H_dmi_trial = df.TrialFunction(self.S3)
        self.a = df.dot(H_dmi_trial, self.v) * df.dx
        self.L = self.dE_dM
        self.H_dmi_project = df.Function(self.S3)

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
