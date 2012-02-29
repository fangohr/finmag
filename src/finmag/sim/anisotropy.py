import numpy as np
import dolfin as df

class Anisotropy(object):
    """
    Compute the anisotropy field.

    The magnetocrystalline anisotropy energy for uniaxial
    anisotropy is given by

    .. math::

        E_{\\text{ani}} = \\int_\\Omega \\sum_j K(1 - 
        (\\vec a \\cdot \\vec M_j)^2) dx,

    where :math:`K` is the anisotropy constant, 
    :math:`\\vec a` the easy axis and :math:`\\sum_i \\vec M_i` 
    the discrete approximation of the magnetic polarization.

    *Arguments*
        V
            a dolfin VectorFunctionSpace object.
        M
            the dolfin object representing the magnetisation
        K
            the anisotropy constant (int or float)
        a
            the easy axis (use dolfin.Constant)
        
    *Example of Usage*
        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)

            V = VectorFunctionSpace(mesh, 'Lagrange', 1)
            K = 520e3 # For Co (J/m3)

            a = Constant((0, 0, 1)) # Easy axis in z-direction
            M = project(Constant((1, 0, 0)), V) # Initial magnetisation

            anisotropy = Anisotropy(V, M, K, a)

            # Print energy
            print anisotropy.compute_energy()

            # Anisotropy field
            H_ani = anisotropy.compute_field()
            
        """   
    
    def __init__(self, V, M, K, a, method=None):
        if method == None:
            method = 'box-matix-petsc'

        # Local testfunction
        v = df.TestFunction(V)
        
        # Convert K1 to dolfin.Constant
        K = df.Constant((K))
        
        # Anisotropy energy
        self.E = K*(1 - (df.dot(a, M))**2)*df.dx

        # Gradient
        self.dE_dM = df.derivative(self.E, M)

        # Store for later
        self.V = V
        self.M = M


        if method=='box-assemble':
            self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
            self.compute_field = self.compute_field_box_assemble
        
        elif method == 'box-matrix-numpy':
            self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
            self.compute_field = self.compute_field_box_matrix_numpy
            
            #linearise dE_dM with respect to M. As we know this is
            #linear ( should add reference to Werner Scholz paper and
            #relevant equation for g), this creates the right matrix
            #to compute dE_dM later as dE_dM=g*M.  We essentially
            #compute a Taylor series of the energy in M, and know that
            #the first two terms (dE_dM=Hex, and ddE_dMdM=g) are the
            #only finite ones as we know the expression for the
            #energy.

            g_form = df.derivative(self.dE_dM, M)
            self.g = df.assemble(g_form).array() #store matrix as numpy array
        
        elif method == 'box-matrix-petsc':
            #petsc version of the scheme above.
            self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
            self.compute_field = self.compute_field_box_matrix_petsc
            g_form = df.derivative(self.dE_dM,M)
            self.g_petsc = df.PETScMatrix()
            df.assemble(g_form,tensor=self.g_petsc)
            #self.g = df.assemble(g_form).array() #store matrix as numpy array
            self.H_ex_petsc = df.PETScVector()


        elif method=='project':
            self.v = v
            self.V = V
            H_exch_trial = df.TrialFunction(self.V)
            self.a = df.dot(H_exch_trial, self.v) * df.dx
            self.L = self.dE_dM
            self.H_exch_project = df.Function(self.V)        
            #Note that we could make this 'project' method faster by computing the matrices
            #that represent a and L, and only to solve the matrix system in 'compute'().
            #IF this method is actually useful, we can do that. HF 16 Feb 2012
            self.compute_field = self.compute_field_project
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'box-assemble', 
                                    * 'box-matrix-numpy',
                                    * 'box-matrix-petsc'  
                                    * 'project'""")

    def compute_field_box_assemble(self):
        """
        Assemble vector with H_exchange using the 'box' method.
        
        *Returns*
            numpy.ndarray
                The effective field.
                
        """
        return df.assemble(self.dE_dM).array() / self.vol

    def compute_field_box_matrix_numpy(self):
        """
        Assemble vector with H_exchange using the 'box' method
        and the pre-computed matrix g.

        *Returns*
            numpy.ndarray
                The effective field.
                
        """
        
        Mvec = self.M.vector().array()
        H_ex = np.dot(self.g,Mvec)
        return H_ex/self.vol

    def compute_field_box_matrix_petsc(self):
        """
        PETSc version of the function above. Takes advantage of the 
        sparsity of g.
        
        *Returns*
            numpy.ndarray
                The effective field

        """
        
        print "Check H_ex_petsc_vector"
        self.g_petsc.mult(self.M.vector(),self.H_ex_petsc)
        return self.H_ex_petsc.array()/self.vol
                

    def compute_field_project(self):
        """
        Assemble vector with H_exchange using the 'project' method.

        This may not work.
        
        *Returns*
            numpy.ndarray
                The effective field.
                
        """
        df.solve(self.a == self.L, self.H_exch_project)
        return self.H_exch_project.vector().array()


    def compute_energy(self):
        """
        Return the anisotropy energy.

        *Returns*
            Float
                The anisotropy energy.

        """
        V = self.V
        return df.assemble(self.E_ani, mesh=V.mesh())

