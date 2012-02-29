import numpy as np
import dolfin as df

class Exchange(object):
    """
    Compute the exchange field.

    .. math::
        
        E_{\\text{exch}} = \\int_\\Omega A (\\nabla M)^2  dx
        
    *Arguments*
        V 
            a dolfin VectorFunctionSpace object.
        M 
            the dolfin object representing the magnetisation
        C 
            the exchange constant

    Possible methods are 'box','box-assemble', 'project'.

    At the moment, we think 'box' works (and is what is used in Magpar and Nmag)
    and is fastest.

    - 'box-assemble' is a slower version that assembles the H_ex for a given M in every
      iteation.

    - 'box' precomputes a matrix g, so that H_ex = g*M
      (faster). Should be called 'box-matrix', but as this is the
      default choice 'box' might be appropriate.  [Default choice]

    - 'project': does not use the box method but 'properly project' the exchange field
      into the function space. Should explore whether this works and/or makes any difference
      (other than being slow.) Untested.


    *Example of Usage*

        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)
            
            V  = VectorFunctionSpace(mesh, "Lagrange", 1)
            C  = 1.3e-11 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), V) # Initial magnetisation

            exchange = Exchange(V, M, C)

            # Print energy
            print exchange.compute_energy()

            # Exchange field ('box' method)
            H_exch = exchange.compute_field_box()

            # Using 'project' method
            exchange_pro = Exchange(V, M, C, Ms, method='project')
            H_exch_pro = exchange_pro.compute_field_project()
            
    """

    def __init__(self, V, M, C, Ms, method='box'):
        
        mu0 = 4 * np.pi * 10**-7 # Vs/(Am)
        self.exchange_factor = df.Constant(-2 * C / (mu0 * Ms))
        self.method = method
        self.M = M #keep reference to M

        v = df.TestFunction(V)
        self.E = self.exchange_factor * df.inner(df.grad(M), df.grad(M)) * df.dx
        self.dE_dM = df.derivative(self.E, M, v)

        if method=='box-assemble':
            self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
            self.compute_field = self.compute_field_box_assemble
        elif method == 'box':
            self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
            self.compute_field = self.compute_field_box_gmatrix
            #linearise dE_dM with respect to M. As we know this is
            #linear ( should add reference to Werner Scholz paper and
            #relevant equation for g), this creates the right matrix
            #to compute dE_dM later as dE_dM=g*M.  We essentially
            #compute a Taylor series of the energy in M, and know that
            #the first two terms (dE_dM=Hex, and ddE_dMdM=g) are the
            #only finite ones as we know the expression for the
            #energy.
            g_form = df.derivative(self.dE_dM,M)

            self.g_petsc = df.PETScMatrix()
            df.assemble(g_form,tensor=self.g_petsc)
            self.g = df.assemble(g_form).array() #store matrix as numpy array
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
            NotImplementedError("Only 'box' and 'project' methods are implemented")

    def compute_field_box_assemble(self):
        """
        Assemble vector with H_exchange using the 'box' method.
        
        *Returns*
            numpy.ndarray
                The effective field.
                
        """
        return df.assemble(self.dE_dM).array() / self.vol

    def compute_field_box_gmatrix(self):
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
        
        #this is the petsc version to replace the three lines above. Works, I think, but 
        #we should allow to choose this via a new method keyword, say 'box-matrix-petsc'
        #and add tests
        self.g_petsc.mult(self.M.vector(),self.H_ex_petsc)
        return self.H_ex_petsc.array()/self.vol
        print "Check H_ex_petsc_vector"
        


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
        Return the exchange energy.

        *Returns*
            Float
                The exchange energy.

        """
        return df.assemble(self.E)
