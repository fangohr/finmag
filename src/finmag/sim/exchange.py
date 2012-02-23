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

    Possible methods are 'box','project'.

    At the moment, we think 'box' works (and is what is used in Magpar and Nmag).

    However, we are curious what happens if we 'properly project' the exchange field
    into the function space.

    *Example of Usage*

        .. code-block:: python

            >>> m    = 1e-8
            >>> n    = 5
            >>> mesh = Box(0, m, 0, m, 0, m, n, n, n)
            
            >>> V  = VectorFunctionSpace(mesh, "Lagrange", 1)
            >>> C  = 1.3e-11 # J/m exchange constant
            >>> M  = project((Constant((Ms, 0, 0)), V) # Initial magnetisation

            >>> exchange = Exchange(V, M, C)

            >>> # Print energy
            >>> print exchange.compute_energy()

            >>> # Exchange field ('box' method)
            >>> H_exch = exchange.compute_field_box()

            >>> # Using 'project' method
            >>> exchange_pro = Exchange(V, M, C, Ms, method='project')
            >>> H_exch_pro = exchange_pro.compute_field_project()
            
    """

    def __init__(self, V, M, C, Ms, method='box'):
        
        mu0 = 4 * np.pi * 10**-7 # Vs/(Am)
        self.exchange_factor = df.Constant(-2 * C / (mu0 * Ms))
        self.method = method

        v = df.TestFunction(V)
        self.E = self.exchange_factor * df.inner(df.grad(M), df.grad(M)) * df.dx
        self.dE_dM = df.derivative(self.E, M, v)

        if method=='box':
            self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
            self.compute_field = self.compute_field_box
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

    def compute_field_box(self):
        """
        Assemble vector with H_exchange using the 'box' method.
        
        *Returns*
            numpy.ndarray
                The effective field.
                
        """
        return df.assemble(self.dE_dM).array() / self.vol

    def compute_field_project(self):
        """
        Assemble vector with H_exchange using the 'project' method.
        
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
