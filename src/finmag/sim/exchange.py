import numpy as np
import dolfin as df

class Exchange(object):
    def __init__(self, V, M, C, Ms, method='box'):
        """
        Compute the exchange field.
        
        The constructor takes the parameters
            V a dolfin VectorFunctionSpace
            M the dolfin object representing the magnetisation
            C the exchange constant
            Ms the saturation magnetisation

        Possible methods are 'box','project'.

        At the moment, we think 'box' works (and is what is used in Magpar and Nmag).
        However, we are curious what happens if we 'properly project' the exchange field
        into the function space.
        
        """

        mu0 = 4 * np.pi * 10**-7 # Vs/(Am)
        self.exchange_factor = df.Constant(-2 * C / (mu0 * Ms))
        self.method = method

        v = df.TestFunction(V)
        E = self.exchange_factor * df.inner(df.grad(M), df.grad(M)) * df.dx
        self.dE_dM = df.derivative(E, M, v)

        if method=='box':
            self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
        elif method=='project':
            self.v = v
            self.V = V
            
            H_exch_trial = df.TrialFunction(self.V)
            self.a = df.dot(H_exch_trial, self.v) * df.dx
            self.L = self.dE_dM
            self.H_exch_project = df.Function(self.V)        

        else:
            NotImplementedError("Only 'box' and 'project' methods are implemented")

    def compute(self):
        """Assembles vector with H_exchange, and returns effective field as 
        numpy array.
        
        Question: How fast is assemble? Would it be faster to compute
        the matrix G as in (22) in Scholz etal, Computational
        Materials Science 28 (2003) 366-383, and just to carry out a
        matrix vector multiplication here? (Might be able to exploit
        that the matrix G looks the same for the x, y and z component?)
        """

        if self.method == 'box':
            box = df.assemble(self.dE_dM).array() / self.vol
            return box
        elif self.method == 'project':
            df.solve( self.a==self.L, self.H_exch_project)
            return self.H_exch_project.vector().array()
        else: 
            raise NotImplementedError

