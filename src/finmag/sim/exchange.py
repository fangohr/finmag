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
        self.Ms = Ms

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
        f = df.assemble(self.dE_dM).array() / (self.Ms * self.vol)
        return f

    def compute_field_project(self):
        df.solve(self.a == self.L, self.H_exch_project)
        return self.H_exch_project.vector().array()

    def compute_energy(self):
        return df.assemble(self.E)
