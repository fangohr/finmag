import numpy as np
import dolfin as df

class Exchange(object):
    def __init__(self, V, M, C, Ms):
        """
        Compute the exchange field.
        
        The constructor takes the parameters
            V a dolfin VectorFunctionSpace
            M the dolfin object representing the magnetisation
            C the exchange constant
            Ms the saturation magnetisation
        
        """

        mu0 = 4 * np.pi * 10**-7 # Vs/(Am)
        self.exchange_factor = df.Constant(-2 * C / (mu0 * Ms))

        v = df.TestFunction(V)
        E = self.exchange_factor * df.inner(df.grad(M), df.grad(M)) * df.dx
        self.dE_dM = df.derivative(E, M, v)

        self.vol = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()

    def compute(self):
        """Assembles vector with H_exchange, and returns effective field as 
        numpy array.
        """

        return df.assemble(self.dE_dM).array() / self.vol

