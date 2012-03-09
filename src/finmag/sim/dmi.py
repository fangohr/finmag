import numpy as np
import dolfin as df

class DMI(object):
    def __init__(self, V, M, Ms):
        """
        Compute the DMI field 
        
        The constructor takes the parameters
            V a dolfin VectorFunctionSpace
            M the dolfin object representing the magnetisation
            Ms the saturation magnetisation

        """
        print "DMI - Default method is compute_field_assemble"
        
        self.M = M
        self.V = V
        self.DMIconstant = df.Constant(13e-3) #Dzyaloshinsky-Moriya Constant
        self.v = df.TestFunction(V)
        self.E = self.DMIconstant * df.inner(self.M, df.curl(self.M)) *df.dx
        self.dE_dM = df.derivative(self.E, M, self.v)
        self.vol = df.assemble(df.dot(self.v, df.Constant([1,1,1]))*df.dx).array()
        
    def compute_field(self):
        
        return df.assemble(self.dE_dM).array() / self.vol

    def compute_energy(self):
        return df.assemble(self.E)
