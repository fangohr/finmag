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
        """Compute the DMI field, and return it as a numpy array

        Access to the current magnetisation is available through self.M 
        """
    #    H_dmi_tmp = df.Constant((0,0,0.8e6))        
    #    H_dmi_dolfin_field = df.project(H_dmi_tmp,self.V)
    #    return H_dmi_dolfin_field.vector().array()
        

        return df.assemble(self.dE_dM).array() / self.vol

    def compute_energy(self):
        return df.assemble(self.E)
