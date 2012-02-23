import numpy as np
import dolfin as df

#This file needs updating for DMI -- based on exchange.py
class DMI(object):
    def __init__(self, V, M, Ms):
        """
        Compute the DMI field 
        
        The constructor takes the parameters
            V a dolfin VectorFunctionSpace
            M the dolfin object representing the magnetisation
            Ms the saturation magnetisation

        """

        #v = df.TestFunction(V)

        self.M = M   
        self.V = V

    def compute_field(self):
        """Compute the DMI field, and return it as a numpy array

        Access to the current magnetisation is available through self.M 
        """
        H_dmi_tmp = df.Constant((0,0,0.8e6))        
        H_dmi_dolfin_field = df.project(H_dmi_tmp,self.V)
        return H_dmi_dolfin_field.vector().array()
        

    def compute_energy(self):
        """Compute the DMI energy, and return as a number """
        raise NotImplementedError("Compute_energy not implemented yet for DMI term")

