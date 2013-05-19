import dolfin as df
import numpy as np
import logging
from finmag.util.consts import mu0
from finmag.util.timings import mtimed
from finmag.util import helpers
from scipy.sparse import csr_matrix


logger=logging.getLogger('finmag')

"""
This is also not a real DG method.
"""
    
class ExchangeDG(object):
    def __init__(self, C, in_jacobian = False, name='ExchangeDG'):
        self.C = C
        self.in_jacobian=in_jacobian
        self.name = name
   
    #@mtimed
    def setup(self, DG3, m, Ms, unit_length=1.0): 
        self.DG3 = DG3
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        
        mesh = DG3.mesh()
        
        DG = df.FunctionSpace(mesh, "DG", 0)
        CG3 = df.VectorFunctionSpace(mesh, "CG", 1)
    
        n = df.FacetNormal(mesh)
    
        u_dg = df.TrialFunction(DG)
        v_dg = df.TestFunction(DG)
    
        u3 = df.TrialFunction(CG3)
        v3 = df.TestFunction(CG3)
    
        a1 = u_dg * df.inner(v3, n) * df.ds - u_dg * df.div(v3) * df.dx
        self.K1 = csr_matrix(df.assemble(a1).array())
        self.L3 = df.assemble(df.dot(v3, df.Constant([1,1,1])) * df.dx).array()
    
        a2 = df.div(u3) * v_dg * df.dx
        self.K2 = csr_matrix(df.assemble(a2).array())
        self.L = df.assemble(v_dg * df.dx).array()
    
        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms**2 * self.unit_length**2)
                
        self.L = df.assemble(v_dg * df.dx).array()
        
        self.coeff1 = 1.0/self.L3
        self.coeff2 = self.exchange_factor/self.L
        
        self.H = m.vector().array()
    
    def compute_field(self):
        mm = self.m.vector().array()
        mm.shape = (3,-1)
        self.H.shape=(3,-1)

        for i in range(3):
            f = self.coeff1*(self.K1*mm[i])
            self.H[i][:] = self.coeff2 * (self.K2 * f)
        
        mm.shape = (-1,)
        self.H.shape=(-1,)
        
        return self.H
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())
    


    
    
    
    

    
