import dolfin as df
import numpy as np
import logging
from finmag.util.consts import mu0
from finmag.util.timings import mtimed
from finmag.util import helpers
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import linsolve

logger=logging.getLogger('finmag')

"""
Compute the exchange field in DG0 space with the help of BDM1 space.
""" 
class ExchangeDG(object):
    def __init__(self, C, in_jacobian = False, name='ExchangeDG'):
        self.C = C
        self.in_jacobian=in_jacobian
        self.name = name
   
    @mtimed
    def setup(self, DG3, m, Ms, unit_length=1.0): 
        self.DG3 = DG3
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        
        mesh = DG3.mesh()
        self.mesh = mesh
        
        DG = df.FunctionSpace(mesh, "DG", 0)
        BDM = df.FunctionSpace(mesh, "BDM", 1)
        
        #deal with three components simultaneously, each represents a vector
        W1 = df.MixedFunctionSpace([BDM, BDM, BDM])
        (sigma0,sigma1,sigma2) = df.TrialFunctions(W1)
        (tau0,tau1,tau2) = df.TestFunctions(W1)
        
        W2 = df.MixedFunctionSpace([DG, DG, DG])
        (u0,u1,u2) = df.TrialFunctions(W2)
        (v0,v1,v2) = df.TestFunction(W2)
        
        # what we need is A x = K1 m 
        a0 = (df.dot(sigma0, tau0) + df.dot(sigma1, tau1) + df.dot(sigma2, tau2)) * df.dx
        self.A = df.assemble(a0)
         
        a1 = - (df.div(tau0) * u0 + df.div(tau1) * u1 + df.div(tau2) * u2 ) * df.dx
        self.K1 = df.assemble(a1)
    
        
        def boundary(x, on_boundary):
            return on_boundary
        
        # actually, we need to apply the Neumann boundary conditions.
        # we need a tensor here
        zero = df.Constant((0,0,0,0,0,0,0,0,0))
        self.bc = df.DirichletBC(W1, zero, boundary)
        self.bc.apply(self.A)
        
        #tmp =  self.A.array()
        #self.A = sp.lil_matrix(tmp)
        #self.A = self.A.tocsr()
        
        a2 = (df.div(sigma0) * v0 + df.div(sigma1) * v1 + df.div(sigma2) * v2) * df.dx
        self.K2 = df.assemble(a2)
        self.L = df.assemble((v0 + v1 + v2) * df.dx).array()
    
        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms * self.unit_length**2)

        self.coeff = self.exchange_factor/self.L
        
        # b = K m
        self.b = df.PETScVector()
        
        # the vector in BDM space
        self.sigma_v = df.PETScVector(self.K2.size(1))
        
        # to store the exchange fields
        self.H = df.PETScVector()
    
    @mtimed
    def compute_field(self):
        
        # b = K2 * m 
        self.K1.mult(self.m.vector(), self.b)
        
        self.bc.apply(self.b)
        
        df.solve(self.A, self.sigma_v, self.b)
        #phi = linsolve.spsolve(self.A,self.b.array())
        #self.sigma_v.set_local(phi)
        
        self.K2.mult(self.sigma_v, self.H)

        return self.H.array()*self.coeff
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())
    

    
    
    
    

    
