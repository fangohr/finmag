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

With the known magnetisation m in DG space, its gradient sigma in BDM 
space can be obtained by solving the linear equation:

    A sigma = K1 m

then the exchange fields F can be approached by 

    F = K2 sigma
    
""" 
def copy_petsc_to_csc(pm):
    (m,n) = pm.size(0), pm.size(1)
    matrix = sp.lil_matrix((m,n))
    for i in range(m):
        ids, values = pm.getrow(i)
        matrix[i,ids] = values

    return matrix.tocsc()

def copy_petsc_to_csr(pm):
    (m,n) = pm.size(0), pm.size(1)
    matrix = sp.lil_matrix((m,n))
    for i in range(m):
        ids, values = pm.getrow(i)
        matrix[i,ids] = values
    
    return matrix.tocsr()


class ExchangeDG(object):
    def __init__(self, C, in_jacobian = True, name='ExchangeDG'):
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
        BDM = df.FunctionSpace(mesh, "RT", 1)
        
        #deal with three components simultaneously, each represents a vector
       
        sigma = df.TrialFunction(BDM)
        tau = df.TestFunction(BDM)
        
        
        u = df.TrialFunction(DG)
        v = df.TestFunction(DG)
        
        # what we need is A x = K1 m 
        #a0 = (df.dot(sigma0, tau0) + df.dot(sigma1, tau1) + df.dot(sigma2, tau2)) * df.dx
        a0 = df.dot(sigma, tau) * df.dx
        self.A = df.assemble(a0)
         
        a1 = - (df.div(tau) * u) * df.dx
        self.K1 = df.assemble(a1)
        
        C = sp.lil_matrix(self.K1.array())
        self.KK1 = C.tocsr()
    
        
        def boundary(x, on_boundary):
            return on_boundary
        
        # actually, we need to apply the Neumann boundary conditions.
        
        zero = df.Constant((0,0,0))
        self.bc = df.DirichletBC(BDM, zero, boundary)
        #print 'before',self.A.array()
        
        self.bc.apply(self.A)
        
        #print 'after',self.A.array()
        
        #AA = sp.lil_matrix(self.A.array())
        AA = copy_petsc_to_csc(self.A)
        
        self.solver = sp.linalg.factorized(AA.tocsc())
        
        #LU = sp.linalg.spilu(AA)
        #self.solver = LU.solve
        
        a2 = (df.div(sigma) * v) * df.dx
        self.K2 = df.assemble(a2)
        self.L = df.assemble(v * df.dx).array()
    
        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms * self.unit_length**2)

        self.coeff = self.exchange_factor/self.L
        
        self.K2 = copy_petsc_to_csr(self.K2)
        
        # b = K m
        self.b = df.PETScVector()        
        # the vector in BDM space
        self.sigma_v = df.PETScVector()
        
        # to store the exchange fields
        #self.H = df.PETScVector()
        self.H_eff = m.vector().array()
        
        self.m_x = df.PETScVector(self.m.vector().size()/3)
        
    
    @mtimed
    def compute_field(self):
        
        mm = self.m.vector().array()
        mm.shape = (3,-1)
        self.H_eff.shape=(3,-1)
        
        for i in range(3):
            self.m_x.set_local(mm[i])
            self.K1.mult(self.m_x, self.b)
            self.bc.apply(self.b)      
            
            H = self.solver(self.b.array())
            #df.solve(self.A, self.sigma_v, self.b)
            
            self.H_eff[i][:] = (self.K2*H)*self.coeff
        
        mm.shape = (-1,)
        self.H_eff.shape=(-1,)
        
        return self.H_eff
        
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())
    


class ExchangeDG2(object):
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
        BDM = df.FunctionSpace(mesh, "BDM", 1)
    
        n = df.FacetNormal(mesh)
    
        u_dg = df.TrialFunction(DG)
        v_dg = df.TestFunction(DG)
    
        u3 = df.TrialFunction(BDM)
        v3 = df.TestFunction(BDM)
    
        #a1 = u_dg * df.inner(v3, n) * df.ds - u_dg * df.div(v3) * df.dx
        a1 = u_dg * df.inner(v3, n) * df.ds - u_dg * df.div(v3) * df.dx
        self.K1 = sp.csr_matrix(df.assemble(a1).array())
        
        f_ones = df.Function(BDM)
        f_ones.vector()[:] = 1
        self.L3 = df.assemble(df.dot(f_ones, v3) * df.dx).array()
        print 'YY'*50, self.L3
    
        a2 = df.div(u3) * v_dg * df.dx
        self.K2 = sp.csr_matrix(df.assemble(a2).array())
        self.L = df.assemble(v_dg * df.dx).array()
    
        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms * self.unit_length**2)
        
        # coeff1 should multiply something, I have no idea so far ....
        self.coeff1 = -1/self.L3
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


"""
Compute the exchange field in DG0 space with the help of BDM1 space.
""" 
class ExchangeDG_bak(object):
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
 
