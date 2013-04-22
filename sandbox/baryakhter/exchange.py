import dolfin as df
import numpy as np
import logging
from finmag.util.consts import mu0
from finmag.util.timings import mtimed
from finmag.native import llg as native_llg

logger=logging.getLogger('finmag')

    
class Exchange(object):
    def __init__(self, C, chi=1, in_jacobian=True):
        self.C = C
        self.chi = chi
        self.in_jacobian=in_jacobian
   
    @mtimed
    def setup(self, S3, M, M0, unit_length=1.0): 
        self.S3 = S3
        self.M = M
        self.M0 = M0
        self.unit_length = unit_length
        self.Ms2=np.array(self.M.vector().array())

        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * M0**2 * self.unit_length**2)

        u3 = df.TrialFunction(S3)
        v3 = df.TestFunction(S3)
        self.K = df.PETScMatrix()
        df.assemble(df.inner(df.grad(u3),df.grad(v3))*df.dx, tensor=self.K)
        self.H = df.PETScVector()
        
        self.vol = df.assemble(df.dot(v3, df.Constant([1, 1, 1])) * df.dx).array()
        
        self.coeff1=-self.exchange_factor/(self.vol)
        self.coeff2=-0.5/(self.chi*M0**2)
    
    def compute_field(self):
        self.K.mult(self.M.vector(), self.H)
        v=self.M.vector().array()
        
        native_llg.baryakhtar_helper_M2(v,self.Ms2)
        
        relax = self.coeff2*(self.Ms2-self.M0**2)*v
        
        return  self.coeff1*self.H.array()+relax
    
  
if __name__ == "__main__":
    mesh = df.Interval(3, 0, 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    C = 1
    expr = df.Expression(('4.0*sin(x[0])', '4*cos(x[0])','0'))
    Ms = 2
    M = df.interpolate(expr, S3)
        
    exch2 = Exchange(C)
    exch2.setup(S3, M, Ms)
    
    print 'ex2',exch2.compute_field()
    
    from finmag.energies.exchange import Exchange
    exch3 = Exchange(C)
    exch3.setup(S3, M, Ms)

    print 'ex3',exch3.compute_field()
    
