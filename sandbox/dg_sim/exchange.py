import dolfin as df
import numpy as np
import logging
from finmag.util.consts import mu0
from finmag.util.timings import mtimed
from finmag.util import helpers

logger=logging.getLogger('finmag')


    
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
        
        n = df.FacetNormal(mesh)
        h = df.CellSize(mesh)
        h_avg = (h('+') + h('-'))/2
        
        u = df.TrialFunction(DG3)
        v = df.TestFunction(DG3)
        
        u0 = df.dot(u, df.Constant([1, 0, 0]))
        v0 = df.dot(v, df.Constant([1, 0, 0]))
        u1 = df.dot(u, df.Constant([0, 1, 0]))
        v1 = df.dot(v, df.Constant([0, 1, 0]))
        u2 = df.dot(u, df.Constant([0, 0, 1]))
        v2 = df.dot(v, df.Constant([0, 0, 1]))
        v3 = df.dot(v, df.Constant([1, 1, 1]))
        
    
        a = 1.0/h_avg*df.dot(df.jump(v0, n), df.jump(u0, n))*df.dS \
            + 1.0/h_avg*df.dot(df.jump(v1, n), df.jump(u1, n))*df.dS \
            + 1.0/h_avg*df.dot(df.jump(v2, n), df.jump(u2, n))*df.dS 
        

        self.mu0 = mu0
        self.exchange_factor = 2.0 * self.C / (self.mu0 * Ms**2 * self.unit_length**2)

        self.K = df.PETScMatrix()
        df.assemble(a, tensor=self.K)
        self.H = df.PETScVector()
        
        self.vol = df.assemble(v3 * df.dx).array()
        
        self.coeff = -self.exchange_factor/(self.vol)
        #self.K = df.assemble(a).array()

    
    def compute_field(self):
        
        self.K.mult(self.m.vector(), self.H)
        
        return  self.coeff*self.H.array()
    
    def average_field(self):
        """
        Compute the average field.
        """
        return helpers.average_field(self.compute_field())
  
if __name__ == "__main__":
    mesh = df.IntervalMesh(5, 0, 2*np.pi)
    S3 = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
    C = 1
    expr = df.Expression(('0', '4*cos(x[0])','4.0*sin(x[0])'))
    Ms = 2
    m = df.interpolate(expr, S3)
        
    exch = ExchangeDG(C)
    exch.setup(S3, m, Ms)
    f = exch.compute_field()
    f.shape = (3,-1)
    print 'field\n--------------',f
    
    field=df.Function(S3)
    field.vector().set_local(exch.compute_field())
    
    # the export field are wrong (still using CG spave)
    file = df.File('field.pvd')
    file << field
    
    file = df.File('m.pvd')
    file << m
    
    df.plot(m)
    df.plot(field)
    df.interactive()

    
