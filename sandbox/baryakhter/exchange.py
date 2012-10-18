import dolfin as df
import numpy as np
import logging
from finmag.energies.energy_base import EnergyBase 
from finmag.util.consts import mu0
from finmag.util.timings import timings

logger=logging.getLogger('finmag')


class Exchange(EnergyBase):
    def __init__(self, C, chi=1,method="box-matrix-petsc"):
        super(Exchange, self).__init__(method, in_jacobian=True)     
        self.C = C
        self.chi=chi
        

    def setup(self, S3, M, Mo, unit_length=1):
        timings.start('Exchange-setup')
        
        #Mo=df.interpolate(M,S3)

        self.Mo=Mo
        #expression for the energy
        exchange_factor = df.Constant(1.0 / (Mo*unit_length) ** 2)

        self.exchange_factor = exchange_factor  

        E = exchange_factor * \
            (self.C) * df.inner(df.grad(M), df.grad(M))* df.dx  + \
               (df.inner(M,M)-Mo*Mo)**2/(8.0*self.chi*Mo*Mo)*df.dx
            
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        #nodal_E is not correct
        nodal_E = df.dot(self.exchange_factor \
                * df.inner(df.grad(M), df.grad(M)), w) * df.dx
                
        super(Exchange, self).setup(
                E=E,
                nodal_E=nodal_E,
                S3=S3,
                M=M,
                Ms=Mo,
                unit_length=unit_length)

        timings.stop('Exchange-setup')
    
  
if __name__ == "__main__":
    mesh = df.Interval(2, 0, 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
    C = 1
    expr = df.Expression(('sin(x[0])', 'cos(x[0])'))
    Ms = 1
    M = df.project(expr, S3)
    
    exch = Exchange(C)

    exch.setup(S3, M, Ms)

    print exch.compute_field()
    print exchange(mesh,M.vector().array(),2,C,Ms)
    
    

    
    



    
