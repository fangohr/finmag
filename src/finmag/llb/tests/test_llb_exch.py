import unittest
import dolfin as df
import numpy as np


from finmag.llb.material import Material

from finmag.energies.exchange import Exchange 
import finmag.llb.exchange as llb



class LLBTestExch(unittest.TestCase):
    def test_exch(self):
        mesh = df.Interval(10, 0, 5)
        S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
        C = 1
        expr = df.Expression(('sin(x[0])', 'cos(x[0])'))
        Ms = 1
        M = df.project(expr, S3)
    
        exch = Exchange(C)
        exch.setup(S3, M, Ms)
        H1=exch.compute_field()

        llb_exch=llb.Exchange(C)
        llb_exch.setup(S3, M, Ms,1)
        H2=llb_exch.compute_field()
        
        assert(np.max(H2-H1)==0)
        
        
        
        
        
        
        

    



            
