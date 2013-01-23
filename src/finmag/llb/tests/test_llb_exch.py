import unittest
import dolfin as df
import numpy as np


from finmag.energies.exchange import Exchange
import finmag.llb.exchange as llb

#from IPython import embed

class LLBTestExch(unittest.TestCase):
    def test_exch(self):
        """Test case that compares the LLB exchange
        with the normal exchange class for a system with
        preserved magnitude.

        It does this for a two-dimensional magnetisation
        vector.

        There was a change on 11 Sept 2012 in the repository
        that required to change the tolerance in the assert
        statement below from 0 to 2e-8. It was probably 
        coincidence that the test passed with deviation 0.0 so
        far.
        """
        mesh = df.IntervalMesh(10, 0, 5)
        S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
        C = 1
        expr = df.Expression(('sin(x[0])', 'cos(x[0])'))
        Ms = 1
        M = df.project(expr, S3)
        exch = Exchange(C)
        exch.setup(S3, M, Ms)
        H1 = exch.compute_field()

        llb_exch = llb.Exchange(C)
        llb_exch.setup(S3, M, Ms, 1)
        H2 = llb_exch.compute_field()
        print("max(H1-H2)=%g" % np.max(H2 - H1))
        assert(np.max(H2 - H1) <  2e-8)
        #embed()

