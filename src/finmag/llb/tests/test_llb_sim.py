import unittest
import dolfin as df
import numpy as np


from finmag.llb.material import Material
import finmag.native.llb as native_llb

def test_random():
    mu, sigma = 0, 1
    s = np.random.normal(mu, sigma, 1000000)
    print np.average(s),np.max(s),np.min(s)
    pass

class LLBSimTest(unittest.TestCase):
    
    def test_materials(self):
        mesh = df.UnitCube(1, 1, 1)
        mat = Material(mesh, name='FePt')
        mat.T = 659.9

        print mat.T
        print mat.alpha
        print mat.inv_chi_par
        print 'inv_chi_perp:', mat.inv_chi_perp
        print 'A',mat.A
        print 'm_e',mat.m_e
        print mat.Ms0
        print mat.gamma_LL
        
        
        mat.T=601
        print mat.inv_chi_par
        mat.T=600
        print mat.inv_chi_par

    def test_native_llb(self):
        m = np.array([1., 2., 3.])
        native_llb.test_numpy(m)
        print m
    
    
    def testHeun(self):
        m = np.array([1., 2., 3.])
        #integrator=native_llb.HeunStochasticIntegrator(m,m,m,1,1,1,1,1,1,1)
        #integrator.Hello()

    
    
