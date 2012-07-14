import unittest
import dolfin as df
import numpy as np


from finmag.llb.material import Material
import finmag.native.llb as native_llb
from finmag.llb.llb import LLB




class LLBSimTest(unittest.TestCase):
    def test_materials(self):
        mesh = df.UnitCube(1, 1, 1)
        mat = Material(mesh,name='FePt')
        mat.T = 10
        
        print mat.T
        print mat.alpha
        print mat.inv_chi_par
        print 'inv_chi_perp:',mat.inv_chi_perp
        print mat.A
        print mat.m_e
        print mat.Ms0
        print mat.gamma_LL
    
    def test_native_llb(self):
        m=np.array([1.,2.,3.])
        native_llb.test_numpy(m)
        print m
        
    def test_exampe(self):
        pass
        
        
        

    



            
