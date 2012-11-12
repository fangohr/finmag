import logging 
import numpy as np
import dolfin as df

import finmag.util.consts as consts
import finmag.native.llb as native_llb
from finmag.native.llb import LLBFePt
from finmag.util import helpers
from finmag.util.consts import mu0

from scipy.optimize import fsolve

logger = logging.getLogger(name='finmag')


class Nickel(object):
    def __init__(self):
        self.S=1.0/2
        self.Tc=630
        self.T=0
        self.M0=4.8e5
        self.M=self.M0
        self.A0=9e-12
        self.A_coeff=3.90625e-23 
        self.xi_par_coeff=0.0006017068615252775  # M/(k_B T_c)*(S+1)/(3S)* a^3/ c 
        self.coth=lambda x:np.cosh(x)/np.sinh(x)
        
    def Bs(self,x):
        t1=(2*self.S+1)/(2*self.S)
        t2=1/(2*self.S)
        if x<1e-3:
            return 0
        
        return t1*self.coth(t1*x)-t2*self.coth(t2*x)
    
    def Bsp(self,x):
        if x<1e-3:
            x=1e-3
        elif x>1e3:
            return 0
        
        t=(2*self.S+1)/(2*self.S)
        t1=np.sinh(t*x)
        t2=np.cosh(2*t*x)
        return -4*t*t*t1*t1/(1-t2)**2
    
    def xi_par(self,T):
        T_bak=T
        if T<10:
            T=10
        elif T>=self.Tc:
            return self.xi_par_coeff
        
        beta = self.Tc/T
        t1=beta*self.m_e(T)
        self.T=T_bak
        
        t2=beta*self.Bsp(t1)
        #need to check here
        res=-self.xi_par_coeff*t2/(1.0-t2)
        
        if res<1e-12:
            return 1e-12
        return res
        

    def Bsm(self,m,T,Tc):
        
        if T<1e-3:
            T=1e-3
        
        x=3.0*self.S/(self.S+1)*m*Tc/T
        
        return m-self.Bs(x)

        
    def m_e(self,T):
        if self.T==T:
            return self.M/self.M0
        
        self.T=T
        
        if T<1:
            return 1.0
        
        if T>=self.Tc:
            self.M=0
        else:
            m=fsolve(self.Bsm,1,args=(T,self.Tc))
            self.M=m[0]*self.M0
        
        return self.M/self.M0
            
    def A(self,T):
        if self.T==T:
            pass
        else:
            self.m_e(T)
        return self.A_coeff*self.M**2
            
        
    
    def inv_chi_perp(self,T):
        return 0
    
    def inv_chi_par(self,T):
        return 1.0/self.xi_par(T)/mu0
    
    

class Material(object):
    """
    The aim to define this class is to collect materials properties in one class, such as 
    the common parameters Ms, A, and K since these properties may have different response
    to temperature T. Another reason is that saturation magnetisation Ms should be 
    defined in cells in the framework of FEM but for some reasons it's convenience to use 
    the related definition in nodes for dynamics, which will cause some confusion if put them
    in one class. 
    
    Despite the traditional definition that the magnetisation M(r) are separated by the unit
    magnetisation m(r) and Ms which stored in nodes and cells respectively, we just focus on 
    magnetisation M(r) and pass it into other classes such as Exchange, Anisotropy and Demag.
    Therefore, Ms in this class in fact is mainly used for users to input.
    
    Certainly, another way to deal with such confusion is to define different class for 
    different scenarios, for example, if the simulation just focus on one material and at
    temperature zero we can define a class have constant Ms.
    
    
    We will adapt this class to the situation that LLB case.    
    
    """
    
    def __init__(self, mesh, name='FePt'):
        self.mesh = mesh
        self.name = name
        self.V = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1,dim=3)
        self._m = df.Function(self.S3)
        self._T = np.zeros(mesh.num_vertices())
        self.h = self._m.vector().array()#just want to create a numpy array 
        
        if self.name == 'FePt':
            self.mat = LLBFePt()
            self.Ms0 = self.mat.M_s()
            self.Tc = self.mat.T_C()
            self.alpha = 0.5
            self.gamma_LL = consts.gamma
        elif self.name == 'Nickel':
            self.mat = Nickel()
            self.Ms0 = self.mat.M0
            self.Tc = self.mat.Tc
            self.alpha = 0.5
            
        else:
            raise NotImplementedError("Only FePt and Nickel available")
        
        self.T = 0
        
    def compute_field(self):
        #self.mat.compute_relaxation_field(self._T, self.m, self.h)
        native_llb.compute_relaxation_field(self._T, self.m, self.h,
                                            self.Tc,self.m_e,
                                            self.inv_chi_par)
        print 'fields',self.inv_chi_par,self.h
        return self.h

    
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, value):
        if isinstance(value, (df.Constant, np.ndarray)):
            assert(value.shape==self._T.shape)
            self._T[:]=value[:]
        else:
            self._T[:]=value
        #TODO: Trying to use spatial parameters
        self.A = self.mat.A(value)
        self.m_e = self.mat.m_e(value)
        self.inv_chi_perp = self.mat.inv_chi_perp(value)
        self.inv_chi_par = self.mat.inv_chi_par(value)           
            
    @property
    def m(self):
        """
        not too good since this will return a copy
        try to solve this later
        """
        return self._m.vector().array()
    
    
    def set_m(self, value, **kwargs):
        """
        Set the magnetisation (scaled automatically).
       
        There are several ways to use this function. Either you provide
        a 3-tuple of numbers, which will get cast to a dolfin.Constant, or
        a dolfin.Constant directly.
        Then a 3-tuple of strings (with keyword arguments if needed) that will
        get cast to a dolfin.Expression, or directly a dolfin.Expression.
        You can provide a numpy.ndarray of nodal values of shape (3*n,),
        where n is the number of nodes.
        Finally, you can pass a function (any callable object will do) which
        accepts the coordinates of the mesh as a numpy.ndarray of
        shape (3, n) and returns the magnetisation like that as well.

        You can call this method anytime during the simulation. However, when
        providing a numpy array during time integration, the use of
        the attribute m instead of this method is advised for performance
        reasons and because the attribute m doesn't normalise the vector.

        """
        self._m = helpers.vector_valued_function(value, self.S3, normalise=True)
        

if __name__ == "__main__":
    mesh = df.UnitCube(1, 1, 1)
    mat = Material(mesh)
    mat.set_m((1,0,0))
    mat.T=3
    print mat.T
    print mat.inv_chi_par
    print mat.compute_field()
    
