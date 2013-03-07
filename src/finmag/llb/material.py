import logging 
import numpy as np
import dolfin as df

import finmag
import finmag.util.consts as consts
import finmag.native.llb as native_llb
from finmag.util import helpers
from finmag.util.consts import mu0


logger = logging.getLogger(name='finmag')


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
    
    def __init__(self, mesh, name='FePt',unit_length=1):
        self.mesh = mesh
        self.name = name
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
        
        self.nxyz=mesh.num_vertices()
        self._m = df.Function(self.S3)
        
        self._T = np.zeros(self.nxyz)
        self._Ms = np.zeros(3*self.nxyz)
        self._m_e = np.zeros(3*self.nxyz)
        self.inv_chi_par = np.zeros(self.nxyz)
        self.h = np.zeros(3*self.nxyz)
        self.unit_length=unit_length 
        
        self.alpha = 0.5
        self.gamma_LL = consts.gamma
        
        if self.name == 'FePt':
            self.Tc=660
            self.Ms0=1047785.4656
            self.A0=2.148042e-11
            self.K0=8.201968e6
            self.mu_a=2.99e-23
        elif self.name == 'Nickel':
            self.Tc=630
            self.Ms0=4.9e5
            self.A0=9e-12
            self.K0=0
            self.mu_a=0.61e-23
        elif self.name == 'Permalloy':
            self.Tc=870
            self.Ms0=8.6e5
            self.A0=13e-12
            self.K0=0
            #TODO: find the correct mu_a for permalloy
            self.mu_a=1e-23
        else:
            raise NotImplementedError("Only FePt and Nickel available")
                
        self.volumes = df.assemble(df.dot(df.TestFunction(self.S3),
                                      df.Constant([1, 1, 1])) * df.dx).array()
                                      
        self.real_vol = self.volumes*self.unit_length**3
        
        self.mat=native_llb.Materials(self.Ms0,self.Tc,self.A0,self.K0,self.mu_a)
        
        dg=df.FunctionSpace(mesh, "DG", 0)
        self._A_dg = df.Function(dg)
        self._m_e_dg=df.Function(dg)
        
        self.T = 0
        self.Ms=self.Ms0*self._m_e_dg.vector().array()
      
    @property  
    def me(self):
        return self._m_e[0]
        
    def compute_field(self):
        native_llb.compute_relaxation_field(self._T, self.m, self.h, self._m_e, self.inv_chi_par,self.Tc)
        return self.h
        
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, value):
        self._T[:]=helpers.scalar_valued_function(value,self.S1).vector().array()[:]
        
        self._T_dg=helpers.scalar_valued_dg_function(value,self.mesh)
        
        As = self._A_dg.vector().array()
        Ts = self._T_dg.vector().array()
        mes = self._m_e_dg.vector().array()
        
        for i in range(len(Ts)):
            As[i] = self.mat.A(Ts[i])
            mes[i] = self.mat.m_e(Ts[i])
            
        self._A_dg.vector().set_local(As)
        self._m_e_dg.vector().set_local(mes)
        
        self._m_e.shape=(3,-1)
        for i in range(len(self._T)):
            self._m_e[:,i] = self.mat.m_e(self._T[i])
            self.inv_chi_par[i] = self.mat.inv_chi_par(self._T[i])
        self._m_e.shape=(-1,)    
        
        #TODO: Trying to use spatial parameters
        self.inv_chi_perp = self.mat.inv_chi_perp(self._T[0])
    
        
    @property
    def Ms(self):
        return self._Ms
    
    @Ms.setter
    def Ms(self, value):
        self._Ms_dg=helpers.scalar_valued_dg_function(value,self.mesh)

        tmp_Ms = df.assemble(self._Ms_dg*df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx)/self.volumes
        
        self._Ms[:]=tmp_Ms[:]       
            
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
        self._m = helpers.vector_valued_function(value, self.S3, normalise=False)
        

if __name__ == "__main__":
    mesh = df.UnitCubeMesh(1, 1, 1)
    mat = Material(mesh,name='Nickel')
    mat.set_m((1,0,0))
    mat.T=3
    print mat.T
    print mat.inv_chi_par
    print mat.compute_field()
    
