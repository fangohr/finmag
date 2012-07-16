import logging 
import numpy as np
import dolfin as df

import finmag.util.helpers as h
from finmag.native.llb import LLBFePt



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
    
    def __init__(self, mesh, name='FePt'):
        self.mesh = mesh
        self.name = name
        self.V = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
        self._m = df.Function(self.S3)
        self._T = np.zeros(mesh.num_vertices())
        self.h = self._m.vector().array()#just want to create a numpy array 
        
        if self.name == 'FePt':
            self.mat = LLBFePt()
            self.Ms0 = self.mat.M_s()
            self.Tc = self.mat.T_C()
            self.alpha = 0.5
            self.gamma_LL = 2.210173e5 # m/(As)
            
        else:
            raise NotImplementedError("Only FePt available")
        
        self.T = 0
        
    def compute_field(self):
        self.mat.compute_relaxation_field(self._T, self.m, self.h)
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
        if isinstance(value, tuple):
            if isinstance(value[0], str):
                # a tuple of strings is considered to be the ingredient
                # for a dolfin expression, whereas a tuple of numbers
                # would signify a constant
                val = df.Expression(value, **kwargs)
            else:
                val = df.Constant(value)
            new_m = df.interpolate(val, self.S3)
        elif isinstance(value, (df.Constant, df.Expression)):
            new_m = df.interpolate(value, self.S3)
        elif isinstance(value, (list, np.ndarray)):
            new_m = df.Function(self.S3)
            new_m.vector()[:] = value
        elif hasattr(value, '__call__'):
            coords = np.array(zip(* self.mesh().coordinates()))
            new_m = df.Function(self.S3)
            new_m.vector()[:] = value(coords).flatten()
        else:
            raise AttributeError
        new_m.vector()[:] = h.fnormalise(new_m.vector().array())
        self._m.vector().set_local(new_m.vector().array())

   
        

if __name__ == "__main__":
    mesh = df.UnitCube(1, 1, 1)
    mat = Material(mesh)
    mat.m = (1, 2, 4)
    print mat.M
    
