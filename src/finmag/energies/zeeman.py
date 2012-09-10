import numpy as np
import dolfin as df
from finmag.util.consts import mu0
from energy_base import AbstractEnergy

class Zeeman(AbstractEnergy):
    def __init__(self, H, **kwargs):
        """
        Set the external field.

        H can be any of the following:

         - dolfin.Constant representing a 3-vector

         - 3-tuple of numbers, which will get cast to a dolfin.Constant

         - 3-tuple of strings (with keyword arguments if needed),
           which will get cast to a dolfin.Expression where any variables in
           the expression are substituted with the values taken from 'kwargs'

         - numpy.ndarray of nodal values of the shape (3*n,), where n
           is the number of nodes

         - function (any callable object will do) which accepts the
           coordinates of the mesh as a numpy.ndarray of shape (3, n)
           and returns the field H in this form as well

        """
        self.in_jacobian = False
        self.value = H
        self.kwargs = kwargs

    def setup(self, S3, m, Ms, unit_length=1):
        self.m = m
        self.Ms = Ms
        self.S3 = S3
        self.set_value(self.value)

    def set_value(self, value):
        """
        Set the value of the field. The argument `value` can have any
        of the forms accepted by the Zeeman.__init__() function (see
        its docstring for details).
        """
        if isinstance(value, tuple):
            if isinstance(value[0], str):
                # a tuple of strings is considered to be the ingredient
                # for a dolfin expression, whereas a tuple of numbers
                # would signify a constant
                val = df.Expression(value, **self.kwargs)
            else:
                val = df.Constant(value)
            H = df.interpolate(val, self.S3)
        elif isinstance(value, (df.Constant, df.Expression)):
            H = df.interpolate(value, self.S3)
        elif isinstance(value, (list, np.ndarray)):
            if len(value) == 3:
                tmp_value = np.empty((self.S3.mesh().num_vertices(), 3))
                tmp_value[:] = value
                value = tmp_value.reshape((-1,))
            H = df.Function(self.S3)
            H.vector()[:] = value
        elif hasattr(value, '__call__'):
            coords = np.array(zip(* self.S3.mesh().coordinates()))
            H = df.Function(self.S3)
            H.vector()[:] = value(coords).flatten()
        else:
            raise AttributeError

        self.H = H
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H) * df.dx 

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self):
        E = df.assemble(self.E)
        return E
