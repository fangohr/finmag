import numpy as np
import dolfin as df
from energy_base import EnergyBase

mu0 = 4 * np.pi * 1e-7

class Zeeman(EnergyBase):
    def __init__(self, value, **kwargs):
        """
        Set the external field.
       
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

        """
        self.value = value
        self.kwargs = kwargs

    def setup(self, S3, m, Ms, unit_length=1):
        self.m = m
        self.Ms = Ms

        if isinstance(self.value, tuple):
            if isinstance(self.value[0], str):
                # a tuple of strings is considered to be the ingredient
                # for a dolfin expression, whereas a tuple of numbers
                # would signify a constant
                val = df.Expression(self.value, **self.kwargs)
            else:
                val = df.Constant(self.value)
            H = df.interpolate(val, S3)
        elif isinstance(self.value, (df.Constant, df.Expression)):
            H = df.interpolate(self.value, S3)
        elif isinstance(self.value, (list, np.ndarray)):
            H = df.Function(self.S3)
            H.vector()[:] = self.value
        elif hasattr(self.value, '__call__'):
            coords = np.array(zip(* S3.mesh().coordinates()))
            H = df.Function(S3)
            H.vector()[:] = self.value(coords).flatten()
        else:
            raise AttributeError

        self.H = H
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H) * df.dx 

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self):
        E = df.assemble(self.E)
        return E
