import dolfin as df
from finmag.util.consts import mu0
from finmag.util import helpers

class Zeeman(object):
    def __init__(self, H, **kwargs):
        """
        Set the external field.

        H can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        """
        self.in_jacobian = False
        self.value = H
        self.kwargs = kwargs

    def setup(self, S3, m, Ms, unit_length=1):
        self.m = m
        self.Ms = Ms
        self.S3 = S3
        self.set_value(self.value, **self.kwargs)

    def set_value(self, value, **kwargs):
        """
        Set the value of the field.

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        """
        self.H = helpers.vector_valued_function(value, self.S3, **self.kwargs)
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H) * df.dx 

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self):
        E = df.assemble(self.E)
        return E
