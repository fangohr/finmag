import dolfin as df
import logging
from finmag.util.timings import mtimed
from finmag.util.consts import mu0

logger = logging.getLogger('finmag')


class UniaxialAnisotropy(object):
    def __init__(self, K1, axis):
    	self.K1=K1
	self.axis = df.Constant(axis)

    @mtimed
    def setup(self, S3, M, Ms0, unit_length=1):
        self.S3 = S3
        self.M = M
        v3 = df.TestFunction(S3)

        E = -df.Constant(self.K1/Ms0**2) *((df.dot(self.axis, self.M)) ** 2) * df.dx

        # Gradient
        self.dE_dM = df.Constant(-1.0/mu0) * df.derivative(E, self.M)
        self.vol = df.assemble(df.dot(v3, df.Constant([1, 1, 1])) * df.dx).array()

        self.K = df.PETScMatrix()
        self.H = df.PETScVector()
        g_form = df.derivative(self.dE_dM, self.M)
        
        df.assemble(g_form, tensor=self.K)

    def compute_field(self):
        self.K.mult(self.M.vector(), self.H)
        return  self.H.array()/self.vol


if __name__ == "__main__":
    from dolfin import *
    m = 1e-8
    Ms = 0.8e6
    n = 5
    mesh = Box(0, m, 0, m, 0, m, n, n, n)

    S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
    C = 1.3e-11  # J/m exchange constant
    M = project(Constant((Ms, 0, 0)), S3)  # Initial magnetisation
    uniax = UniaxialAnisotropy(K1=1e11, axis=[1, 0, 0])

    uniax.setup(S3, M, Ms)

    print uniax.compute_field()

