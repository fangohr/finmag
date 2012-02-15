from dolfin import *

class Anisotropy(object):
    def __init__(self, V, M, Ms, K1, a):

        # Local testfunction
        w = TestFunction(V)
        
        # Anisotropy energy
        E_ani = K1*(1 - (dot(a, M))**2)*dx

        # Gradient
        self.dE_dM = derivative(E_ani, M)

        # Volume
        self.vol = assemble(dot(w, Constant((1,1,1)))*dx).array()
        self.Ms  = Ms

    def compute(self):
        return assemble(self.dE_dM).array() / self.vol

