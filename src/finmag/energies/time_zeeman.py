import dolfin as df
import numpy as np
from energy_base import EnergyBase

mu0 = 4 * np.pi * 1e-7

class TimeZeeman(EnergyBase):
    def __init__(self, expression):
        self.in_jacobian = False
        self.expression = expression

    def setup(self, S3, m, Ms, unit_length=1):
        self.m = m
        self.Ms = Ms
        self.S3 = S3

        self.H = df.interpolate(self.expression, self.S3)
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H) * df.dx

    def update(self, t):
        self.expression.t = t
        self.H = df.interpolate(self.expression, self.S3)

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self):
        E = df.assemble(self.E)
        return E

        


