import logging
import dolfin as df
import numpy as np
from energy_base import EnergyBase

log = logging.getLogger("finmag")

mu0 = 4 * np.pi * 1e-7

class TimeZeeman(EnergyBase):
    def __init__(self, field_expression, t_off=None):
        self.in_jacobian = False
        self.f_expr = field_expression
        self.t_off = t_off
        self.switched_off = False

    def setup(self, S3, m, Ms, unit_length=1):
        self.m = m
        self.Ms = Ms
        self.S3 = S3

        self.H = df.interpolate(self.f_expr, self.S3)
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H) * df.dx

    def update(self, t):
        if not self.switched_off:
            if self.t_off and t > self.t_off:
                self.switch_off()
            self.f_expr.t = t
            self.H = df.interpolate(self.f_expr, self.S3)

    def switch_off(self):
        log.debug("Switching external field off.")
        self.H = df.Function(self.S3)
        self.switched_off = True

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self):
        E = df.assemble(self.E)
        return E

class DiscreteTimeZeeman(TimeZeeman):
    def __init__(self, field_expression, t_off, dt_update):
        super(DiscreteTimeZeeman, self).__init__(field_expression)
        self.dt_update = dt_update
        self.t_last_update = 0.0

    def update(self, t):
        if not self.switched_off:
            if t >= self.t_off:
                self.switch_off()
                return
                
            dt_since_last_update = t - self.t_last_update
            if dt_since_last_update >= self.dt_update:
                self.H = df.interpolate(self.f_expr, self.S3)
                log.debug("At t={}, after dt={}, update external field again.".format(
                    t, dt_since_last_update))
