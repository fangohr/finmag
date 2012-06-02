import logging
import dolfin as df
import numpy as np
from energy_base import EnergyBase

log = logging.getLogger("finmag")

mu0 = 4 * np.pi * 1e-7

class TimeZeeman(EnergyBase):
    def __init__(self, expression, update_times):
        self.in_jacobian = False

        self.expression = expression
        self.update_times = update_times
        self.current_t_index = 0
        self.switched_off = False

    def setup(self, S3, m, Ms, unit_length=1):
        self.m = m
        self.Ms = Ms
        self.S3 = S3

        self.H = df.interpolate(self.expression, self.S3)
        self.E = - mu0 * self.Ms * df.dot(self.m, self.H) * df.dx

    def update(self, t):
        current_t = self.update_times[self.current_t_index]
        if t >= current_t:
            self.expression.t = t
            self.H = df.interpolate(self.expression, self.S3)
            log.debug("At t={}, passed update value {} for H_ext(t). {} points remaining.".format(
                t, current_t, len(self.update_times)- (self.current_t_index + 1)))
            self.current_t_index += 1
            if t > self.update_times[-1] or len(self.update_times) == 0:
                print "Haha."
                print self.update_times[-1]
                print len(self.update_times)
                self.switch_off()

    def switch_off(self):
        log.debug("Switching off field.")
        print "Switching off field."
        """
        Careful: Not reversible.

        """
        self.H = df.Function(self.S3)

        def new_update(self, t):
            pass
        self.update = new_update
        self.switched_off = True

    def compute_field(self):
        return self.H.vector().array()

    def compute_energy(self):
        E = df.assemble(self.E)
        return E

        


