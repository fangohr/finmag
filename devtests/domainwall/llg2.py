import dolfin as df
import instant
import os
import numpy as np
import finmag.sim.helpers as h
from finmag.sim.exchange import Exchange
from finmag.sim.anisotropy import Anisotropy
from finmag.sim.dmi import DMI
from finmag.sim.llg import LLG

class LLG2(LLG):
    def set_m0(self, value, **kwargs):
        """
        Set the initial magnetisation (scaled automatically).
        
        You can either provide a dolfin.Constant or a dolfin.Expression
        directly, or the ingredients for either, i.e. a tuple of numbers
        or a tuple of strings (with keyword arguments if needed), or provide
        the nodal values directly as a numpy array.
        
        """
        if isinstance(value, tuple):
            if isinstance(value[0], str):
                # a tuple of strings is considered to be the ingredient
                # for a dolfin expression, whereas a tuple of numbers
                # would signify a constant
                val = df.Expression(value, **kwargs)
            else:
                val = df.Constant(value)
            self._m0 = df.interpolate(val, self.V)
        elif isinstance(value, df.Function):
            self._m0 = value
        elif isinstance(value, (df.Constant, df.Expression)):
            self._m0 = df.interpolate(value, self.V)
        elif isinstance(value, (list, np.ndarray)):
            self._m0 = df.Function(self.V)
            self._m0.vector()[:] = value
        else:
            raise AttributeError
        self._m0.vector()[:] = h.fnormalise(self._m0.vector().array())
        self._m = self._m0


    def set_default_values(self):
        self.alpha = 0.5
        self.gamma =  2.210173e5 # m/(As)
        #source for gamma:  OOMMF manual, and in Werner Scholz thesis, 
        #after (3.7), llg_gamma_G = m/(As).
        self.c = 1e11 # 1/s numerical scaling correction
                      # 0.1e12 1/s is the value used by default in nmag 0.2
        self.C = 1.3e-11 # J/m exchange constant
        self.Ms = 8.6e5 # A/m saturation magnetisation
        self.t = 0 # s
        self.H_app = (0, 0, 0)
        self.H_dmi = (0, 0, 0) #DMI for Skyrmions
        self.pins = [] # nodes where the magnetisation gets pinned
        self.K = 520e3 # J/m3 anisotropy constant
        self.a = df.Constant((0,0,1)) # easy axis
    
    def update_H_eff(self):
        self.H_eff = self.H_app + self.H_ex + self.H_ani
        if self.use_dmi:
            self.H_eff += self.H_dmi

    def solve(self):
        if self.exchange_flag:
            self.H_ex = self.exchange.compute_field()
        if self.anisotropy_flag:
            self.H_ani = self.anisotropy.compute_field()
        if self.use_dmi:
            self.H_dmi = self.dmi.compute_field()
        self.update_H_eff()

        status, dMdt = self._solve(self.alpha, self.gamma, self.c,
            self.m, self.H_eff, self.m.shape[0], self.pins)
        if status == 0:
            return dMdt
        raise Exception("An error was encountered in the C-code; status=%d" % status)
        return None

    def setup(self, exchange_flag=True, anisotropy_flag=False, use_dmi=False, exchange_method="box-matrix-petsc"):
        self.exchange_flag = exchange_flag
        self.anisotropy_flag = anisotropy_flag

        if exchange_flag:
            self.exchange = Exchange(self.V, self._m, self.C, self.Ms, method=exchange_method)
        else:
            zero = df.Constant((0, 0, 0))
            self.H_ex = df.interpolate(zero, self.V).vector().array()
        
        if anisotropy_flag:
            self.anisotropy = Anisotropy(self.V, self._m, self.K, self.a, method=exchange_method)
        else:
            zero = df.Constant((0, 0, 0))
            self.H_ani = df.interpolate(zero, self.V).vector().array()

        self.use_dmi = use_dmi
        if use_dmi:
            self.dmi = DMI(self.V, self._m, self.Ms)

