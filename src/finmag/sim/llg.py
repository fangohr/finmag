import dolfin as df
import instant
import os
import numpy as np
import finmag.sim.helpers as h
from finmag.sim.exchange import Exchange
from finmag.sim.dmi import DMI

class LLG(object):

    def __init__(self, mesh, order=1):
        self.mesh = mesh
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', order, dim=3)
        self.Volume = df.assemble(df.Constant(1)*df.dx, mesh=self.mesh)

        self.set_default_values()
        self._solve = self.load_c_code()

    def set_default_values(self):
        self.alpha = 0.5
        self.gamma = 2.211e5 # m/(As)
        self.c = 1e11 # 1/s numerical scaling correction
                      # 0.1e12 1/s is the value used by default in nmag 0.2
        self.C = 1.3e-11 # J/m exchange constant
        self.Ms = 8.6e5 # A/m saturation magnetisation
        self.t = 0 # s
        self.H_app = (0, 0, 0)
        self.H_dmi = (0, 0, 0) #DMI for Skyrmions
        self.pins = [] # nodes where the magnetisation gets pinned

    def load_c_code(self):
        """
        Loads the C-code in the file dmdt.c, that will later
        get called to compute the right-hand side of the LLG equation.

        """
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        with open(os.path.join(__location__, 'dmdt.c'), "r") as f:
            c_code = f.read()

        args = [["Mn", "M", "in"], ["Hn", "H", "in"],
                ["dMdtn", "dMdt", "out"], ["Pn", "P", "in"]]
        return instant.inline_with_numpy(c_code, arrays = args)
    
    @property
    def M(self):
        """ the magnetisation, with length Ms """
        return self.Ms * self.m

    @property
    def M_average(self):
        """ the average magnetisation, computed with m_average() """
        return self.Ms * self.m_average

    @property
    def m(self):
        """ the unit magnetisation """
        return self._m.vector().array()

    @m.setter
    def m(self, value):
        # Not enforcing unit length here, as that is better done
        # once at the initialisation of m.
        self._m.vector()[:] = value

    @property
    def m_average(self):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`
        
        """
        mx = df.assemble(df.dot(self._m, df.Constant([1,0,0])) * df.dx)
        my = df.assemble(df.dot(self._m, df.Constant([0,1,0])) * df.dx)
        mz = df.assemble(df.dot(self._m, df.Constant([0,0,1])) * df.dx)
        return np.array([mx, my, mz]) / self.Volume
    
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
        elif isinstance(value, (df.Constant, df.Expression)):
            self._m0 = df.interpolate(value, self.V)
        elif isinstance(value, (list, np.ndarray)):
            self._m0 = df.Function(self.V)
            self._m0.vector()[:] = value
        else:
            raise AttributeError
        self._m0.vector()[:] = h.fnormalise(self._m0.vector().array())
        self._m = self._m0

    def update_H_eff(self):
        self.H_eff = self.H_app + self.H_ex 
        if self.use_dmi:
            self.H_eff += self.H_dmi

    @property
    def H_app(self):
        return self._H_app.vector().array()

    @H_app.setter
    def H_app(self, value):
        self._H_app = df.interpolate(df.Constant(value), self.V)

    def solve(self):
        if self.exchange_flag:
            self.H_ex = self.exchange.compute_field()
        if self.use_dmi:
            self.H_dmi = self.dmi.compute_field()
        self.update_H_eff()

        status, dMdt = self._solve(self.alpha, self.gamma, self.c,
            self.m, self.H_eff, self.m.shape[0], self.pins)
        if status == 0:
            return dMdt
        raise Exception("An error was encountered in the C-code; status=%d" % status)
        return None

    def solve_for(self, m, t):
        self.m = m
        self.t = t
        return self.solve()

    def setup(self, exchange_flag=True, use_dmi=False):
        self.exchange_flag = exchange_flag
        if exchange_flag:
            self.exchange = Exchange(self.V, self._m, self.C, self.Ms)
        else:
            zero = df.Constant((0, 0, 0))
            self.H_ex = df.interpolate(zero, self.V).vector().array()

        self.use_dmi = use_dmi

        if use_dmi:
            self.dmi = DMI(self.V, self._M, self.Ms)


