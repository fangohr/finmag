import dolfin as df
import instant
import os
import numpy
from finmag.sim.exchange import Exchange
from finmag.sim.dmi import DMI

class LLG(object):

    def __init__(self, mesh, order=1):
        self.mesh = mesh
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', order, dim=3)
        
        self.Volume = df.assemble(df.dot(df.TestFunction(self.V),
            df.Constant([1, 1, 1])) * df.dx).array()
        self.Vi = self.Volume.reshape((3, -1))[0]

        self.alpha = 0.5
        self.gamma = 2.211e5 # m/(As)
        self.c = 1e11 # 1/s numerical scaling correction
                      # 0.1e12 1/s is the value used by default in nmag 0.2
        self.C = 1.3e-11 # J/m exchange constant

        self.MS = 8.6e5 # A/m
        self.t = 0 #s
        self.H_app = (0, 0, 0)
        self.H_dmi = (0, 0, 0) #DMI for Skyrmions

        self.pins = [] # nodes where the magnetisation gets pinned

        self._solve = self.load_c_code()
        
    @property
    def M(self):
        return self._M.vector().array()

    @M.setter
    def M(self, value):
        self._M.vector()[:] = value

    def average_M(self):
        # compute <M> = 1/V  *  \int M dV
        integral = df.dot(self._M, df.TestFunction(self.V)) * df.dx
        average = df.assemble(integral).array() / self.Volume
        average = average.reshape((3, -1))
        return (average[0][0], average[1][0], average[2][0])

    def initial_M(self, value):
        self.M0 = df.Constant(value)
        self.reset()

    def initial_M_expr(self, expression, **kwargs):
        self.M0 = df.Expression(expression, **kwargs)
        self.reset()

    def reset(self):
        self._M = df.interpolate(self.M0, self.V)

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

    def load_c_code(self):
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        with open(os.path.join(__location__, 'dmdt.c'), "r") as f:
            c_code = f.read()

        args = [["Mn", "M", "in"], ["Hn", "H", "in"],
                ["dMdtn", "dMdt", "out"], ["Pn", "P", "in"]]
        return instant.inline_with_numpy(c_code, arrays = args)
    
    def solve(self):
        if self.exchange_flag:
            self.H_ex = self.exchange.compute_field()
        if self.use_dmi:
            self.dmi.compute_field()
        self.update_H_eff()

        status, dMdt = self._solve(self.alpha, self.gamma, self.MS, self.c,
            self.M, self.H_eff, self.M.shape[0], self.pins)
        if status == 0:
            return dMdt
        raise Exception("An error was encountered in the C-code; status=%d" % status)
        return None

    def solve_for(self, M, t):
        self.M = M
        self.t = t
        return self.solve()

    def setup(self, exchange_flag=True, use_dmi=False):
        self.exchange_flag = exchange_flag
        if exchange_flag:
            self.exchange = Exchange(self.V, self._M, self.C, self.MS)
        else:
            zero = df.Constant((0, 0, 0))
            self.H_ex = df.interpolate(zero, self.V).vector().array()

        self.use_dmi = use_dmi

        if use_dmi:
            self.dmi = DMI(self.V, self._M, self.C, self.MS)


