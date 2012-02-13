import dolfin as df
import instant
import os
import numpy

class LLG(object):

    def __init__(self, mesh, order=1):
        self.mesh = mesh
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', order, dim=3)

        self.alpha = 0.5
        self.gamma = 2.211e5 # m/(As)
        self.c = 1e12 # 1/s numerical scaling correction

        self.MS = 8.6e5 # A/m
        self.H_app = (0, 0, 0)

        self.pins = [] # nodes where the magnetisation gets pinned

        self._solve = self.load_c_code()
        
    @property
    def M(self):
        return self._M.vector().array()

    @M.setter
    def M(self, value):
        self._M.vector()[:] = value

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
        self.solve_exchange()
        self.update_H_eff()

        status, dMdt = self._solve(self.alpha, self.gamma, self.MS, self.c,
            self.M, self.H_eff, self.M.shape[0], self.pins)
        if status == 0:
            return dMdt
        raise Exception("An error was encountered in the C-code.")
        return None

    def solve_for(self, M, t):
        self.M = M
        return self.solve()

    def setup(self, exchange_flag=True):
        self.exchange_flag = exchange_flag
        if exchange_flag:
            self.setup_exchange()
        else:
            self.H_ex = df.interpolate(df.Constant((0,0,0)), self.V).array()

    def setup_exchange(self):
        C = 1.3e-11 # J/m exchange constant
        mu0 = 4 * numpy.pi * 10**-7 # Vs/Am
        ex_fac = df.Constant(- 2 * C / (mu0 * self.MS))

        ex = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)

        a = df.inner(ex, v) * df.dx
        U = - ex_fac * df.inner(df.grad(self._M), df.grad(self._M)) * df.dx
        H_ex_form = df.derivative(U, self._M, v) 

        V = df.assemble(df.dot(v, df.Constant([1,1,1])) * df.dx).array()
        self.H_ex_matrix = df.assemble(H_ex_form).array() / V
        print self.H_ex_matrix, self.H_ex_matrix.shape
        self.H_ex = self.H_ex_matrix * self.M 
        print self.H_ex, self.H_ex.shape
   
    def solve_exchange(self):
        if self.exchange_flag:
            self.H_ex = self.H_ex_matrix * self.M
