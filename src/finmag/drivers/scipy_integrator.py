from scipy.integrate import ode

class ScipyIntegrator(object):
    def __init__(self, llg, m0, reltol=1e-8, abstol=1e-8, nsteps=10000, method="bdf", tablewriter=None, **kwargs):
        self.llg = llg
        self.cur_t = 0.0
        self.ode = ode(self.rhs, jac=None)
        self.m = self.llg.m[:] = m0
        self.ode.set_integrator("vode", method=method, rtol=reltol, atol=abstol, nsteps=nsteps, **kwargs)
        self.ode.set_initial_value(m0, 0)
        self._n_rhs_evals = 0
        self.tablewriter = tablewriter

    n_rhs_evals = property(lambda self: self._n_rhs_evals, "Number of function evaluations performed")

    def rhs(self, t, y):
        self._n_rhs_evals += 1
        return self.llg.solve_for(y, t)

    def advance_time(self, t):
        new_m = self.ode.integrate(t)
        assert self.ode.successful()
        self.m = new_m
        self.cur_t = t

    def reinit(self):
        raise NotImplementedError("{}: This integrator doesn't support the reinit method.".format(self.__class__.__name__))
