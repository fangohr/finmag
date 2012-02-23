from dolfin import *
from finmag.sim.llg import LLG
from finmag.sim.exchange import Exchange
from math import log
import random

class MyLLG(LLG):
    """
    Temporary extension of LLG because the current version
    does its computations externally in C++, and doesn't 
    compute the variational forms, and thus makes it
    impossible to compute the jacobian.
    """
    def __init__(self, mesh, order=1):
        LLG.__init__(self, mesh, order)
        self.p = Constant(self.gamma/(1 + self.alpha**2))

    def M(self):
        return self._M

    def H_eff(self):
        """Very temporary function to make things simple."""
        H_app = project((Constant((0, 1e5, 0))), self.V)
        H_ex  = Function(self.V)

        # Comment out these two lines if you don't want exchange.
        exch  = Exchange(self.V, self._M, self.C, self.Ms)
        H_ex.vector().array()[:] = exch.compute_field()

        H_eff = H_ex + H_app
        return H_eff

    def compute_variational_forms(self):
        M, H, Ms, p, c, alpha, V = self._M, self.H_eff(), \
                self.Ms, self.p, self.c, self.alpha, self.V
        
        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(u, v)*dx
        L = inner((-p*cross(M,H)
                   -p*alpha/Ms*cross(M,cross(M,H))
                   -c*(inner(M,M) - Ms**2)*M/Ms**2), v)*dx

        self.a, self.L = a, L

    def variational_forms(self):
        self.compute_variational_forms()
        return self.a, self.L

    def compute_jacobian(self):
        L, M = self.L, self._M
        return derivative(L, M)


def derivative_test(L, M, x, hs, J=None):
    """
    Taylor remainder test. If Jacobian J is given, use that, if not
    don't.
    """

    L_M = assemble(L)
    errors = []
    for h in hs:
        H = Function(V)
        H.vector()[:] = h*x.vector() # h*x

        P = Function(V)
        P.vector()[:] = M.vector() + H.vector()

        L_P = assemble(replace(L, {M: P}))

        # Without Jacobian information
        if J is None:
            errors += [norm(L_P - L_M)]
        # With Jacobian information
        else:
            J_M_H = assemble(action(J, H))
            errors += [norm(L_P - L_M - J_M_H)]

    return errors

def convergence_rates(xs, ys):
    assert(len(xs) == len(ys))
    rates = [(log(ys[i]) - log(ys[i-1]))/(log(hs[i]) - log(hs[i-1]))
             for i in range(1, len(xs))]
    return rates


m = 1e-5
mesh = Box(0,m,0,m,0,m,5,5,5)
llg = MyLLG(mesh)
llg.initial_M((8.6e5,0,0))
llg.setup()

M, V = llg.M(), llg.V
a, L = llg.variational_forms()

x = Function(V)
s = random.random()
#s = random.randint(0, 1e5)
x.vector()[:] = s
hs = [2.0/n for n in (1, 2, 4, 8, 16, 32)]

TOL = 1e-5

def test_convergence():
    """All convergence rates should be 1 as the differences 
    should convert as O(n)."""

    errors = derivative_test(L, M, x, hs)
    rates = convergence_rates(hs, errors)
    for rate in rates:
        assert abs(rate - 1) < TOL

def test_derivative():
    """This should be zero because the rhs of LLG is linear in M."""
    J = llg.compute_jacobian()
    errors = derivative_test(L, M, x, hs, J=J)
    for err in errors:
        assert abs(err) < TOL

if __name__ == '__main__':
    errors = derivative_test(L, M, x, hs)
    print "This should be close to one:"
    print convergence_rates(hs, errors)
    J = llg.compute_jacobian()
    errors = derivative_test(L, M, x, hs, J=J)
    print "This should be close to zero:"
    print errors

