from dolfin import *
from finmag.physics.llg import LLG
from finmag.energies import Exchange
from math import log

class MyLLG(LLG):
    """
    Temporary extension of LLG because the current version
    does its computations externally in C++, and doesn't
    compute the variational forms, and thus makes it
    impossible to compute the jacobian.
    """
    def __init__(self, S1, S3):
        LLG.__init__(self, S1, S3)
        self.alpha = 0.5
        self.p = Constant(self.gamma/(1 + self.alpha**2))

    def M(self):
        return self._m

    def H_eff(self):
        """Very temporary function to make things simple."""
        H_app = project((Constant((0, 1e5, 0))), self.S3)
        H_ex  = Function(self.S3)

        # Comment out these two lines if you don't want exchange.
        exch = Exchange(1.3e-11)
        print "About to cal setup"
        exch.setup(self.S3, self._m, self.Ms)
        H_ex.vector().array()[:] = exch.compute_field()

        H_eff = H_ex + H_app
        return H_eff

    def compute_variational_forms(self):
        M, H, Ms, p, c, alpha, V = self._m, self.H_eff(), \
                self.Ms, self.p, self.c, self.alpha, self.S3

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
        L, M = self.L, self._m
        return derivative(L, M)


def derivative_test(L, M, x, hs, J=None):
    """
    Taylor remainder test.

    *Arguments*

      L - right hand side of equation

      M - magnetisation field vector around which we develop the taylor series

      x - random vector

      hs - sequence of step width h

      We compute the taylor series of dm/dt represented by L for a statevector P = M + h*x

      J - Jacobian. If Jacobian J is given, use that, if not don't.
    """

    L_M = assemble(L)
    errors = []
    for h in hs:
        H = Function(V)
        H.vector().set_local(h * x.vector().array())

        P = Function(V)
        P.vector().set_local(M.vector().array() + H.vector().array())

        L_P = assemble(replace(L, {M: P})) #Compute exact result

        # Without Jacobian information
        if J is None:
            errors.append(norm(L_P - L_M))
        # With Jacobian information
        else:
            J_M_H = assemble(action(J, H))
            errors.append(norm(L_P - L_M - J_M_H))

    return errors

def convergence_rates(hs, ys):
    assert(len(hs) == len(ys))
    rates = [(log(ys[i]) - log(ys[i-1]))/(log(hs[i]) - log(hs[i-1]))
             for i in range(1, len(hs))]
    return rates


m = 1e-5
mesh = BoxMesh(0,0,0,m,m,m,5,5,5)
S1 = FunctionSpace(mesh, "Lagrange", 1)
S3 = VectorFunctionSpace(mesh, "Lagrange", 1)
llg = MyLLG(S1, S3)
llg.set_m((1,0,0))

M, V = llg._m, llg.S3
a, L = llg.variational_forms()

x = Function(V)
s = 0.25 #some random number
x.vector()[:] = s
hs = [2.0/n for n in (1, 2, 4, 8, 16, 32)]

TOL = 1e-11

def test_convergence_linear():
    """All convergence rates should be 1 as the differences
    should convert as O(n)."""

    errors = derivative_test(L, M, x, hs)
    rates = convergence_rates(hs, errors)
    for h,rate in zip(hs,rates):
        print "h= %g, rate=%g, rate-1=%g " % (h,rate,rate-1)
        assert abs(rate - 1) < TOL

def test_derivative_linear():
    """This should be zero because the rhs of LLG is linear in M."""
    J = llg.compute_jacobian()
    errors = derivative_test(L, M, x, hs, J=J)
    for h,err in zip(hs,errors):
        print "h= %g, error=%g" % (h,err)
        assert abs(err) < TOL

if __name__ == '__main__':
    # L is linear
    print "Testing linear functional."
    print "This should convert as O(h):"
    errors = derivative_test(L, M, x, hs)
    print errors
    print "This should be close to one:"
    print convergence_rates(hs, errors)
    J = llg.compute_jacobian()
    errors = derivative_test(L, M, x, hs, J=J)
    print "This should be close to zero since L is linear:"
    print errors

    test_derivative_linear()
    test_convergence_linear()

    print ''
    '''
    # L is nonlinear
    print "Testing nonlinear functional."
    print "This should convert as O(h):"
    errors = derivative_test(L, M, x, hs)
    print errors
    print "This should be close to one:"
    print convergence_rates(hs, errors)
    J = llg.compute_jacobian()
    print "This should converge as O(h^2):"
    errors = derivative_test(L, M, x, hs, J=J)
    print errors
    print "This should be close to two:"
    print convergence_rates(hs, errors)
    '''
