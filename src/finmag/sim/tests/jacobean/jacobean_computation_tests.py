import numpy as np
import dolfin as df
import unittest
import math
from finmag.sim.llg import LLG
from domain_wall_cobalt import setup_domain_wall_cobalt, domain_wall_error

def norm(a):
    return np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

def flat(a):
    res = a.view()
    res.shape = (-1,)
    return res

def nonflat(a):
    res = a.view()
    res.shape = (3,-1)
    return res

class JacobeanComputationTests(unittest.TestCase):
    # Use 4th order FD scheme
    # Without demag, this scheme should produce an exact result hence set eps to 1
    def compute_jacobean_fd(self, m, eps=1):
        n = self.llg.m.size / 3
        # Compute the jacobean using the finite difference approximation
        jac = np.zeros((3 * n, 3 * n))
        w = np.array([1./12., -2./3., 2./3., -1./12.])/eps
        for j, v in enumerate(np.eye(3 * n)):
            f0 = self.llg.solve_for(m - 2 * eps * v, 0)
            f1 = self.llg.solve_for(m - eps * v, 0)
            f2 = self.llg.solve_for(m + eps * v, 0)
            f3 = self.llg.solve_for(m + 2 * eps * v, 0)
            jac[:, j] = w[0] * f0 + w[1] * f1+ w[2] * f2 + w[3] * f3
        return jac

    # Use the jtimes function to compute the jacobean
    def compute_jacobean_jtimes(self, m):
        n = self.llg.m.size / 3
        jac = np.zeros((3 * n, 3 * n))
        for j, v in enumerate(np.eye(3*n)):
            # use fy=None and tmp=None since they are not used for the computation
            self.assertGreaterEqual(self.llg.sundials_jtimes(v, jac[:, j], 0, m, None, None), 0)
        return jac

    def setup_test_m(self):
        # Set up the LLG with all parameters close to 1
        self.llg = setup_domain_wall_cobalt(node_count=5, A=3.6 * 4e-7 * np.pi, Ms=6.7, K1=4.3, length=1.3)
        self.llg.c = 1.23
        self.llg.gamma = 1.56
        self.llg.alpha = 2.35
        self.llg.pins = []
        n = self.llg.m.size / 3
        # Generate a random (non-normalised) magnetisation vector with norm close to 1
        np.random.seed(1)
        m = np.random.rand(3, n) * 2 - 1
        m = 0.1 * m + 0.9 * (m / norm(m))
        m.shape = (-1,)
        return m

    def test_compute_fd(self):
        m = self.setup_test_m()

        # Jacobean computation should be exact with eps=1 or eps=2
        self.assertLess(np.max(np.abs(self.compute_jacobean_fd(m, eps=1) - self.compute_jacobean_fd(m, eps=2))), 1e-13)

    def atest_compute_jtimes(self):
        m = self.setup_test_m()

        # TODO: implement LLG.sundials_jtimes
        self.assertLess(np.max(np.abs(self.compute_jacobean_jtimes(m) - self.compute_jacobean_fd(m))), 1e-13)

if __name__=="__main__":
    unittest.main()
