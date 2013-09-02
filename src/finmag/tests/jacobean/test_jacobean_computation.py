import numpy as np
import unittest
from domain_wall_cobalt import setup_domain_wall_cobalt

def norm(a):
    return np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

# Set up the LLG with all parameters close to 1
def setup_llg_params_near_one(node_count=5, A=3.6 * 4e-7 * np.pi, Ms=6.7e5, K1=4.3, do_precession=True):
    llg = setup_domain_wall_cobalt(node_count=node_count, A=A, Ms=Ms, K1=K1, length=1.3, do_precession=do_precession)
    llg.c = 1.23
    llg.gamma = 1.56
    llg.set_alpha(2.35)
    llg.pins = []
    n = llg.m.size / 3
    # Generate a random (non-normalised) magnetisation vector with norm close to 1
    np.random.seed(1)
    m = np.random.rand(3, n) * 2 - 1
    m = 0.1 * m + 0.9 * (m / norm(m))
    m.shape = (-1,)
    return llg, m

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
        tmp = np.zeros(m.shape)
        jtimes = np.zeros(m.shape)
        for j, v in enumerate(np.eye(3*n)):
            # use fy=None since it's not used for the computation
            self.assertGreaterEqual(self.llg.sundials_jtimes(v, jtimes, 0., m, None, tmp), 0)
            jac[:, j] = jtimes
        return jac

    def test_compute_fd(self):
        self.llg, m = setup_llg_params_near_one()

        # Jacobean computation should be exact with eps=1 or eps=2
        self.assertLess(np.max(np.abs(self.compute_jacobean_fd(m, eps=1) - self.compute_jacobean_fd(m, eps=2))), 1e-13)

    def test_compute_jtimes(self):
        self.llg, m = setup_llg_params_near_one()
        self.assertLess(np.max(np.abs(self.compute_jacobean_jtimes(m) - self.compute_jacobean_fd(m))), 1e-13)

    def test_compute_jtimes_pinning(self):
        self.llg, m = setup_llg_params_near_one()
        self.llg.pins = [0,3,4]
        self.assertLess(np.max(np.abs(self.compute_jacobean_jtimes(m) - self.compute_jacobean_fd(m))), 1e-13)

    def test_compute_jtimes_no_precession(self):
        self.llg, m = setup_llg_params_near_one(do_precession=False)
        self.assertLess(np.max(np.abs(self.compute_jacobean_jtimes(m) - self.compute_jacobean_fd(m))), 1e-13)

if __name__=="__main__":
    unittest.main()
