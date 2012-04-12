import numpy as np
from finmag.native.llg import compute_solid_angle
import scipy.linalg
import scipy.stats
import math
import unittest

# Quaternion multiplicatoin
def quaternion_product(a, b):
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    ])

# Returns a nx3x3 array with n random 3x3 matrix uniformly distributed in SO(3)
def random_3d_rotation_matrix(n):
    # Create a random unit quaternion
    q = np.random.randn(4,n)
    q /= np.sqrt(np.sum(q*q, axis=0))
    qinv = np.array([q[0], -q[1], -q[2], -q[3]])
    # Set up the SO(3) matrix defined by the quaternion
    a = np.array([
        quaternion_product(qinv, quaternion_product([0,1,0,0], q))[1:],
        quaternion_product(qinv, quaternion_product([0,0,1,0], q))[1:],
        quaternion_product(qinv, quaternion_product([0,0,0,1], q))[1:]
    ])
    # Disabled - return a^2 to check that the test fails
#    return np.array([np.dot(x, x) for x in a.T])
    return a.T

from finmag.util.solid_angle_magpar import return_csa_magpar
csa_magpar = return_csa_magpar()

def solid_angle_magpar(r, T):
    assert r.shape == (3,)
    assert T.shape == (3,3)
    # First index of T is node number, second spatial
    return csa_magpar(r, T[0], T[1], T[2])

def solid_angle_llg(r, T):
    assert r.shape == (3,)
    assert T.shape == (3,3)
    # First index of T is node number, second spatial
    return compute_solid_angle(r.reshape(3,1), T.reshape(3,3,1))[0]

class SolidAngleInvarianceTests(unittest.TestCase):
    def test_rotation_matrix(self):
        np.random.seed(1)
        matrices = random_3d_rotation_matrix(1000)
        # Test that the determinant is 1
        assert np.max(np.abs([scipy.linalg.det(m) - 1 for m in matrices])) < 1e-12
        # Test that the matrix is orthogonal
        assert np.max(np.abs([np.dot(m, m.T)-np.eye(3) for m in matrices])) < 1e-12
        np.random.seed(1)
        # The uniform distribution in SO(3) is unchanged under arbitrary rotationss
        # Here, we only test the [0,0] component
        n = 2000
        m1 = random_3d_rotation_matrix(n)
        m2 = random_3d_rotation_matrix(n)

        def p_values():
            for a in random_3d_rotation_matrix(10):
                for i in xrange(3):
                    for j in xrange(3):
                        yield scipy.stats.ks_2samp(m1[:,i,j], np.dot(m2, a)[:,i,j])[1]
        p = list(p_values())
        assert np.min(p) > 0.0001

    # The solid angle is invariant under 3d rotations that preserve orientation (SO(3))
    # and changes sign for orthogonal transformations that change orientation
    # (O(3) transformations not in SO(3))
    def test_solid_angle(self):
        np.random.seed(1)
        for i in xrange(1000):
            r = np.random.randn(3)
            T = np.random.randn(3,3)
            q = random_3d_rotation_matrix(1)[0]
            r_rotated = np.dot(q.T, r)
            T_rotated = np.dot(T, q)
            r_mirror = r[[1,0,2]].copy()
            T_mirror = T[:,[1,0,2]].copy()
            angle_llg = solid_angle_llg(r, T)
            angle_magpar = solid_angle_magpar(r, T)
            angle_llg_rotated = solid_angle_llg(r_rotated, T_rotated)
            angle_llg_mirror = solid_angle_llg(r_mirror, T_mirror)
            # Check the C++ solid angle vs magpar solid angle
            self.assertAlmostEqual(math.fabs(angle_llg), angle_magpar)
            # Check the LLG solid angle vs rotated LLG solid angle
            self.assertAlmostEqual(angle_llg, angle_llg_rotated)
            # Check the C++ solid angle vs magpar solid angle
            self.assertAlmostEqual(angle_llg, -angle_llg_mirror)

if __name__=="__main__":
    unittest.main()
