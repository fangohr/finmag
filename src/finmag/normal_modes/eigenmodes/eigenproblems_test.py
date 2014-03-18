from __future__ import division
import pytest
import logging
import os
from eigenproblems import *
from eigensolvers import *
from helpers import is_diagonal_matrix


def test_assert_eigenproblems_were_defined():
    """
    Check that `available_eigenproblems` is not the empty set and that
    a few known eigenproblem types are contained in it.

    """
    assert(set(available_eigenproblems) != set())
    assert(all([isinstance(ep, AbstractEigenproblem) for ep in available_eigenproblems]))
    assert(any([isinstance(ep, DiagonalEigenproblem) for ep in available_eigenproblems]))


class MockEigenproblem(AbstractEigenproblem):
    eigvals = {0: 0.0,
               1: 4.0,
               2: 4.0,
               3: 7.2,
               4: 7.2,
               5: 7.2,
               6: 10.4,
               }

    def get_kth_analytical_eigenvalue(self, k, size=None):
        return self.eigvals[k]

    def get_kth_analytical_eigenvector(self, k, size):
        return np.zeros(size)


class AbstractEigenproblemTest(object):
    """
    This class provides generic tests that will be called from each of
    the individual subclasses below (it won't be executed by py.test
    directly, however).

    """
    def test_plot_analytical_solutions(self, tmpdir):
        os.chdir(str(tmpdir))
        self.eigenproblem.plot_analytical_solutions(
            [0, 2, 5, 6], N=20, figsize=(12, 3), filename='solutions.png')
        assert(os.path.exists('solutions.png'))

    def test_plot_computed_solutions(self, tmpdir):
        os.chdir(str(tmpdir))
        solver = ScipyLinalgEig()
        self.eigenproblem.plot_computed_solutions(
            [0, 2, 5, 6], solver=solver, N=50, dtype=float, tol_eigval=1e-1,
            figsize=(12, 3), filename='solutions.png')
        assert(os.path.exists('solutions.png'))


class TestDiagonalEigenproblem(AbstractEigenproblemTest):
    def setup(self):
        self.eigenproblem = DiagonalEigenproblem()
        self.omega_ref = [1, 2, 3, 4]
        self.num = 4
        self.size = 10
        self.w_ref = \
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
        self.w_ref2 = \
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, -3.43, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1.173e-3, 0, 0, 0, 0, 0, 0]]
        self.w_wrong = \
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
        self.eigenpairs_ref = zip(self.omega_ref, self.w_ref)
        self.eigenpairs_wrong = zip(self.omega_ref, self.w_wrong)

    def test_instantiate(self):
        """
        Check that the matrices instantiated by DiagonalEigenproblem are
        a diagonal matrix and `None` as expected.

        """
        for N in [10, 20, 40]:
            for dtype in [float, complex]:
                A, M = self.eigenproblem.instantiate(N, dtype=dtype)
                assert(is_diagonal_matrix(A))
                assert(A.dtype == dtype)
                assert(M == None)

    def test_solve(self):
        """
        Call 'self.solve()' with two different solvers and check that
        the solutions are as expected.

        """
        solver1 = ScipyLinalgEig(num=4)
        solver2 = ScipySparseLinalgEigs(sigma=0.0, which='LM', num=4)
        omega1, w1, _ = self.eigenproblem.solve(solver1, N=10, dtype=float)
        omega2, w2, _ = self.eigenproblem.solve(solver2, N=10, dtype=complex)
        assert(self.eigenproblem.verify_eigenpairs_analytically(zip(omega1, w1)))
        assert(self.eigenproblem.verify_eigenpairs_analytically(zip(omega2, w2)))

    def test_print_analytical_eigenvalues(self):
        self.eigenproblem.print_analytical_eigenvalues(10, unit='Hz')
        self.eigenproblem.print_analytical_eigenvalues(4, unit='KHz')
        self.eigenproblem.print_analytical_eigenvalues(5, unit='MHz')
        self.eigenproblem.print_analytical_eigenvalues(20, unit='GHz')

        with pytest.raises(TypeError):
            # Without explicitly specifying the 'unit' keyword the
            # second argument will be interpreted as the problem size,
            # which should lead to an error.
            self.eigenproblem.print_analytical_eigenvalues(10, 'Hz')

    def test_get_kth_analytical_eigenvalue(self):
        for k in xrange(self.num):
            omega = self.eigenproblem.get_kth_analytical_eigenvalue(k=k)
            omega_ref = self.omega_ref[k]
            assert(np.allclose(omega, omega_ref))

    def test_get_analytical_eigenvalues(self):
        omega = self.eigenproblem.get_analytical_eigenvalues(num=self.num)
        assert(np.allclose(omega, self.omega_ref))

    def test_get_kth_analytical_eigenvector(self):
        for k in xrange(self.num):
            w = self.eigenproblem.get_kth_analytical_eigenvector(k=k, size=self.size)
            w_ref = self.w_ref[k]
            assert(np.allclose(w, w_ref))

    def test_get_analytical_eigenvectors(self):
        w = self.eigenproblem.get_analytical_eigenvectors(num=self.num, size=self.size)
        assert(np.allclose(w, self.w_ref))

    def test_get_kth_analytical_eigenpair(self):
        for k in xrange(self.num):
            omega, w = self.eigenproblem.get_kth_analytical_eigenpair(k=k, size=self.size)
            omega_ref = self.omega_ref[k]
            w_ref = self.w_ref[k]
            assert(np.allclose(omega, omega_ref) and np.allclose(w, w_ref))

    def test_get_analytical_eigenpairs(self):
        omega, w = self.eigenproblem.get_analytical_eigenpairs(num=self.num, size=self.size)
        assert(np.allclose(omega, self.omega_ref))
        assert(np.allclose(w, self.w_ref))

    def test_get_analytical_eigenspace_basis(self):
        N = self.size
        omega1 = 1.0
        omega4 = 4.0
        omega6 = 6.0
        eigenvec1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        eigenvec4 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        eigenvec6 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        def assert_eigenspace_basis(esp, eigenvec):
            esp = np.asarray(esp)
            esp_ref = np.asarray([eigenvec])
            assert(np.allclose(esp, esp_ref))

        esp1 = self.eigenproblem.get_analytical_eigenspace_basis(omega1, N)
        esp4 = self.eigenproblem.get_analytical_eigenspace_basis(omega4, N)
        esp6 = self.eigenproblem.get_analytical_eigenspace_basis(omega6, N)

        assert_eigenspace_basis(esp1, eigenvec1)
        assert_eigenspace_basis(esp4, eigenvec4)
        assert_eigenspace_basis(esp6, eigenvec6)

        # Check that using a non-eigenvalue raises an exception
        omega_wrong = 4.2
        with pytest.raises(ValueError):
            self.eigenproblem.get_analytical_eigenspace_basis(omega_wrong, N)

        # Check that with a less strict tolerance we get the expected result
        omega4b = 4.0001
        esp4b = self.eigenproblem.get_analytical_eigenspace_basis(omega4b, N, tol_eigval=1e-3)
        assert_eigenspace_basis(esp4b, eigenvec4)

        # Check some corner cases (e.g. where 'computed' eigenvalues are
        # slightly smaller or larger than the exact ones).
        a1 = 4.000001
        a2 = 3.999999
        a3 = 7.200001
        a4 = 7.199999
        mock_eigenproblem = MockEigenproblem()
        esp1 = mock_eigenproblem.get_analytical_eigenspace_basis(a1, size=50, tol_eigval=1e-3)
        esp2 = mock_eigenproblem.get_analytical_eigenspace_basis(a2, size=50, tol_eigval=1e-3)
        esp3 = mock_eigenproblem.get_analytical_eigenspace_basis(a3, size=50, tol_eigval=1e-3)
        esp4 = mock_eigenproblem.get_analytical_eigenspace_basis(a4, size=50, tol_eigval=1e-3)
        assert(len(esp1) == 2)
        assert(len(esp2) == 2)
        assert(len(esp3) == 3)
        assert(len(esp4) == 3)


    def test_verify_eigenpair_numerically(self):
        """
        Apply the method `verify_eigenpair_numerically()` to the
        reference eigenpairs (which are known to be correct) and check
        that it returns `True` (= succesful verification). Similarly,
        check that it returns wrong on a set of wrong eigenpairs.

        """
        # Check all correct eigenpairs (should succeed)
        for k in xrange(self.num):
            a = self.omega_ref[k]
            v = self.w_ref[k]
            res = self.eigenproblem.verify_eigenpair_numerically(a, v)
            assert(res == True)

        # Check a wrong eigenpair (should fail)
        a = 3.0
        v = std_basis_vector(1, self.size)
        res = self.eigenproblem.verify_eigenpair_numerically(a, v, tol=1e-8)
        assert(res == False)

        with pytest.raises(ValueError):
            # v should be a 1D vector, so this should raise an exception
            v_wrong_shape = [[1, 2, 3], [4, 5, 6]]
            self.eigenproblem.verify_eigenpair_numerically(1.0, v_wrong_shape)

    def test_verify_eigenpairs_numerically(self):
        assert(self.eigenproblem.verify_eigenpairs_numerically(self.eigenpairs_ref) == True)
        assert(self.eigenproblem.verify_eigenpairs_numerically(self.eigenpairs_wrong) == False)

    def test_verify_eigenpair_analytically(self):
        """
        Apply the method `verify_eigenpair_analytically()` to the
        reference eigenpairs (which are known to be correct) and check
        that it returns `True` (= successful verification). Similarly,
        check that it returns wrong on a set of wrong eigenpairs.

        """
        # Check all correct eigenpairs (should succeed)
        for k in xrange(self.num):
            a = self.omega_ref[k]
            v = self.w_ref[k]
            res = self.eigenproblem.verify_eigenpair_analytically(a, v)
            assert(res == True)

        # Check a wrong eigenpair (should fail)
        a = 3.0
        v = std_basis_vector(1, self.size)
        res = self.eigenproblem.verify_eigenpair_analytically(
            a, v, tol_residual=1e-8, tol_eigval=1e-8)
        assert(res == False)

        with pytest.raises(TypeError):
            # v should be a 1D vector, so this should raise an exception
            v_wrong_shape = [[1, 2, 3], [4, 5, 6]]
            self.eigenproblem.verify_eigenpair_analytically(1.0, v_wrong_shape)

    def test_verify_eigenpairs_analytically(self):
        assert(self.eigenproblem.verify_eigenpairs_analytically(self.eigenpairs_ref) == True)
        assert(self.eigenproblem.verify_eigenpairs_analytically(self.eigenpairs_wrong) == False)


class TestRingGraphLaplaceEigenproblem(AbstractEigenproblemTest):
    def setup(self):
        self.eigenproblem = RingGraphLaplaceEigenproblem()

    def test_instantiate(self):
        """
        Check that the matrices instantiated by RingGraphLaplaceEigenproblem
        are as expected.

        """
        A4, M4 = self.eigenproblem.instantiate(N=4, dtype=float)
        assert(M4 == None)
        assert(np.allclose(A4, np.array([[ 2, -1,  0, -1],
                                         [-1,  2, -1,  0],
                                         [ 0, -1,  2, -1],
                                         [-1,  0, -1,  2]])))

        A5, M5 = self.eigenproblem.instantiate(N=5, dtype=complex)
        assert(M5 == None)
        assert(np.allclose(A5, np.array([[2, -1, 0, 0, -1],
                                         [-1, 2, -1, 0, 0],
                                         [0, -1, 2, -1, 0],
                                         [0, 0, -1, 2, -1],
                                         [-1, 0, 0, -1, 2]])))

        for N in [10, 20, 40]:
            for dtype in [float, complex]:
                A, M = self.eigenproblem.instantiate(N, dtype=dtype)
                assert(M == None)
                A_ref = np.zeros((N, N), dtype=dtype)
                A_ref += np.diag(2 * np.ones(N))
                A_ref -= np.diag(np.ones(N-1), k=1)
                A_ref -= np.diag(np.ones(N-1), k=-1)
                A_ref[0, N-1] = -1
                A_ref[N-1, 0] = -1

                assert(np.allclose(A, A_ref))
                assert(M == None)


class TestNanostrip1dEigenproblemFinmag(AbstractEigenproblemTest):
    def setup(self):
        self.eigenproblem = Nanostrip1dEigenproblemFinmag(13e-12, 8e5, 0, 100, unit_length=1e-9)
