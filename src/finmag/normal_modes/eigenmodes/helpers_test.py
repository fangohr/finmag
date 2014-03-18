from __future__ import division
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from helpers import *


def test_iseven():
    for n in [0, 2, 6, 1223456, -18]:
        assert(iseven(n))
    for n in [1, 7, -19, -5, 787823]:
        assert(not iseven(n))


def test_is_hermitian():
    A1 = np.array([[1, -2, 3, 4],
                   [-2, 5, 12, -33],
                   [3, 12, 42, -8],
                   [4, -33, -8, 0.2]])
    assert(is_hermitian(A1))

    A2 = np.array([[0.3, 4-5j, 0.2+3j],
                   [4+5j, 2.0, -1+3.3j],
                   [0.2-3j, -1-3.3j, -4]])
    assert(is_hermitian(A2))

    A3 = np.array([[1, 2],
                   [4, 4]])
    assert(not is_hermitian(A3))

    A4 = np.array([[1, 2+3j],
                   [2+3j, 4]])
    assert(not is_hermitian(A4))

    A5 = np.array([[1+1j, 2+3j],
                   [2-3j, 4]])
    assert(not is_hermitian(A5))


@pytest.mark.parametrize(
    "v, vn_expected",
    [([1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]),
     ([1, 2, 3, -1, -1], [0.25, 0.5, 0.75, -0.25, -0.25]),
     ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
     ([1e-14, 1e-14, 1e-13, 1e-13], [1e-14, 1e-14, 1e-13, 1e-12]),
     ])
def test_normalise_if_not_zero(v, vn_expected):
    vn = normalise_if_not_zero(v)
    assert(np.allclose(vn, vn_expected))


def test_normalise_rows():
    A = np.array([[2, 0, 0, 0, 0],
                  [0, -3, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 1e-16, 0],
                  [0, 3, 0, 0, 4]])
    A_normalised = normalise_rows(A)
    A_normalised_ref = np.array([[1, 0, 0, 0, 0],
                                 [0, -1, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1e-16, 0],
                                 [0, 3./5, 0, 0, 4./5]])
    assert(np.allclose(A_normalised, A_normalised_ref))


def test_is_diagonal_matrix():
    A1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    A2 = np.array([[4, 0, 0],
                   [0, -5, 0],
                   [0, 0, 1e-4]])
    assert(not is_diagonal_matrix(A1))
    assert(is_diagonal_matrix(A2))


@pytest.mark.parametrize("a, tol",
                         [(0.253, 5e-9),
                          (1.42e-8, 5e-5),
                          (5.4e4, 1e-8),
                          (1.2 + 4.4j, 1e-8)])
def test_is_scalar_multiple(a, tol):
    """
    Create two random arrays of dtype `float` and `complex`, respectively.
    Multiply each by the scalar factor `a` and check that the result is
    recognised as a scalar multiple.
    """
    print "[DDD] a={}".format(a)

    N = 100
    rand_vec = np.random.random_sample(N)
    rand_vec[[0, 40, 62]] = 0.0  # set a few values to zero
    v = np.asarray(rand_vec, dtype=type(a))
    w = a * v
    assert(is_scalar_multiple(v, w, tol=tol))


@pytest.mark.parametrize("eps, num_elements",
                         [(4.2, 1),
                          (0.155, 1),
                          (1e-3, 1),
                          (1e-5, 5),
                          (5e-6, 50),
                          ])
def test_not_is_scalar_multiple(eps, num_elements):
    print "[DDD] eps={}, num_elements={}".format(eps, num_elements)
    N = 100
    w = np.random.random_sample(N)
    v = 3.24 * w
    v[range(num_elements)] += eps
    assert(not is_scalar_multiple(v, w))


def test_find_matching_eigenpair():
    """
    Construct a list of reference eigenpairs and check for a few artificial
    'computed' eigenpairs whether they match any of the pairs in the list.

    """
    eigenpairs_ref = [
        (42.0, [1, 2, 3, 4, 5]),
        (23.0, [4, 4, 4, -4, 6]),
        (12.4, [-1, -2, 3, 2, -1]),
        (23.0, [4, 4, 4, -4, 6])
        ]

    omega1, w1 = (12.4, [-1, -2, 3, 2, -1])             # matches 3rd eigenpair
    omega2, w2 = (12.4, [1.9, 3.8, -5.7, -3.8, 1.9])    # matched 3rd eigenpair
    omega3, w3 = (42.000000001, [-2, -4, -6, -8, -10])  # matches 1st eigenpair
    omega4, w4 = (42.0, [1, 3, 3, 4, 5])  # no match
    omega5, w5 = (42.3, [1, 2, 3, 4, 5])  # no match
    omega6, w6 = (23.0, [4, 4, 4, -4, 6]) # duplicate match; should throw error

    idx1 = find_matching_eigenpair((omega1, w1), eigenpairs_ref)
    idx2 = find_matching_eigenpair((omega2, w2), eigenpairs_ref)
    idx3 = find_matching_eigenpair((omega3, w3), eigenpairs_ref)
    idx4 = find_matching_eigenpair((omega4, w4), eigenpairs_ref)
    idx5 = find_matching_eigenpair((omega5, w5), eigenpairs_ref)

    assert(idx1 == 2)
    assert(idx2 == 2)
    assert(idx3 == 0)
    assert(idx4 == None)
    assert(idx5 == None)

    with pytest.raises(EigenproblemVerifyError):
        # Duplicate match should raise an exception
        find_matching_eigenpair((omega6, w6), eigenpairs_ref)


def test_standard_basis_vector():
    e_1_5 = [1, 0, 0, 0, 0]
    e_4_5 = [0, 0, 0, 1, 0]
    e_5_7 = [0, 0, 0, 0, 1, 0, 0]
    assert(np.allclose(e_1_5, std_basis_vector(1, N=5)))
    assert(np.allclose(e_4_5, std_basis_vector(4, N=5, dtype=float)))
    assert(np.allclose(e_5_7, std_basis_vector(5, N=7, dtype=complex)))


def test_sort_eigensolutions():
    w1 = [0, 1, 2, 3, 4, 5]
    w2 = [10, 11, 12, 13, 14, 15]
    w3 = [20, 21, 22, 23, 24, 25]
    w4 = [30, 31, 32, 33, 34, 35]
    w5 = [40, 41, 42, 43, 44, 45]
    ws = [w1, w2, w3, w4, w5]

    omegas_sorted, ws_sorted = sort_eigensolutions([4, 1, 3, 7, 5], ws)
    assert(np.allclose(omegas_sorted, [1, 3, 4, 5, 7]))
    assert(np.allclose(ws_sorted, [w2, w3, w1, w5, w4]))

    omegas_sorted, ws_sorted = sort_eigensolutions([4, 3.3, 2.1, 5.5, 2.9], ws)
    assert(np.allclose(omegas_sorted, [2.1, 2.9, 3.3, 4, 5.5]))
    assert(np.allclose(ws_sorted, [w3, w5, w2, w1, w4]))


def test_best_linear_combination():
    """
    TODO: Write me!!
    """
    e1 = [1, 0, 0]
    e2 = [0, 1, 0]
    e3 = [0, 0, 1]
    v = [-2, 1, -3]

    w, coeffs, res = best_linear_combination(v, [e1, e2])
    assert(np.allclose(w, [-2, 1, 0]))
    assert(np.allclose(coeffs, [-2, 1]))
    assert(np.allclose(res, 3.0))

    w, coeffs, res = best_linear_combination(v, [e1, e3])
    assert(np.allclose(w, [-2, 0, -3]))
    assert(np.allclose(coeffs, [-2, -3]))
    assert(np.allclose(res, 1.0))

    w, coeffs, res = best_linear_combination(v, [e1, e2, e3])
    assert(np.allclose(w, v))
    assert(np.allclose(coeffs, [-2, 1, -3]))
    assert(np.allclose(res, 0.0))

    # More complicated example
    e1 = np.array([1, 0, 3])
    e2 = np.array([-2, 4, -6])
    a = 0.5
    b = 0.25
    v0 =  np.array([-3, 0, 1])
    v = v0 + a * e1 + b * e2
    w, coeffs, res = best_linear_combination(v, [e1, e2])
    assert(np.allclose(w, a * e1 + b * e2))
    assert(np.allclose(coeffs, [a, b]))
    assert(np.allclose(res, np.linalg.norm(v0)))


def scipy_sparse_linear_operator_to_dense_array():
    """
    Create a random matrix, convert it to a LinearOperator and use
    'as_dense_array' to convert it back. Check that the result is
    the same as the original matrix.

    """
    for N in [10, 20, 30, 50, 100]:
        A = np.random.random_sample((N, N))
        A_sparse = LinearOperator(shape=(N, N), matvec=lambda v: np.dot(A, v))
        A_dense = as_dense_array(A_sparse)
        assert(np.allclose(A, A_dense))


def test_as_dense_array():
    """
    Define a dense tridiagonal matrix (with 'periodic boundary
    conditions') and its equivalents as LinearOperator, PETSc.Mat and
    dolfin.PETScMatrix. Then check that as_dense_array() returns the
    same array for all of them.

    """
    N = 50

    # Define dense matrix
    A = np.zeros((N, N))
    for i in xrange(N):
        A[i, i] = 2.4
        A[i, i-1] = 5.5
        A[i, (i+1)%N] = 3.7

    # Define equivalent LinearOperator
    def A_matvec(v):
        # Note that the vectors end up as column vectors, thus the
        # direction we need to roll the arrays might seem
        # counterintuitive.
        return 2.4 * v + 5.5 * np.roll(v, +1) + 3.7 * np.roll(v, -1)
    A_LinearOperator = LinearOperator((N, N), matvec=A_matvec)

    # Define equivalent PETSc.Mat()
    A_petsc = as_petsc_matrix(A)

    # Define equivalent dolfin.PETScMatrix
    #
    # TODO: This only works in the development version of dolfin,
    #       which is not released yet. Reactivate this test once it's
    #       available!
    #
    # A_petsc_dolfin = df.PETScMatrix(A_petsc)


    # For numpy.arrays the exact same object should be returned (if
    # the dtype is the same)
    assert(id(A) == id(as_dense_array(A)))
    A_complex = np.asarray(A, dtype=complex)
    assert(id(A_complex) == id(as_dense_array(A_complex)))

    # Check that the conversion works correctly
    assert(as_dense_array(None) == None)
    assert(np.allclose(A, as_dense_array(A, dtype=complex)))
    assert(np.allclose(A, as_dense_array(A_LinearOperator)))
    assert(np.allclose(A, as_dense_array(A_petsc)))
    #assert(np.allclose(A, as_dense_array(A_petsc_dolfin)))


def test_as_petsc_matrix():
    N = 20

    # Create a tridiagonal matrix with random entries
    A = np.zeros((N, N))
    a = np.random.random_sample(N-1)
    b = np.random.random_sample(N)
    c = np.random.random_sample(N-1)
    A += np.diag(a, k=-1)
    A += np.diag(b, k=0)
    A += np.diag(c, k=+1)
    print "[DDD] A:"
    print A

    # Convert to PETSC matrix
    A_petsc = as_petsc_matrix(A)

    # Check that the sparsity pattern is as expected
    indptr, _, data = A_petsc.getValuesCSR()
    indptr_expected = [0] + range(2, 3*(N-1), 3) + [3*N-2]
    assert(all(indptr == indptr_expected))

    # Convert back to numpy array
    B = as_dense_array(A_petsc)
    assert(np.allclose(A, B))

    # A numpy array of dtype complex can only be converted to a PETSc
    # matrix if the imaginary part is zero.
    C = (1.0 + 0.j) * np.random.random_sample((N, N))
    assert(C.dtype == complex)  # check that we created a complex array
    C2 = as_dense_array(as_petsc_matrix(C))
    assert(np.allclose(C, C2))

    D = 1j * np.random.random_sample((N, N))
    assert(D.dtype == complex)  # check that we created a complex array
    with pytest.raises(TypeError):
        as_petsc_matrix(D)

    # Check that we can also convert a LinearOperator to a PETScMatrix
    A_LinearOperator = LinearOperator(shape=(N, N), matvec=lambda v: np.dot(A, v))
    A_petsc2 = as_petsc_matrix(A_LinearOperator)
    A_roundtrip = as_dense_array(A_petsc2)
    assert(np.allclose(A, A_roundtrip))
