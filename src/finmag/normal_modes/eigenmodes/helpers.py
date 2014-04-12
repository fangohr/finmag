from __future__ import division
import copy
import hashlib
import logging
import numpy as np
import dolfin as df
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from scipy.optimize import minimize_scalar
from custom_exceptions import EigenproblemVerifyError
from types import NoneType

logger = logging.getLogger("finmag")


def iseven(n):
    """
    Return True if n is an even number and False otherwise.
    """
    return (n % 2 == 0)


def is_hermitian(A, rtol=1e-5, atol=1e-8):
    if not isinstance(A, np.ndarray):
        logger.warning(
            "Converting sparse matrix A to dense array to check whether it is "
            "Hermitian. This might consume a lot of memory if A is big!.")
        A = as_dense_array(A)
    return np.allclose(A, np.conj(A.T), rtol=rtol, atol=atol)


def make_human_readable(nbytes):
    """
    Given a number of bytes, return a string of the form "12.2 MB" or "3.44 GB"
    which makes the number more digestible by a human reader. Everything less
    than 500 MB will be displayed in units of MB, everything above in units of GB.
    """
    if nbytes < 500*1024**2:
        res = '{:.2f} MB'.format(nbytes / 1024**2)
    else:
        res = '{:.2f} GB'.format(nbytes / 1024**3)
    return res


def print_eigenproblem_memory_usage(num_mesh_nodes, generalised=False):
    """
    Given the number of nodes in a mesh, print the amount of memory
    that the eigenproblem matrix or matrices (in case of a generalised
    eigenproblem) will occupy in memory. This is useful when treating
    very big problems in order to "interactively" adjust a mesh until
    the matrix fits in memory.

    """
    N = num_mesh_nodes
    if generalised == False:
        byte_size_float = np.zeros(1, dtype=float).nbytes
        memory_usage = (2*N)**2 * byte_size_float
    else:
        byte_size_complex = np.zeros(1, dtype=complex).nbytes
        memory_usage = 2 * (2*N)**2 * byte_size_complex

    logger.debug("Eigenproblem {} (of {}generalised eigenproblem) "
                 "for mesh with {} vertices will occupy {} "
                 "in memory.".format(
                     'matrices' if generalised else 'matrix',
                     '' if generalised else 'non-',
                     N, make_human_readable(memory_usage)))


def compute_relative_error(A, M, omega, w):
    if not isinstance(A, np.ndarray) or not isinstance(M, (np.ndarray, NoneType)):
        logger.warning(
            "Converting sparse matrix to numpy.array as this is the only "
            "supported matrix type at the moment for computing relative errors.")
        A = as_dense_array(A)
        M = as_dense_array(M)
    lhs = np.dot(A, w)
    rhs = omega*w if (M == None) else omega*np.dot(M, w)
    rel_err = np.linalg.norm(lhs - rhs) / np.linalg.norm(omega*w)
    return rel_err


def normalise_if_not_zero(v, threshold=1e-12):
    v_norm = np.linalg.norm(v)
    if v_norm > threshold:
        return v / v_norm
    else:
        return v


def normalise_rows(A, threshold=1e-12):
    """
    Return a copy of the matrix `A` where each row vector is
    normalised so that its absolute value is 1.

    If the norm of a row vector is below `threshold` then the vector is
    copied over without being normalised first (this is to ensure that
    rows with very small entries are effectively treated as if they
    contained all zeros).

    """
    B = A.copy()
    for i, v in enumerate(A):
        a = np.linalg.norm(v)
        if a > threshold:
            B[i] /= np.linalg.norm(v)
    return B


def is_diagonal_matrix(A):
    """
    Return `True` if A is a diagonal matrix and False otherwise.

    """
    return np.allclose(A, np.diag(np.diag(A)))


def is_scalar_multiple(v, w, tol=1e-12):
    """
    Return `True` if the numpy.array `w` is a scalar multiple of the
    numpy.array `v` (and `False` otherwise).
    """
    v_colvec = v.reshape(-1, 1)
    w_colvec = w.reshape(-1, 1)
    _, residuals, _, _ = np.linalg.lstsq(v_colvec, w_colvec)
    assert(len(residuals) == 1)
    rel_err = residuals[0] / np.linalg.norm(v)
    return (rel_err < tol)
    # a, _, _, _ = np.linalg.lstsq(v_colvec, w_colvec)
    # return np.allclose(v, a*w, atol=tol, rtol=tol)

def is_matching_eigenpair(pair1, pair2, tol_eigenval=1e-8, tol_eigenvec=1e-6):
    """
    Given two eigenpairs `pair1 = (omega1, w1)` and `pair2 = (omega2, w2)`,
    return True if `omega1` and `omega2` coincide (within `tol_eigenval`)
    and `w1` is a scalar multiple of `w2` (within `tol_eigenvec`).
    """
    omega1, w1 = pair1
    omega2, w2 = pair2
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    eigenvals_coincide = np.isclose(omega1, omega2, atol=tol_eigenval, rtol=tol_eigenval)
    eigenvecs_coincide = is_scalar_multiple(w1, w2, tol=tol_eigenvec)
    return eigenvals_coincide and eigenvecs_coincide


def find_matching_eigenpair((omega, w), ref_eigenpairs,
                            tol_eigenval=1e-8, tol_eigenvec=1e-6):
    """
    Given a pair `(omega, w)` consisting of a computed eigenvalue and
    eigenvector, check whether any of the eiganpairs in `ref_eigenpairs`
    match the pair `(omega, w)` in the sense that the eigenvalues
    coincide (up to `tolerance_eigenval`) and the eigenvectors are
    linearly dependent (up to `tolerance_eigenvec`).

    """
    matching_indices = \
        [i
         for (i, pair_ref) in enumerate(ref_eigenpairs)
         if is_matching_eigenpair((omega, w), pair_ref,
                                  tol_eigenval=tol_eigenval,
                                  tol_eigenvec=tol_eigenvec)]

    if len(matching_indices) >= 2:
        raise EigenproblemVerifyError("Found more than one matching eigenpair.")
    elif matching_indices == []:
        return None
    else:
        return matching_indices[0]


def std_basis_vector(i, N, dtype=None):
    """
    Return the `i`-th standard basis vector of length `N`, where `i`
    runs from 1 through N.

    Examples:

       std_basis_vector(2, 4) = [0, 1, 0, 0]
       std_basis_vector(1, 6) = [1, 0, 0, 0, 0, 0]
       std_basis_vector(2, 6) = [0, 1, 0, 0, 0, 0]
       std_basis_vector(6, 6) = [0, 0, 0, 0, 0, 1]

    """
    v = np.zeros(N, dtype=dtype)
    v[i-1] = 1.0
    return v


def sort_eigensolutions(eigvals, eigvecs):
    """
    Sort the lists of eigenvalues and eigenvalues in ascending order
    of the eigenvalues.

    """
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)

    sort_indices = abs(eigvals).argsort()
    eigvals = eigvals[sort_indices]
    eigvecs = eigvecs[sort_indices]

    return eigvals, eigvecs


def best_linear_combination(v, basis_vecs):
    """
    Given a vector `v` and a list <e_i> of basis vectors in `basis_vecs`,
    determine the coefficients b_i which minimise the residual:

        res = |v - w|

    where `w` is the vector

        w = \sum_{i} b_i * e_i

    Returns the triple (w, [b_i], res).

    """
    v = np.asarray(v)
    assert(v.ndim == 1)
    N = len(v)
    num = len(basis_vecs)
    v_colvec = np.asarray(v).reshape(-1, 1)
    basis_vecs = np.asarray(basis_vecs)
    assert(basis_vecs.shape == (num, N))
    coeffs, residuals, _, _ = np.linalg.lstsq(basis_vecs.T, v_colvec)
    assert(coeffs.shape == (num, 1))
    coeffs.shape = (num,)
    # XXX TODO: Figure out why it can happen that residuals.shape == (0,)!!
    #assert(residuals.shape == (1,))
    if residuals.shape == (0,):
        logger.warning("[DDD] Something went wrong! Assuming the residuals are zero!")
        residuals = np.array([0.0])
    w = np.dot(basis_vecs.T, coeffs).ravel()
    assert(w.shape == v.shape)
    return w, coeffs, np.sqrt(residuals[0])


def scipy_sparse_linear_operator_to_dense_array(A, dtype=complex):
    """
    Convert a sparse LinearOperator `A` to a dense array.
    This is quite inefficient and should be used for debugging only.

    """
    m, n = A.shape
    A_arr = np.zeros((m, n), dtype=dtype)

    # Multiply A by all the standard basis vectors and accumulate the results.
    for j, e_j in enumerate(np.eye(n)):
        v = A.matvec(e_j)
        if dtype == float:
            if not(all(v.imag == 0.0)):
                raise ValueError("Cannot cast complex vector into real array.")
            else:
                v = v.real
        A_arr[:, j] = v

    return A_arr


def petsc_matrix_to_numpy_array(A, dtype=float):
    # XXX TODO: Move this import to the top once we have found an easy
    #           and reliable (semi-)automatic way for users to install
    #           petsc4py.  -- Max, 20.3.2014
    from petsc4py import PETSc

    if not isinstance(A, PETSc.Mat):
        raise TypeError("Matrix must be of type petsc4py.PETSc.Mat, "
                        "but got: {}".format(type(A)))
    indptr, indices, data = A.getValuesCSR()
    data = np.asarray(data, dtype=dtype)
    A_csr = csr_matrix((data, indices, indptr), shape=A.size)
    return A_csr.todense()


def as_dense_array(A, dtype=None):
    # XXX TODO: Move this import to the top once we have found an easy
    #           and reliable (semi-)automatic way for users to install
    #           petsc4py.  -- Max, 20.3.2014
    from petsc4py import PETSc

    if A == None:
        return None

    if isinstance(A, np.ndarray):
        return np.asarray(A, dtype=dtype)

    if dtype == None:
        # TODO: Do we have a better option than using 'complex' by default?
        dtype = complex

    if isinstance(A, LinearOperator):
        return scipy_sparse_linear_operator_to_dense_array(A, dtype=dtype)
    elif isinstance(A, PETSc.Mat):
        return petsc_matrix_to_numpy_array(A, dtype=dtype)
        #raise NotImplementedError()
    elif isinstance(A, df.PETScMatrix):
        #return petsc_matrix_to_dense_array(A.mat(), dtype=dtype)
        raise NotImplementedError()
    else:
        raise TypeError(
            "Matrix must be either a scipy LinearOperator, a dolfin "
            "PETScMatrix or a petsc4py matrix. Got: {}".format(type(A)))


def as_petsc_matrix(A):
    """
    Return a (sparse) matrix of type `petsc4py.PETSc.Mat` containing
    the same entries as the `numpy.array` `A`. The returned matrix
    will be as sparse as possible (i.e. only the non-zero entries of A
    will be set).

    """
    # XXX TODO: Move this import to the top once we have found an easy
    #           and reliable (semi-)automatic way for users to install
    #           petsc4py.  -- Max, 20.3.2014
    from petsc4py import PETSc

    if isinstance(A, PETSc.Mat):
        return A

    m, n = A.shape

    if isinstance(A, np.ndarray):
        def get_jth_column(j):
            return A[:, j]
    elif isinstance(A, LinearOperator):
        def get_jth_column(j):
            e_j = np.zeros(m)
            e_j[j] = 1.0
            return A.matvec(e_j)
    else:
        raise TypeError("Unkown matrix type: {}".format(type(A)))

    A_petsc = PETSc.Mat().create()
    A_petsc.setSizes([m, n])
    A_petsc.setType('aij')  # sparse
    A_petsc.setUp()

    for j in xrange(0, n):
        col = get_jth_column(j)
        if col.dtype == complex:
            if np.allclose(col.imag, 0.0):
                col = col.real
            else:
                raise TypeError("Array with complex entries cannot be converted "
                                "to a PETSc matrix.")

        for i in xrange(0, m):
            # We try to keep A_petsc as sparse as possible by only
            # setting nonzero entries.
            if col[i] != 0.0:
                A_petsc[i, j] = col[i]
    A_petsc.assemble()
    return A_petsc


def irregular_interval_mesh(xmin, xmax, n):
    """
    Create a mesh on the interval [xmin, xmax] with n vertices.
    The first and last mesh node coincide with xmin and xmax,
    but the other nodes are picked at random from within the
    interval.

    """
    # Create an 'empty' mesh
    mesh = df.Mesh()

    # Open it in the MeshEditor as a mesh of topological and geometrical dimension 1
    editor = df.MeshEditor()
    editor.open(mesh, 1, 1)

    # We're going to define 5 vertices, which result in 4 'cells' (= intervals)
    editor.init_vertices(n)
    editor.init_cells(n-1)

    coords = (xmax - xmin) * np.random.random_sample(n-1) + xmin
    coords.sort(0)

    # Define the vertices and cells with their corresponding indices
    for (i, x) in enumerate(coords):
        editor.add_vertex(i, np.array([x], dtype=float))
    editor.add_vertex(n-1, np.array([xmax], dtype=float))
    for i in xrange(n-1):
        editor.add_cell(i, np.array([i, i+1], dtype='uintp'))

    editor.close()
    return mesh
