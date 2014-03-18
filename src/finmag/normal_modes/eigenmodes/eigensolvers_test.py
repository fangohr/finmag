from __future__ import division
import pytest
import numpy as np
import itertools
from eigensolvers import *
from eigenproblems import *
from helpers import normalise_rows
np.set_printoptions(precision=3)


# NOTE: Don't change this list without adapting the "known_failures" below!
sample_eigensolvers = [
    ScipyLinalgEig(),
    ScipyLinalgEigh(),
    ScipySparseLinalgEigs(sigma=0.0, which='LM', num=10),
    ScipySparseLinalgEigsh(sigma=0.0, which='LM', num=10),
    SLEPcEigensolver(problem_type='GNHEP', method_type='KRYLOVSCHUR',
                     which='SMALLEST_MAGNITUDE'),
    ]


def test_str():
    """
    Test the __str__ method for each of the `sample_eigensolvers`.

    """
    assert(set([str(s) for s in sample_eigensolvers]) ==
           set(["<ScipyLinalgEig>",
                "<ScipyLinalgEigh>",
                "<ScipySparseLinalgEigs: sigma=0.0, which='LM', num=10>",
                "<ScipySparseLinalgEigsh: sigma=0.0, which='LM', num=10>",
                "<SLEPcEigensolver: GNHEP, KRYLOVSCHUR, SMALLEST_MAGNITUDE, num=6, tol=1e-12, maxit=100>",
               ]))


def test_scipy_dense_solvers_num_argument():
    """
    By default, the Scipy dense solvers compute all eigenpairs of the
    eigenproblem. We can limit the number of solutions returned by
    specifying the 'num' argument, either directly when instantiating
    the solver or when calling the eigenproblem.solve() method. This
    test checks that the expected number of solutions is returned in
    all cases.

    """
    diagproblem = DiagonalEigenproblem()

    # Dense solvers should yield all solutions by default
    solver1 = ScipyLinalgEig()
    solver2 = ScipyLinalgEigh()
    omega1, w1, _ = diagproblem.solve(solver1, N=10, dtype=float)
    omega2, w2, _  = diagproblem.solve(solver2, N=10, dtype=float)
    assert(len(omega1) == 10)
    assert(len(omega2) == 10)
    assert(w1.shape == (10, 10))
    assert(w2.shape == (10, 10))

    # By initialising them with the 'num' argument they should only
    # yield the first 'num' solutions.
    solver3 = ScipyLinalgEig(num=5)
    solver4 = ScipyLinalgEigh(num=3)
    omega3, w3, _ = diagproblem.solve(solver3, N=10, dtype=float)
    omega4, w4, _ = diagproblem.solve(solver4, N=10, dtype=float)
    assert(len(omega3) == 5)
    assert(len(omega4) == 3)
    assert(w3.shape == (5, 10))
    assert(w4.shape == (3, 10))

    # We can also provide the 'num' argument in the solve() method
    # directly. If provided, this argument should be used. Otherwise
    # The fallback value from the solver initialisation is used.
    solver5 = ScipyLinalgEig(num=7)
    solver6 = ScipyLinalgEig()
    solver7 = ScipyLinalgEigh(num=3)
    omega5, w5, _ = diagproblem.solve(solver5, N=10, num=5, dtype=float)
    omega6, w6, _ = diagproblem.solve(solver6, N=10, num=8, dtype=float)
    omega7, w7, _ = diagproblem.solve(solver7, N=10, num=6, dtype=float)
    omega8, w8, _ = diagproblem.solve(solver7, N=10, num=None, dtype=float)
    assert(len(omega5) == 5)
    assert(len(omega6) == 8)
    assert(len(omega7) == 6)
    assert(len(omega8) == 3)
    assert(w5.shape == (5, 10))
    assert(w6.shape == (8, 10))
    assert(w7.shape == (6, 10))
    assert(w8.shape == (3, 10))


def test_scipy_sparse_solvers_num_argument():
    """
    We can limit the number of solutions returned by the sparse
    solvers specifying the 'num' argument, either directly when
    instantiating the solver or when calling the eigenproblem.solve()
    method. This test checks that the expected number of solutions is
    returned in all cases.

    """
    diagproblem = DiagonalEigenproblem()

    # By initialising them with the 'num' argument they should only
    # yield the first 'num' solutions.
    solver1 = ScipySparseLinalgEigs(sigma=0.0, which='LM', num=4)
    solver2 = ScipySparseLinalgEigsh(sigma=0.0, which='LM', num=3)
    omega1, w1, _ = diagproblem.solve(solver1, N=10, dtype=float)
    omega2, w2, _ = diagproblem.solve(solver2, N=10, dtype=float)
    assert(len(omega1) == 4)
    assert(len(omega2) == 3)
    assert(w1.shape == (4, 10))
    assert(w2.shape == (3, 10))

    # We can also provide the 'num' argument in the solve() method
    # directly. If provided, this argument should be used. Otherwise
    # The fallback value from the solver initialisation is used.
    solver3 = ScipySparseLinalgEigs(sigma=0.0, which='LM', num=7)
    solver4 = ScipySparseLinalgEigsh(sigma=0.0, which='LM', num=3)
    omega3, w3, _ = diagproblem.solve(solver3, N=10, num=5, dtype=float)
    omega4, w4, _ = diagproblem.solve(solver4, N=10, num=6, dtype=float)
    omega5, w5, _ = diagproblem.solve(solver4, N=10, num=None, dtype=float)
    assert(len(omega3) == 5)
    assert(len(omega4) == 6)
    assert(len(omega5) == 3)
    assert(w3.shape == (5, 10))
    assert(w4.shape == (6, 10))
    assert(w5.shape == (3, 10))


@pytest.mark.parametrize("dtype, solver",
                         itertools.product([float, complex],
                                           sample_eigensolvers))
def test_compute_eigenvalues_of_diagonal_matrix(dtype, solver):
    N = 60

    diagvals = np.arange(1, N+1, dtype=dtype)
    A = np.diag(diagvals)

    print("[DDD] Testing eigensolver {} with diagonal matrix of size "
          "N={}, dtype={}".format(solver, N, dtype))
    omega, w, _ = solver.solve_eigenproblem(A)
    print("[DDD]    omega: {}".format(omega))

    # Compute expected eigenvalues and eigenvectors
    k = len(omega)
    omega_ref = diagvals[:k]
    w_ref = np.eye(N)[:k]

    assert(np.allclose(omega, omega_ref))
    assert(np.allclose(abs(normalise_rows(w)), w_ref))


# # The following test illustrates a strange failure of the scipy
# # sparse solver to compute the correct eigenvalues for a diagonal
# # matrix. The largest computed eigenvalue is 11 instead of the
# # expected value 10. However, I can't reproduce this at the moment
# # so the test is commented out.
# #
# def test_weird_wrong_computation():
#     #solver = ScipySparseLinalgEigsh(sigma=0.0, which='LM', num=10)
#     solver = sample_eigensolvers[2]
#     diagproblem = DiagonalEigenproblem()
#     omega, _ = diagproblem.solve(solver, N=50, dtype=float)
#     omega_wrong = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
#     assert(np.allclose(omega, omega_wrong))


# We cannot use a Hermitian eigensolver to solve a non-Hermitian problem
def solver_and_problem_are_compatible(solver, eigenproblem):
    return not(solver.is_hermitian() and not eigenproblem.is_hermitian())

# Create a set of fixtures where eigensolvers and compatible
# eigenproblems are paired.
fixtures = itertools.product(sample_eigensolvers, available_eigenproblems)


# TODO: Currently we simply skip these and call pytest.xfail directly.
#       It would be better to actually execute them and wait for them
#       to fail, so that the failures are confirmed. However, this
#       might take a long time (due to missing convergence), so we're
#       not doing it at the moment. Perhaps we can mark the tests as
#       'long_running' and skip them during normal runs?
# Note: On the other hand, some of these pass on certain computers and
#       fail on others. So it might be better to keep this as 'blacklist',
#       and document the known failures in the test 'test_document_failures'
#       below.
known_failures = \
    [### The first case fails on omicron but passes on hathor...
     (sample_eigensolvers[2], # ScipySparseLinalgEigs
      available_eigenproblems[2], # Nanostrip1d
      200,
      float),
     (sample_eigensolvers[4], # SLEPc
      available_eigenproblems[1], # RingGraphLaplace
      101,
      float),
     (sample_eigensolvers[4], # SLEPc
      available_eigenproblems[2], # Nanostrip1d
      200,
      float),
     (sample_eigensolvers[4], # SLEPc
      available_eigenproblems[2], # Nanostrip1d
      200,
      complex),
     ]


@pytest.mark.parametrize("solver, eigenproblem", fixtures)
def test_eigensolvers(solver, eigenproblem):
    print("\n[DDD] solver: {}".format(solver))
    print("[DDD] eigenproblem: {}".format(eigenproblem))
    for N in [50, 101, 200]:
        for dtype in [float, complex]:
            if (solver, eigenproblem, N, dtype) in known_failures:
                pytest.xfail("Known failure: {}, {}, {}, {}".format(
                        solver, eigenproblem, N, dtype))
            print("[DDD] N={}, dtype={}".format(N, dtype))

            if not solver_and_problem_are_compatible(solver, eigenproblem):
                with pytest.raises(ValueError):
                    eigenproblem.solve(solver, N, dtype=dtype, num=40)
                continue

            # Nanostrip1d can only solve problems of even size
            if not iseven(N) and isinstance(eigenproblem, Nanostrip1dEigenproblemFinmag):
                with pytest.raises(ValueError):
                    eigenproblem.solve(solver, N, dtype=dtype, num=40)
                continue

            omega, w, _ = eigenproblem.solve(solver, N, dtype=dtype, num=40)
            print "[DDD] len(omega): {}".format(len(omega))
            try:
                if isinstance(eigenproblem, Nanostrip1dEigenproblemFinmag):
                    # The Nanostrip1d seems to be quite ill-behaved
                    # with the sparse solvers.
                    tol_eigval = 0.1
                elif isinstance(solver, ScipySparseSolver):
                    # The sparse solvers seem to be less accurate, so
                    # we use a less strict tolerance.
                    tol_eigval = 1e-3
                elif isinstance(solver, SLEPcEigensolver):
                    tol_eigval = 1e-13
                else:
                    tol_eigval = 1e-14
                eigenproblem.verify_eigenpairs_numerically(zip(omega, w))
                eigenproblem.verify_eigenpairs_analytically(zip(omega, w), tol_eigval=tol_eigval)
            except NotImplementedError:
                pytest.xfail("Analytical solution not implemented for "
                             "solver {}".format(solver))


# XXX TODO: The following test illustrates a spurious problem for N=100, where
#           the test hangs because the iterations don't finish. We also get the
#           following RuntimeWarning:
#
# /usr/local/lib/python2.7/dist-packages/scipy/linalg/decomp_lu.py:71:
#     RuntimeWarning: Diagonal number 100 is exactly zero. Singular matrix.
#
# We skip the test but keep it to illustrate the problem
@pytest.mark.skipif('True')
def test_strange_degenerate_case():
    eigenproblem = RingGraphLaplaceEigenproblem()
    solver = ScipySparseLinalgEigs(sigma=0.0, which='LM', num=10)
    omega, w, _ = eigenproblem.solve(solver, N=100, dtype=float)


def test_document_failures():
    """
    This test documents some cases where the eigensolvers fail. This
    is valuable if I want to write these results up later.
    """

    # The RingGraphLaplaceEigenproblem seems to be very ill-conditioned.
    # Strangely, some values of N seem to be more susceptible to failures
    # than others...
    # TODO: This is duplicated in known_failures, but there it's not currently
    #       executed, so we run it here instead.
    solver = SLEPcEigensolver(problem_type='GNHEP', method_type='KRYLOVSCHUR',
                              which='SMALLEST_MAGNITUDE')
    eigenproblem = RingGraphLaplaceEigenproblem()
    N = 101
    num = 40
    for dtype in [float, complex]:
        with pytest.raises(RuntimeError):
            eigenproblem.solve(solver, N, dtype=dtype, num=num)

    # It seems that this SLEPcEigensolver can't cope with problems where N is
    # too large, so it doesn't converge.
    solver = SLEPcEigensolver(problem_type='GNHEP', method_type='KRYLOVSCHUR',
                              which='SMALLEST_MAGNITUDE')
    eigenproblem = Nanostrip1dEigenproblemFinmag(13e-12, 8e5, 0, 100)
    N = 200
    num = 40
    for dtype in [float, complex]:
        with pytest.raises(RuntimeError):
            eigenproblem.solve(solver, N, dtype=dtype, num=num)
