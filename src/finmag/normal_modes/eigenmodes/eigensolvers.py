from __future__ import division
import numpy as np
import dolfin as df
import scipy.linalg
import scipy.sparse.linalg
import logging
from helpers import sort_eigensolutions, as_petsc_matrix, is_hermitian, compute_relative_error, as_dense_array
from types import NoneType

logger = logging.getLogger("finmag")


class AbstractEigensolver(object):
    def __repr__(self):
        return "<{}{}>".format(self.__class__.__name__, self._extra_info())

    def _extra_info(self):
        return ""

    def is_hermitian(self):
        """
        Return True if the solver can only solver Hermitian problems.
        """
        return False  # by default, solvers are non-Hermitian

    def _solve_eigenproblem(self, A, M=None, num=None, tol=None):
        # This function should be overridden by the concrete implementations
        raise NotImplementedError("Please use one of the concrete "
                                  "eigensolvers, not the abstract one.")

    def solve_eigenproblem(self, A, M=None, num=None, tol=None):
        """
        Solve the (possibly generalised) eigenvalue problem defined by

           A*v = omega*M*v

        If `M` is `None`, it uses the identity matrix for `M`.


        *Arguments*

        A:  np.array

            The matrix on the left-hand side of the eigenvalue problem.

        M:  np.array | None

            The matrix on the right-hand side of the generalised
            eigenvalue problem. Assumes the identity matrix if not
            given.

        num :  int

            If given, limit the number of returned eigenvalues and
            eigenvectors to at most `num`. Default: `None`.

        tol :  float

            The tolerance for the computed eigensolutions (TODO: what
            exactly does this mean?!? Relative or absolute?!?). The
            meaning depends on the individual solver. Note that not
            all solvers may use/respect this argument (for example,
            the dense Scipy solvers don't). The default is None, which
            means that whatever the solver's default is will be used.


        *Returns*

        A pair `(omega, w)` where `omega` is the list of eigenvalues
        and `w` the list of corresponding eigenvectors. Both lists are
        sorted in in ascending order of the eigenvalues.

        """
        eigenproblem_is_hermitian = is_hermitian(A) and (M == None or is_hermitian(M))
        if self.is_hermitian() and not eigenproblem_is_hermitian:
            raise ValueError("Eigenproblem matrices are non-Hermitian but solver "
                             "assumes Hermitian matrices. Aborting.")
        logger.info("Solving eigenproblem. This may take a while...")
        df.tic()
        omegas, ws = self._solve_eigenproblem(A, M=M, num=num, tol=tol)
        logger.debug("Computing the eigenvalues and eigenvectors took {:.2f} seconds".format(df.toc()))

        # XXX TODO: Remove this conversion to numpy.arrays once we
        #           have better support for different kinds of
        #           matrices (but the conversion would happen in
        #           compute_relative_error() anyway, so by doing it
        #           here we avoid doing it multiple times.
        if not isinstance(A, np.ndarray):
            logger.warning(
                "Converting sparse matrix A to dense array to check whether it is "
                "Hermitian. This might consume a lot of memory if A is big!.")
            A = as_dense_array(A)
        if not isinstance(M, (np.ndarray, NoneType)):
            logger.warning(
                "Converting sparse matrix M to dense array to check whether it is "
                "Hermitian. This might consume a lot of memory if M is big!.")
            M = as_dense_array(M)

        rel_errors = np.array([compute_relative_error(A, M, omega, w) for omega, w in zip(omegas, ws)])
        return omegas, ws, rel_errors


#
# Dense solvers from scipy.linalg
#
class ScipyDenseSolver(AbstractEigensolver):
    _solver_func = None   # needs to be instantiated by derived classes

    def _solve_eigenproblem(self, A, M=None, num=None, tol=None):
        A = as_dense_array(A)  # does nothing if A is already a numpy.array
        M = as_dense_array(M)  # does nothing if M is already a numpy.array
        omega, w = self._solver_func(A, M)
        w = w.T  # make sure that eigenvectors are stored in rows, not columns
        omega, w = sort_eigensolutions(omega, w)

        # Return only the number of requested eigenvalues
        N, _ = A.shape
        num = num or self.num
        num = min(num, N-1)
        if num != None:
            omega = omega[:num]
            w = w[:num]

        return omega, w


class ScipyLinalgEig(ScipyDenseSolver):
    def __init__(self, num=None):
        self.num = num
        self._solver_func = scipy.linalg.eig


class ScipyLinalgEigh(ScipyDenseSolver):
    def __init__(self, num=None):
        self.num = num
        self._solver_func = scipy.linalg.eigh

    def is_hermitian(self):
        return True


#
# Sparse solvers from scipy.sparse.linalg
#
class ScipySparseSolver(AbstractEigensolver):
    _solver_func = None  # needs to be instantiated by derived classes

    def __init__(self, sigma, which, num=6, swap_matrices=False, tol=None):
        """
        *Arguments*

        sigma:

            If given, find eigenvalues near sigma using shift-invert mode.

        which:

            str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
            Which `k` eigenvectors and eigenvalues to find:

            'LM' : largest magnitude

            'SM' : smallest magnitude

            'LR' : largest real part

            'SR' : smallest real part

            'LI' : largest imaginary part

            'SI' : smallest imaginary part

            When sigma != None, 'which' refers to the shifted eigenvalues w'[i]
            (see discussion in 'sigma', above).  ARPACK is generally better at
            finding large values than small values.  If small eigenvalues are
            desired, consider using shift-invert mode for better performance.

        num:  int

            The number of eigenvalues to compute (computes all if not
            given). Must be provided for the sparse solvers.

        """
        self.sigma = sigma
        self.which = which
        self.num = num
        self.swap_matrices = swap_matrices
        self.tol = tol or 0.  # Scipy's default is 0.0

    def _extra_info(self):
        return ": sigma={}, which='{}', num={}".format(
            self.sigma, self.which, self.num)

    def _solve_eigenproblem(self, A, M=None, num=None, tol=None):
        N, _ = A.shape
        num = num or self.num
        num = min(num, N-2)
        tol = tol or self.tol

        if self.swap_matrices:
            if M is None:
                M = id_op(A)
            A, M = M, A

        # Compute eigensolutions
        omega, w = self._solver_func(A, k=num, M=M,
                                     sigma=self.sigma, which=self.which,
                                     tol=tol)
        w = w.T  # make sure that eigenvectors are stored in rows, not columns

        return sort_eigensolutions(omega, w)


def id_op(A):
    return scipy.sparse.linalg.LinearOperator(
        shape=A.shape, matvec=(lambda v: v), dtype=A.dtype)


class ScipySparseLinalgEigs(ScipySparseSolver):
    def __init__(self, *args, **kwargs):
        super(ScipySparseLinalgEigs, self).__init__(*args, **kwargs)
        self._solver_func = scipy.sparse.linalg.eigs


class ScipySparseLinalgEigsh(ScipySparseSolver):
    def __init__(self, *args, **kwargs):
        super(ScipySparseLinalgEigsh, self).__init__(*args, **kwargs)
        self._solver_func = scipy.sparse.linalg.eigsh

    def is_hermitian(self):
        return True


class SLEPcEigensolver(AbstractEigensolver):
    def __init__(self, problem_type=None, method_type=None, which=None, num=6,
                 tol=1e-12, maxit=100, shift_invert=False, swap_matrices=False, verbose=True):
        """
        *Arguments*

        problem_type:  str

            A string describing the problem type. Must be one of the types
            defined in SLEPc.EPS.ProblemType:

            - `HEP`:    Hermitian eigenproblem.
            - `NHEP`:   Non-Hermitian eigenproblem.
            - `GHEP`:   Generalized Hermitian eigenproblem.
            - `GNHEP`:  Generalized Non-Hermitian eigenproblem.
            - `PGNHEP`: Generalized Non-Hermitian eigenproblem
                        with positive definite ``B``.
            - `GHIEP`:  Generalized Hermitian-indefinite eigenproblem.


        method_type:  str

            A string describing the method used for solving the eigenproblem.
            Must be one of the types defined in SLEPc.EPS.Type:

            - `POWER`:        Power Iteration, Inverse Iteration, RQI.
            - `SUBSPACE`:     Subspace Iteration.
            - `ARNOLDI`:      Arnoldi.
            - `LANCZOS`:      Lanczos.
            - `KRYLOVSCHUR`:  Krylov-Schur (default).
            - `GD`:           Generalized Davidson.
            - `JD`:           Jacobi-Davidson.
            - `RQCG`:         Rayleigh Quotient Conjugate Gradient.
            - `LAPACK`:       Wrappers to dense eigensolvers in Lapack.


        which:  str

            A string describing which piece of the spectrum to compute.
            Must be one of the options defined in SLEPc.EPS.Which:

            - `LARGEST_MAGNITUDE`:  Largest magnitude (default).
            - `LARGEST_REAL`:       Largest real parts.
            - `LARGEST_IMAGINARY`:  Largest imaginary parts in magnitude.
            - `SMALLEST_MAGNITUDE`: Smallest magnitude.
            - `SMALLEST_REAL`:      Smallest real parts.
            - `SMALLEST_IMAGINARY`: Smallest imaginary parts in magnitude.
            - `TARGET_MAGNITUDE`:   Closest to target (in magnitude).
            - `TARGET_REAL`:        Real part closest to target.
            - `TARGET_IMAGINARY`:   Imaginary part closest to target.
            - `ALL`:                All eigenvalues in an interval.
            - `USER`:               User defined ordering.

            TODO: Note that `USER` is note supported yet(?!?!?).


        num:  int

            The number of eigenvalues to compute.


        tol:  float

            The solver tolerance.


        maxit:  num

            The maximum number of iterations.

        """
        self.problem_type = problem_type  # string describing the problem type
        self.method_type = method_type  # string describing the solution method
        self.which = which
        self.num = num
        self.tol = tol
        self.shift_invert = shift_invert
        self.swap_matrices = swap_matrices
        self.maxit = maxit
        self.verbose = verbose

    def _extra_info(self):
        return ": {}, {}, {}, num={}, tol={:g}, maxit={}".format(
            self.problem_type, self.method_type, self.which,
            self.num, self.tol, self.maxit)

    def is_hermitian(self):
        return self.problem_type in ['HEP', 'GHEP', 'GHIEP']

    def _create_eigenproblem_solver(self, A, M, num, problem_type, method_type, which, tol, maxit, shift_invert):
        """
        Create a SLEPc eigenproblem solver with the operator
        """
        # XXX TODO: This import should actually happen at the top, but on some
        #           systems it seems to be slightly non-trivial to install
        #           slepc4py, and since we don't use it for the default eigen-
        #           value methods, it's better to avoid raising an ImportError
        #           which forces users to try and install it.  -- Max, 20.3.2014
        from slepc4py import SLEPc

        E = SLEPc.EPS()
        E.create()
        E.setOperators(A, M)
        E.setProblemType(getattr(SLEPc.EPS.ProblemType, problem_type))
        E.setType(getattr(SLEPc.EPS.Type, method_type))
        E.setWhichEigenpairs(getattr(SLEPc.EPS.Which, which))
        E.setDimensions(nev=num)
        E.setTolerances(tol, maxit)
        if shift_invert == True:
            st = E.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            st.setShift(0.0)
        return E

    def _solve_eigenproblem(self, A, M=None, num=None, problem_type=None, method_type=None, which=None, tol=1e-12, maxit=100, swap_matrices=None, shift_invert=None):
        num = num or self.num
        problem_type = problem_type or self.problem_type
        method_type = method_type or self.method_type
        which = which or self.which
        tol = tol or self.tol
        maxit = maxit or self.maxit
        if problem_type == None:
            raise ValueError("No problem type specified for SLEPcEigensolver.")
        if method_type == None:
            raise ValueError("No solution method specified for SLEPcEigensolver.")
        if which == None:
            raise ValueError("Please specify which eigenvalues to compute.")
        if swap_matrices == None:
            swap_matrices = self.swap_matrices
        if shift_invert == None:
            shift_invert = self.shift_invert

        A_petsc = as_petsc_matrix(A)
        M_petsc = None if (M == None) else as_petsc_matrix(M)
        if swap_matrices:
            A_petsc, M_petsc = M_petsc, A_petsc
        size, _ = A_petsc.size

        E = self._create_eigenproblem_solver(
            A=A_petsc, M=M_petsc, num=num, problem_type=problem_type,
            method_type=method_type, which=which, tol=tol, maxit=maxit, shift_invert=shift_invert)
        E.solve()

        its = E.getIterationNumber()
        eps_type = E.getType()
        nev, ncv, mpd = E.getDimensions()
        tol, maxit = E.getTolerances()
        st_type = E.getST().getType()
        nconv = E.getConverged()
        if nconv < num:
            # XXX TODO: Use a more specific error!
            raise RuntimeError("Not all requested eigenpairs converged: "
                               "{}/{}.".format(nconv, num))

        if self.verbose == True:
            print("")
            print("******************************")
            print("*** SLEPc Solution Results ***")
            print("******************************")
            print("")
            print("Number of iterations of the method: %d" % its)
            print("Solution method: %s" % eps_type)
            print("Spectral Transformation type: %s" % st_type)
            print("Number of requested eigenvalues: %d" % nev)
            print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
            print("Number of converged eigenpairs: %d" % nconv)

        if nconv > 0:
            # Create the results vectors
            vr, wr = A_petsc.getVecs()
            vi, wi = A_petsc.getVecs()
            if self.verbose:
                print("")
                print("        k          ||Ax-kx||/||kx|| ")
                print("----------------- ------------------")
            for i in range(nconv):
                k = E.getEigenpair(i, vr, vi)
                error = E.computeRelativeError(i)
                if self.verbose:
                    if k.imag != 0.0:
                        print(" %9f%+9f j %12g" % (k.real, k.imag, error))
                    else:
                        print(" %12f      %12g" % (k.real, error))
            if self.verbose:
                print("")

        omegas = []
        ws = []
        for i in xrange(nconv):
            omega = E.getEigenpair(i, vr, vi)
            vr_arr = vr.getValues(range(size))
            vi_arr = vi.getValues(range(size))
            if omega.imag == 0.0:
                omegas.append(omega.real)
            else:
                omegas.append(omega)
            if np.all(vi_arr == 0.0):
                ws.append(vr_arr)
            else:
                ws.append(vr_arr + 1j*vi_arr)
        omegas = np.array(omegas)
        ws = np.array(ws)
        logger.warning("TODO: Check that the eigensolutions returned by SLEPc are sorted.")
        return omegas[:num], ws[:num]


# List of all available eigensolvers
available_eigensolvers = [ScipyLinalgEig, ScipyLinalgEigh,
                          ScipySparseLinalgEigs, ScipySparseLinalgEigsh,
                          SLEPcEigensolver,
                          ]
