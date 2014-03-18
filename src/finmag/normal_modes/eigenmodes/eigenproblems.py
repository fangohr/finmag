from __future__ import division
import numpy as np
import dolfin as df
import itertools
import inspect
import logging
import sys
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from finmag.energies import Exchange
from finmag.util.consts import gamma, mu0
from math import pi
from helpers import normalise_rows, find_matching_eigenpair, \
    std_basis_vector, best_linear_combination, as_dense_array, \
    irregular_interval_mesh, iseven
from custom_exceptions import EigenproblemVerifyError

color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


class AbstractEigenproblem(object):
    def __init__(self):
        pass

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    def _raise_not_implemented_error(self, msg):
        raise NotImplementedError("No analytical {} available for eigenproblem class "
                                  "'{}'".format(msg, self.__class__.__name__))

    def is_hermitian(self):
        """
        Return True if the matrices defining the eigenproblem are Hermitian.
        """
        return False  # by default, eigenproblems are non-Hermitian

    def solve(self, solver, N, dtype, num=None, **kwargs):
        """
        Solve the (possibly generalised) eigenvalue problem defined by

           A*v = omega*M*v

        where `A` and `M` are the matrices returned by `self.instantiate(N)`.


        *Arguments*

        solver :

            The solver to use. This should be an instance of one of
            the solver classes in the `eigensolvers` module.

        N :  int

            The size of the eigenvalue problem to be solved (i.e. the
            size of the matrices `A` and `M`).

        dtype :  float | complex

            The dtype of the matrices `A` and `M`.

        num :  int

            If given, limit the number of returned eigenvalues and
            eigenvectors to at most `num` (but note that fewer may be
            returned if the solver was instantiated with a lower
            valuer of `num`). If `None` (the default), the number of
            returned solutions is limited by the properties of the
            solver.

        All other keyword arguments are passed on to the specific solver.

        *Returns*

        A triple `(omega, w, rel_errors)` where `omega` is the list of eigenvalues,
        `w` the list of corresponding eigenvectors and `rel_errors` the list of
        relative errors, defined as ||Ax-kMx||_2/||kx||_2.  All three lists are
        sorted in in ascending order of the eigenvalues.

        """
        A, M = self.instantiate(N, dtype)
        omega, w, rel_errors = solver.solve_eigenproblem(A, M=M, num=num, **kwargs)
        return omega, w, rel_errors

    def instantiate(self, N, dtype):
        """
        Return a pair of matrices (A, M) which defines a generalised
        eigenvalue problem

           A*v = omega*M*v

        The second matrix `M` may be `None` for some problems, in
        which case the generalised eigenproblem reduces to a regular
        one.

        """
        raise NotImplementedError(
            "Please choose one of the concrete eigenproblem classes.")

    def print_analytical_eigenvalues(self, num, size=None, unit='GHz'):
        if isinstance(size, str):
            raise TypeError(
                "Wrong value for argument 'size': '{}'. Did you mean to "
                "supply the 'unit' argument instead?".format(size))
        try:
            unit_factor = {'Hz': 1.0,
                           'KHz': 1e3,
                           'MHz': 1e6,
                           'GHz': 1e9,
                           }[unit]
        except KeyError:
            raise ValueError("Unknown unit: {}. Please choose one of "
                             "Hz, KHz, MHz, GHz.".format(unit))

        eigenvals = self.get_analytical_eigenvalues(num, size=size)
        for omega in eigenvals:
            print("{:.3f} {}".format(omega / unit_factor, unit))

    def get_analytical_eigenvalues(self, num, size=None):
        """
        Return a `numpy.array` of len `num` containing the first `num`
        analytical eigenvalues of this eigenproblem.

        For most eigenproblems the eigenvalues depend on the size of the
        matrix. In this case the argument `size` must be specified.
        """
        return np.array([self.get_kth_analytical_eigenvalue(k, size) for k in xrange(num)])

    def get_kth_analytical_eigenvalue(self, k, size=None):
        """
        Return the k-th analytical eigenvalue of this eigenproblem.

        For most eigenproblems the eigenvalues depend on the size of the
        matrix. In this case the argument `size` must be specified.
        """
        self._raise_not_implemented_error('eigenvalues')

    def get_analytical_eigenvectors(self, num, size):
        """
        Return a `numpy.array` of shape `(num, size)` containing the
        first `num` analytical eigenvectors of this eigenproblem.
        """
        return np.array([self.get_kth_analytical_eigenvector(k, size) for k in xrange(num)])

    def get_kth_analytical_eigenvector(self, k, size):
        """
        Return the k-th analytical eigenvector of this eigenproblem.
        """
        self._raise_not_implemented_error('eigenvectors')

    def get_kth_analytical_eigenpair(self, k, size):
        omega = self.get_kth_analytical_eigenvalue(k, size)
        w = self.get_kth_analytical_eigenvector(k, size)
        return omega, w

    def get_analytical_eigenpairs(self, num, size):
        """
        Return the first `num` eigenvalues/eigenvectors for this eigenproblem.
        """
        omega = self.get_analytical_eigenvalues(num, size=size)
        w = self.get_analytical_eigenvectors(num, size)
        return omega, w

    def _set_plot_title(self, fig, title):
        ax = fig.gca()
        ax.set_title(title)

    def plot(self, w, fig, label, fmt=None):
        """
        Generic function to plot a given eigenvector `w` (which must have
        been computed in advance). This function can be overridden by
        subclasses if they require more advanced or specialised plotting
        functionality (e.g. if the system is not 1-dimensional).

        """
        ax = fig.gca()
        fmt = fmt or ''
        ax.plot(w, fmt, label=label)

    def plot_analytical_solutions(self, k_vals, N=None, figsize=None, filename=None):
        """
        *Arguments*

        k_vals:

            List of indices of the analytical solutions to be plotted.

        filename:

            If given, the plot will be saved to a file with the given name.
        """
        fig = plt.figure(figsize=figsize)
        for k in k_vals:
            omega, w = self.get_kth_analytical_eigenpair(k, N)
            label = 'k={} (EV: {:.3g})'.format(k, omega)
            self.plot(w, fig=fig, label=label)

        h, l = fig.gca().get_legend_handles_labels()
        fig.legend(h, l, 'lower center')
        title_str = 'Solutions to {}, N={}'.format(self, N)
        self._set_plot_title(fig, title_str)
        if filename != None:
            fig.savefig(filename)
        return fig

    def plot_computed_solutions(self, k_vals, solver=None, N=None, dtype=None,
                                num=None, plot_analytical_approximations=True,
                                tol_eigval=1e-8, figsize=None, filename=None):
        """
        *Arguments*

        k_vals: list of int

            List of indices of the analytical solutions to be plotted.

        solver:

            The eigensolver to use in order to compute the solutions.

        N:  int

            The size of the matrices defining the eigenvalue problem.

        dtype:  float | complex

            The dtype of the matrices defining the eigenvalue problem.

        num:  int

            The number of solutions to be computed. This argument may
            be of interest for the sparse solvers (which may return
            different results depending on the number of solutions
            requested.)

        plot_analytical_approximations:  bool

            If True (the default), the best approximation in terms of
            exact solutions is also plotted along with each computed
            solution.

        tol_eigval:  float

            Used to determine the best analytical approximation of the
            computed solution. See the docstring of the function
            `AbstractEigenproblem.best_analytical_approximation()` for
            details.

        filename:

            If given, the plot will be saved to a file with the given name.
        """
        fig = plt.figure(figsize=figsize)
        num = num or (max(k_vals) + 1)
        omegas, ws, rel_errors = self.solve(solver, N, dtype, num=num)

        for (idx, k) in enumerate(k_vals):
            # Plot computed solution
            omega = omegas[k]
            w = ws[k]
            rel_err = rel_errors[k]

            # Plot analytical approximation if requested
            if plot_analytical_approximations:
                w_approx, res_approx = \
                    self.best_analytical_approximation(omega, w, tol_eigval=tol_eigval)
                fmt = '--x{}'.format(color_cycle[idx % len(k_vals)])
                label = 'k={}, analyt., res.: {:.2g}'.format(k, res_approx)
                self.plot(w_approx, fmt=fmt, fig=fig, label=label)

            label = 'k={} (EV: {:.3g}, rel. err.: {:.3g})'.format(k, omega, rel_err)
            fmt = '-{}'.format(color_cycle[idx % len(k_vals)])
            self.plot(w, fmt=fmt, fig=fig, label=label)

        h, l = fig.gca().get_legend_handles_labels()
        fig.legend(h, l, 'lower center')
        dtype_str = {float: 'float', complex: 'complex'}[dtype]
        # XXX TODO: Make sure the title reflects the specific values
        #           of the solver used during the 'compute' method.
        title_str = 'Solutions to {}, solver={}, N={}, dtype={}'.format(
            self, solver, N, dtype_str)
        self._set_plot_title(fig, title_str)
        if filename != None:
            fig.savefig(filename)
        return fig

    def verify_eigenpair_numerically(self, a, v, tol=1e-8):
        """
        Check that the eigenpair `(a, eigenvec)` is a valid eigenpair
        of the given eigenproblem. Return `True` if

            |A*v - a*v| < tol

        and `False` otherwise.


        *Arguments*

        a :  float | complex

            The eigenvalue to be verified.

        v :  numpy.array

            The eigenvector to be verified.

        tol :  float

            The tolerance for comparison.
        """
        v = np.asarray(v)
        if not v.ndim == 1:
            raise ValueError("Expected 1D vector as second argument `v`. "
                             "Got: array of shape {}".format(v.shape))
        N = len(v)
        A, _ = self.instantiate(N, dtype=v.dtype)
        residue = np.linalg.norm(np.dot(A, v) - a*v)
        return residue < tol

    def verify_eigenpairs_numerically(self, eigenpairs, tol=1e-8):
        for omega, w in eigenpairs:
            if not self.verify_eigenpair_numerically(omega, w, tol=tol):
                return False
        return True

    def get_analytical_eigenspace_basis(self, eigval, size, tol_eigval=1e-8):
        """
        Return a list of all analytical eigenvectors whose eigenvalue is equal
        to `eigval` (within relative tolerance `tol_eigval`). Thus the returned
        vectors form a basis of the eigenspace associated to `eigval`.

        """
        # if not abs(eigval.imag) < 1e-12:
        #     raise ValueError("Eigenvalue must be real. Got: {}".format(eigval))
        indices = []
        k_left = None
        k_right = None
        for k in xrange(size):
            a = self.get_kth_analytical_eigenvalue(k, size)
            if np.allclose(a, eigval, atol=tol_eigval, rtol=tol_eigval):
                indices.append(k)
            # We use 'abs' in the following comparisons because the
            # eigenvalues may be complex/imaginary.
            if abs(a) < abs(eigval):
                k_left = k
            if abs(a) > abs(eigval):
                if k_right == None:
                    k_right = k
                if abs(a) > abs(eigval) * (1. + 2 * tol_eigval):
                    break
        eigenspace_basis = [self.get_kth_analytical_eigenvector(k, size) for k in indices]
        if eigenspace_basis == []:
            if k_left != None:
                eigval_left = self.get_kth_analytical_eigenvalue(k_left, size)
            else:
                eigval_left = None
            if k_right != None:
                eigval_right = self.get_kth_analytical_eigenvalue(k_right, size)
            else:
                eigval_right = None
            #print "eigval_right - a: {}".format(eigval_right - eigval)
            raise ValueError(
                "Not an eigenvalue of {}: {} (tolerance: {}). Closest "
                "surrounding eigenvalues: ({}, {})".format(
                    self, eigval, tol_eigval, eigval_left, eigval_right))
        return eigenspace_basis

    def best_analytical_approximation(self, a, v, tol_eigval=1e-8):
        """
        Compute a basis <e_i> of the eigenspace associated with `a` and find
        the vector w which minimise the residual:

            res = |v - w|

        where `w` is constrained to be a linear linear combination of
        the basis vectors `e_i`:

            w = \sum_{i} b_i * e_i  (b_i \in R)

        The return value is the pair `(w, res)` consisting of the best
        approximation and the residual.


        *Arguments*

        a:  float

            The eigenvalue to be verified.

        v:  numpy.array

            The eigenvector to be verified.

        tol_eigval:  float

            Used to determine the eigenspace basis associated with the
            given eigenvalue `a`. All those analytical eigenvectors
            are considered for the basis whose associated (exact)
            eigenvalue lies within `tol_eigval` of `a`.

        tol_residual:  float

            The tolerance for the residual. The given eigenpair `(a,
            v)` is accepted as verified iff the minimum of `v` minus
            any linear combination of eigenspace basis vectors is less
            than `tol_residual`.
        """
        v = np.asarray(v)
        if (v.ndim != 1):
            raise TypeError("Expected 1D array for second argument `v`. "
                            "Got: '{}'".format(v))
        size = len(v)
        eigenspace_basis = self.get_analytical_eigenspace_basis(a, size, tol_eigval=tol_eigval)
        w, _, res = best_linear_combination(v, eigenspace_basis)
        return w, res

    def verify_eigenpair_analytically(self, a, v, tol_residual=1e-8, tol_eigval=1e-8):
        """
        Check whether the vector `v` lies in the eigenspace associated
        with the eigenvalue `a`.

        See `best_analytical_approximation` for details about the
        tolerance arguments.

        """
        _, res = self.best_analytical_approximation(a, v, tol_eigval=tol_eigval)
        return res < tol_residual

    def verify_eigenpairs_analytically(self, eigenpairs, tol_residual=1e-8, tol_eigval=1e-8):
        for omega, w in eigenpairs:
            if not self.verify_eigenpair_analytically(
                omega, w, tol_residual=tol_residual, tol_eigval=tol_eigval):
                return False
        # XXX TODO: Also verify that we indeed computed the N smallest
        #           solutions, not just any analytical solutions!
        return True


class DiagonalEigenproblem(AbstractEigenproblem):
    """
    Eigenproblem of the form

       D*v = omega*v

    where D is a diagonal matrix of the form

       D = diag([1, 2, 3, ..., N])

    """
    def instantiate(self, N, dtype):
        """
        Return a pair (D, None), where D is a diagonal matrix of the
        form D=diag([1, 2, 3, ..., N]).

        """
        # XXX TODO: Actually, we shouldn't store these values in 'self'
        #           because we can always re-instantiate the problem,
        #           right?
        self.N = N
        self.dtype = dtype
        self.diagvals = np.arange(1, N+1, dtype=dtype)
        return np.diag(self.diagvals), None

    def is_hermitian(self):
        return True

    def get_kth_analytical_eigenvalue(self, k, size=None):
        return (k + 1)

    def get_kth_analytical_eigenvector(self, k, size):
        w = np.zeros(size)
        w[k] = 1.0
        return w


class RingGraphLaplaceEigenproblem(AbstractEigenproblem):
    """
    Eigenproblem of the form

       A*v = omega*v

    where A is a Laplacian matrix. of the form

       [ 2, -1,  0,  0, ...]
       [-1,  2, -1,  0, ...]
       [ 0, -1,  2, -1, ...]
       ...
       [..., 0, -1,   2, -1]
       [..., 0,  0,  -1,  2]

    """
    def instantiate(self, N, dtype):
        """
        Return a pair (A, None), where A is a Laplacian matrix
        of the form

           [ 2, -1,  0,  0, ...]
           [-1,  2, -1,  0, ...]
           [ 0, -1,  2, -1, ...]
           ...
           [..., 0, -1,   2, -1]
           [..., 0,  0,  -1,  2]

        """
        # XXX TODO: Actually, we shouldn't store these values in 'self'
        #           because we can always re-instantiate the problem,
        #           right?
        self.N = N
        self.dtype = dtype
        A = np.zeros((N, N), dtype=dtype)
        A += np.diag(2 * np.ones(N))
        A -= np.diag(1 * np.ones(N-1), k=1)
        A -= np.diag(1 * np.ones(N-1), k=-1)
        A[0, N-1] = -1
        A[N-1, 0] = -1
        return A, None

    def is_hermitian(self):
        return True

    def get_kth_analytical_eigenvalue(self, k, size=None):
        i = (k + 1) // 2
        return 2 - 2 * np.cos(2 * pi * i / size)

    def get_kth_analytical_eigenvector(self, k, size):
        xs = np.arange(size)
        i = (k + 1) // 2
        if iseven(k):
            res = np.cos(2 * pi * i * xs / size)
        else:
            res = np.sin(2 * pi * i * xs / size)
        return res / np.linalg.norm(res)


def m0_cross(m0, v):
    """
    Compute and return the site-wise cross product
    of the two vector fields `m0` and `v`.
    """
    return np.cross(m0.reshape(3, -1), v.reshape(3, -1), axis=0)


class Nanostrip1dEigenproblemFinmag(AbstractEigenproblem):
    """
    Eigenproblem of the form

       A*v = omega*v

    where A is a matrix that represents the right-hand side of the
    action of the linearised LLG equation (without damping):

       dv/dt = -gamma * m_0 x H_exchange(v)

    """
    def __init__(self, A_ex, Ms, xmin, xmax, unit_length=1e-9, regular_mesh=True):
        """
        *Arguments*

        A_ex:  float

            Exchange coupling constant (in J/m).

        Ms:  float

            Saturation magnetisation (in A/m).

        xmin, xmax:  float

            The bounds of the interval on which the 1D system is
            defined.

        regular_mesh:  bool

            If True (the default), the mesh nodes will be equally
            spaced in the interval [xmin, xmax]. Otherwise they will
            be randomly chosen.

        """
        self.A_ex = A_ex
        self.Ms = Ms
        self.xmin = xmin
        self.xmax = xmax
        self.unit_length = unit_length
        self.regular_mesh = regular_mesh
        self.L = (self.xmax - self.xmin) * self.unit_length

    def compute_exchange_field(self, v):
        """
        Compute the exchange field for the given input vector `v`
        (which should be a `numpy.array`).

        """
        self.exch.m.vector()[:] = v
        return self.exch.compute_field()

    def compute_action_rhs_linearised_LLG(self, m0, v):
        """
        This function computes the action of the right hand side of
        the linearised LLG on the vector `v` and returns the result.
        Explicitly, it computes

           -gamma * (m0 x H_ex(v))

        where `m0` is the equilibrium configuration around which the
        LLG was linearised.

        """
        H = self.compute_exchange_field(v).reshape(3, -1)
        return -gamma * m0_cross(m0, H)

    def compute_action_rhs_linearised_LLG_2K(self, v):
        """
        Compute the action of the right hand side of the linearised
        LLg equation on the vector `v`, which should have a shape
        compatible with (2, N). This function assumes that the
        equilibrium configuration around which the LLG equation was
        linearised is `m0 = (0, 0, 1)^T`.

        """
        K = self.K
        m0 = np.array([0, 0, 1])
        v_3K = np.zeros(3*K)
        v_3K[:2*K] = v
        #v_3K.shape = (3, -1)
        res_3K = self.compute_action_rhs_linearised_LLG(m0, v_3K).ravel()
        res = res_3K[:2*K]
        return res

    def instantiate(self, N, dtype, regular_mesh=None):
        """
        Return a pair (A, None), where A is a matrix representing the
        action on a vector v given by the right-hand side of the
        linearised LLG equation (without damping):

            A*v = dv/dt = -gamma * m_0 x H_exchange(v)

        """
        if not iseven(N):
            raise ValueError("N must be even. Got: {}".format(N))
        # XXX TODO: Actually, we shouldn't store these values in 'self'
        #           because we can always re-instantiate the problem,
        #           right?
        self.N = N
        self.dtype = dtype
        if regular_mesh == None:
            regular_mesh = self.regular_mesh
        self.K = N // 2

        if regular_mesh:
            mesh = df.IntervalMesh(self.K-1, self.xmin, self.xmax)
        else:
            mesh = irregular_interval_mesh(self.xmin, self.xmax, self.K)
            #raise NotImplementedError()

        V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
        v = df.Function(V)
        self.exch = Exchange(A=self.A_ex)
        self.exch.setup(V, v, Ms=self.Ms, unit_length=self.unit_length)

        C_2Kx2K = LinearOperator(shape=(N, N), matvec=self.compute_action_rhs_linearised_LLG_2K, dtype=dtype)
        C_2Kx2K_dense = as_dense_array(C_2Kx2K)
        return C_2Kx2K_dense, None

    def get_kth_analytical_eigenvalue(self, k, size=None):
        i = k // 2
        return 1j * (-1)**k * (2 * self.A_ex * gamma) / (mu0 * self.Ms) * (i * pi / self.L)**2

    def get_kth_analytical_eigenvector(self, k, size):
        assert(iseven(size))
        i = k // 2
        xs = np.linspace(self.xmin * self.unit_length, self.xmax * self.unit_length, size // 2)
        v1 = np.cos(i*pi/self.L * xs)
        v2 = np.cos(i*pi/self.L * xs) * 1j * (-1)**i
        return np.concatenate([v1, v2])

    def _set_plot_title(self, fig, title):
        fig.suptitle(title, fontsize=16, verticalalignment='bottom')
        #fig.subplots_adjust(top=0.55)
        #fig.tight_layout()

    def plot(self, w, fig, label, fmt=None):
        """
        This function plots the real and imaginary parts of the solutions separately,
        and in addition splits each soluton eigenvector into two halves (which represent
        the x- and y-component of the magnetisation, respectively).

        """
        if fig.axes == []:
            fig.add_subplot(2, 2, 1)
            fig.add_subplot(2, 2, 2)
            fig.add_subplot(2, 2, 3)
            fig.add_subplot(2, 2, 4)
        assert(len(fig.axes) == 4)
        ax1, ax2, ax3, ax4 = fig.axes

        N = len(w)
        K = N // 2
        assert(N == 2*K)

        # Scale the vector so that its maximum entry is 1.0
        w_max = abs(w).max()
        if w_max != 0.0:
            w = w / w_max

        fmt = fmt or ''
        ax1.plot(w[:K].real, fmt, label=label)
        ax3.plot(w[:K].imag, fmt, label=label)
        ax2.plot(w[K:].real, fmt, label=label)
        ax4.plot(w[K:].imag, fmt, label=label)
        ax1.set_ylim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax3.set_ylim(-1.1, 1.1)
        ax4.set_ylim(-1.1, 1.1)


# NOTE: Don't change this list without adapting the "known_failures"
#       in "eigensolvers_test.py" accordingly!
available_eigenproblems = [DiagonalEigenproblem(),
                           RingGraphLaplaceEigenproblem(),
                           Nanostrip1dEigenproblemFinmag(13e-12, 8e5, 0, 100),
                           ]
