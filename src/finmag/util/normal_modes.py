from __future__ import division
import dolfin as df
import numpy as np
import logging
import math
import scipy.sparse.linalg
from time import time
from finmag.util.consts import gamma

logger = logging.getLogger('finmag')


# Matrix-vector or Matrix-matrix product
def _mult_one(a, b):
    # a and b are ?x?xn arrays where ? = 1..3
    assert len(a.shape) == 3
    assert len(b.shape) == 3
    assert a.shape[2] == b.shape[2]
    assert a.shape[1] == b.shape[0]
    assert a.shape[0] <= 3 and a.shape[1] <= 3
    assert b.shape[0] <= 3 and b.shape[1] <= 3

    # One of the arrays might be complex, so we determine the type
    # of the resulting array by adding two elements of the argument arrays
    res = np.zeros((a.shape[0], b.shape[1], a.shape[2]), dtype=type(a[0, 0, 0] + b[0, 0, 0]))
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            for k in xrange(a.shape[1]):
                res[i, j, :] += a[i, k, :] * b[k, j, :]

    return res


# Returns the componentwise matrix product of the supplied matrix fields
def mf_mult(*args):
    if len(args) < 2:
        raise Exception("mult requires at least 2 arguments")

    res = args[0]
    for i in xrange(1, len(args)):
        res = _mult_one(res, args[i])

    return res


# Transposes the mxk matrix to a kxm matrix
def mf_transpose(a):
    return np.transpose(a, [1, 0, 2])


# Computes the componentwise cross product of a vector field a
# and a vector or vector field b
def mf_cross(a, b):
    assert a.shape == (3, 1, a.shape[2])

    res = np.empty(a.shape, dtype=a.dtype)
    res[0] = a[1] * b[2] - a[2] * b[1]
    res[1] = a[2] * b[0] - a[0] * b[2]
    res[2] = a[0] * b[1] - a[1] * b[0]
    return res


# Normalises the 3d vector m
def mf_normalise(m):
    assert m.shape == (3, 1, m.shape[2])
    return m / np.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] *m[2])


# Set up the basis for the tangential space and the corresponding projection operator
def compute_tangential_space_basis(m0):
    assert m0.ndim == 3
    n = m0.shape[2]
    assert m0.shape == (3, 1, n)

    # Set up a field of vectors m_perp that are perpendicular to m0
    # Start with e_z and compute e_z x m
    m_perp = mf_cross(m0, [0., 0., -1.])
    # In case m || e_z, add a tiny component in e_y
    m_perp[1] += 1e-100
    # Normalise and compute the cross product with m0 again
    m_perp = mf_cross(mf_normalise(m_perp), m0)
    m_perp = mf_normalise(m_perp)

    # The basis in the 3d space is ((m_perp x m0) x m0, m_perp x m0, m0)
    R = np.zeros((3, 3, n))
    R[:, 2, :] = m0[:, 0, :]
    R[:, 1, :] = m_perp[:, 0, :]
    R[:, 0, :] = mf_cross(m_perp, m0)[:, 0, :]

    # Matrix for the injection from 2n to 3n (3x2)
    S = np.zeros((3, 2, n))
    S[0, 0, :] = 1.
    S[1, 1, :] = 1.
    # Matrix for the projection from 3n to 2n is transpose(S)

    # Matrix for the cross product m0 x in the 2n space
    Mcross = np.zeros((2, 2, n))
    Mcross[0, 1, :] = -1
    Mcross[1, 0, :] = 1

    # The relationship between the 3d tangential vector v
    # and the 2d vector w is
    # v = (R S) w
    # w = (R S)^t v
    Q = mf_mult(R, S)
    return Q, R, S, Mcross


def differentiate_fd4(f, x, dx):
    """
    Compute and return a fourth-order approximation to the directional
    derivative of `f` at the point `x` in the direction of `dx`.
    """
    x_sq = np.dot(x, x)
    dx_sq = np.dot(dx, dx)
    h = 0.001 * np.sqrt(x_sq + dx_sq) / np.sqrt(dx_sq + 1e-50)
    # weights: 1. / 12., -2. / 3., 2. / 3., -1. / 12.
    # coefficients: -2., -1., 1., 2.
    res = (1./12./h)*f(x - 2*h*dx)
    res += (-2./3./h)*f(x - h*dx)
    res += (2./3./h)*f(x + h*dx)
    res += (-1./12./h)*f(x + 2*h*dx)
    return res


def compute_eigenproblem_matrix(sim, frequency_unit=1e9, filename=None):
    """
    Compute and return the square matrix `D` defining the eigenproblem which
    has the normal mode frequencies and oscillation patterns as its solution.

    Note that `sim` needs to be in a relaxed state, otherwise the results will
    be wrong.
    
    """
    ## Create the helper simulation which we use to compute
    ## the effective field for various values of m.
    #Ms = sim.Ms
    #A = sim.get_interaction('Exchange').A
    #unit_length = sim.unit_length
    #try:
    #    sim.get_interaction('Demag')
    #    demag_solver = 'FK'
    #except ValueError:
    #    demag_solver = None
    #sim_aux = sim_with(sim.mesh, Ms=Ms, m_init=[1, 0, 0], A=A, unit_length=unit_length, demag_solver=demag_solver)
    # In order to compute the derivative of the effective field, the magnetisation needs to be set
    # to many different values. Thus we store a backup so that we can restore it later.
    m_orig = sim.m

    def effective_field_for_m(m):
        if np.iscomplexobj(m):
            raise NotImplementedError("XXX TODO: Implement the version for complex arrays!")
        sim.set_m(m)
        return sim.effective_field()

    n = sim.mesh.num_vertices()
    N = 3 * n  # number of degrees of freedom

    m0_array = sim.m.copy()
    m0_flat = m0_array.reshape(3, n)  # 'flat' is a slightly misleading terminology, but it's used in Simlib so we keep it here
    m0_column_vector = m0_array.reshape(3, 1, n)
    H0_array = effective_field_for_m(m0_array)
    H0_flat = H0_array.reshape(3, n)
    h0 = H0_flat[0]*m0_flat[0] + H0_flat[1]*m0_flat[1] + H0_flat[2]*m0_flat[2]

    logger.debug("Computing basis of the tangent space and transition matrices.")
    Q, R, S, Mcross = compute_tangential_space_basis(m0_column_vector)
    Qt = mf_transpose(Q).copy()

    # Returns the product of the linearised llg times vector
    def linearised_llg_times_vector(v):
        assert v.shape == (3, 1, n)
        # The linearised equation is
        # dv/dt = - gamma m0 x (H' v - h_0 v)
        v_array = v.view()
        v_array.shape = (-1,)
        # Compute H'v
        res = differentiate_fd4(effective_field_for_m, m0_array, v_array)
        res.shape = (3, -1)
        # Subtract h0 v
        res[0] -= h0 * v[0,0]
        res[1] -= h0 * v[1,0]
        res[2] -= h0 * v[2,0]
        # Multiply by -gamma m0x
        res *= gamma
        res.shape = (3, 1, -1)
        # Put res on the left in case v is complex
        res = mf_cross(res, m0_column_vector)
        return res

    #The linearised equation in the tangential basis
    def linearised_llg_times_tangential_vector(w):
        w = w.view()
        w.shape = (2, 1, n)
        # Go to the 3d space
        v = mf_mult(Q, w)
        # Compute the linearised llg
        L = linearised_llg_times_vector(v)
        # Go back to 2d space
        res = np.empty(w.shape, dtype=complex)
        res[:] = mf_mult(Qt, L)
        # Multiply by -i/(2*pi*U) so that we get frequencies as the real part of eigenvalues
        res *= -1j / (2 * math.pi * frequency_unit)
        res.shape = (-1,)
        return res

    df.tic()
    logger.info("Assembling eigenproblem matrix.")
    D = np.zeros((2*n, 2*n), dtype=complex)
    for i, w in enumerate(np.eye(2*n)):
        if i % 50 == 0:
            logger.debug("Processing row {}/{}  (time taken so far: {:.2f} seconds)".format(i, 2*n, df.toc()))
        D[:,i] = linearised_llg_times_tangential_vector(w)

    logger.debug("Eigenproblem matrix D occupies {:.2f} MB of memory.".format(D.nbytes / 1024.**2))

    if filename != None:
        logger.info("Saving eigenproblem matrix to file '{}'".format(filename))
        np.save(filename, D)

    # Restore the original magnetisation.
    # XXX TODO: Is this method safe, or does it leave any trace of the temporary changes we did above?
    sim.set_m(m_orig)

    return D



# We use the following class (which behaves like a function due to its
# __call__ method) instead of a simple lambda expression because it is
# pickleable, which is needed if we want to cache computation results.
#
# XXX TODO: lambda expresions can be pickled with the 'dill' module,
# so we should probably get rid of this.
class M_times_w(object):
    def __init__(self, Mcross, n, alpha=0.):
        self.Mcross = Mcross
        self.n = n
        self.alpha = alpha

    def __call__(self, w):
        w = w.view()
        w.shape = (2, 1, self.n)
        res = -1j * mf_mult(self.Mcross, w)
        if self.alpha != 0.:
            res += -1j * self.alpha * w
        res.shape = (-1,)
        return res


class NotImplementedOp(object):
    def __call__(self, w):
        raise NotImplementedError("rmatvec is not implemented")


def compute_generalised_eigenproblem_matrices(sim, alpha=0.0, frequency_unit=1e9, filename_mat_A=None, filename_mat_M=None):
    """
    XXX TODO: write me

    """
    m_orig = sim.m

    def effective_field_for_m(m):
        if np.iscomplexobj(m):
            raise NotImplementedError("XXX TODO: Implement the version for complex arrays!")
        sim.set_m(m)
        return sim.effective_field()

    n = sim.mesh.num_vertices()
    N = 3 * n  # number of degrees of freedom

    m0_array = sim.m.copy()
    m0_flat = m0_array.reshape(3, n)  # 'flat' is a slightly misleading terminology, but it's used in Simlib so we keep it here
    m0_column_vector = m0_array.reshape(3, 1, n)
    H0_array = effective_field_for_m(m0_array)
    H0_flat = H0_array.reshape(3, n)
    h0 = H0_flat[0]*m0_flat[0] + H0_flat[1]*m0_flat[1] + H0_flat[2]*m0_flat[2]

    logger.debug("Computing basis of the tangent space and transition matrices.")
    Q, R, S, Mcross = compute_tangential_space_basis(m0_column_vector)
    Qt = mf_transpose(Q).copy()

    def A_times_vector(v):
        # A = H' v - h_0 v
        assert v.shape == (3, 1, n)
        v_array = v.view()
        v_array.shape = (-1,)
        # Compute H'v
        res = differentiate_fd4(effective_field_for_m, m0_array, v_array)
        res.shape = (3, n)
        # Subtract h0 v
        res[0] -= h0 * v[0, 0]
        res[1] -= h0 * v[1, 0]
        res[2] -= h0 * v[2, 0]
        res.shape = (3, 1, n)
        return res

    df.tic()
    logger.info("Assembling eigenproblem matrix.")
    A = np.zeros((2*n, 2*n), dtype=complex)
    # Compute A
    w = np.zeros(2*n)
    for i in xrange(2*n):
        if i % 50 == 0:
            logger.debug("Processing row {}/{}  (time taken so far: {:.2f} seconds)".format(i, 2*n, df.toc()))

        # Ensure that w is the i-th standard basis vector
        w.shape = (2*n,)
        w[i-1] = 0.0  # this will do no harm if i==0
        w[i] = 1.0

        w.shape = (2, 1, n)
        Av = A_times_vector(mf_mult(Q, w))
        A[:, i] = mf_mult(Qt, Av).reshape(-1)
        # Multiply by (-gamma)/(2 pi U)
        A[:, i] *= -gamma / (2 * math.pi * frequency_unit)

    logger.debug("Eigenproblem matrix A occupies {:.2f} MB of memory.".format(A.nbytes / 1024.**2))

    # # Compute B, which is -i Mcross 2 pi U / gamma
    # B = np.zeros((2, n, 2, n), dtype=complex)
    # for i in xrange(n):
    #     B[:, i, :, i] = Mcross[:, :, i]
    #     B[:, i, :, i] *= -1j
    # B.shape = (2*n, 2*n)

    M = scipy.sparse.linalg.LinearOperator((2 * n, 2 * n), M_times_w(Mcross, n, alpha), NotImplementedOp(), NotImplementedOp(), dtype=complex)

    if filename_mat_A != None:
        logger.info("Saving generalised eigenproblem matrix 'A' to file '{}'".format(filename_mat_A))
        np.save(filename_mat_A, A)

    if filename_mat_M != None:
        logger.info("Saving generalised eigenproblem matrix 'M' to file '{}'".format(filename_mat_M))
        np.save(filename_mat_M, M)

    # Restore the original magnetisation.
    # XXX TODO: Is this method safe, or does it leave any trace of the temporary changes we did above?
    sim.set_m(m_orig)

    return A, M


def compute_normal_modes(D, n_values=10, sigma=0., tol=1e-8, which='LM'):
    logger.debug("Solving eigenproblem. This may take a while...".format(df.toc()))
    df.tic()
    omega, w = scipy.sparse.linalg.eigs(D, n_values, which=which, sigma=0., tol=tol, return_eigenvectors=True)
    logger.debug("Computing the eigenvalues and eigenvectors took {:.2f} seconds".format(df.toc()))

    return omega, w


def compute_normal_modes_generalised(A, M, n_values=10, tol=1e-8):
    logger.debug("Solving eigenproblem. This may take a while...".format(df.toc()))
    df.tic()
    # Have to swap M and A since the M matrix has to be positive definite for eigsh!
    omega, w = scipy.sparse.linalg.eigsh(M, n_values, A, which='LM', tol=tol, return_eigenvectors=True)
    logger.debug("Computing the eigenvalues and eigenvectors took {:.2f} seconds".format(df.toc()))

    # We need to return 1/omega because we swapped M and A above and thus computed the inverse eigenvalues.
    return 1/omega, w


def export_normal_mode_animation(sim, freq, w, filename, num_cycles=1, num_snapshots_per_cycle=20, scaling=0.2):
    """
    Save a number of vtk files of different snapshots of a given normal mode.
    These can be imported and animated in Paraview.

    *Arguments*

    comp :  NormalModesComputation

    freq :  float

        The frequency of the normal mode.

    w :  numpy.array

        The eigenvector representing the normal mode (as returned by `compute_eigenv`
        or `compute_eigenv_generalised`).

    filename :  string

        The filename of the exported animation files. Each individual frame will
        have the same basename but will be given a suffix indicating the frame
        number, too.

    num_cycles :  int

        The number of cycles to be animated.

    num_snapshots_per_cycle :  int

        The number of snapshot per cycle to be exported. Thus the total number of
        exported frames is num_cycles * num_snapshots_per_cycle.

    scaling :  double

        Determines the size of the oscillation amplitudes in the animation (relative
        to the lengths of the magnetic moments).

    """
    if not freq.imag == 0:
        if abs(freq.imag) < 1e-8:
            freq = freq.real
        else:
            raise ValueError("Frequency must be (at least approximately) real. Got: {}".format(freq))
    #basename = os.path.basename(re.sub('\.vtk$', '', filename))
    #dirname = os.path.dirname(filename)
    #if not os.path.exists(dirname):
    #    print "Creating directory '{}' as it doesn't exist.".format(dirname)
    #    os.makedirs(dirname)
    #mesh = comp.mesh
    #mesh_shape = mesh.mesh_size
    m0_array = sim.m.copy()  # we assume that sim is relaxed!!
    Q, R, S, Mcross = compute_tangential_space_basis(m0_array.reshape(3, 1, -1))
    Qt = mf_transpose(Q).copy()

    n = sim.mesh.num_vertices()
    V = df.VectorFunctionSpace(sim.mesh, 'CG', 1, dim=3)
    func = df.Function(V)
    func.rename('m', 'magnetisation')
    w_3d = mf_mult(Q, w.reshape((2, 1, n)))
    w_flat = w_3d.reshape(-1)
    phi = np.angle(w_flat)  # relative phases of the oscillations
    a = np.absolute(w_flat)
    a = a / a.max()  # normalised amplitudes of the oscillations

    t_end = num_cycles * 2 * math.pi / freq
    timesteps = np.linspace(0, t_end, num_cycles * num_snapshots_per_cycle, endpoint=False)
    m_osc = np.zeros(3*n)
    t0 = time()
    f = df.File(filename, 'compressed')
    for (i, t) in enumerate(timesteps):
        print "Saving animation snapshot for timestep {} ({}/{})".format(t, i, num_cycles * num_snapshots_per_cycle)
        m_osc = (m0_array + scaling * a * np.cos(t * freq + phi)).reshape(-1)
        #save_vector_field(m_osc, os.path.join(dirname, basename + '_{:04d}.vtk'.format(i)))
        func.vector().set_local(m_osc)
        f << func
    t1 = time()
    print("Saving the data to file '{}' took {} seconds".format(filename, t1 - t0))
