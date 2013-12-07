from __future__ import division
import dolfin as df
import numpy as np
import logging
import os
import scipy.sparse.linalg
from time import time
from finmag.util.consts import gamma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from itertools import izip
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import pi
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

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
    m0_3xn = m0_array.reshape(3, n)  # this corresponds to the vector 'm0_flat' in Simlib
    m0_column_vector = m0_array.reshape(3, 1, n)
    H0_array = effective_field_for_m(m0_array)
    H0_3xn = H0_array.reshape(3, n)
    h0 = H0_3xn[0]*m0_3xn[0] + H0_3xn[1]*m0_3xn[1] + H0_3xn[2]*m0_3xn[2]

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
        res *= -1j / (2 * pi * frequency_unit)
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


def is_hermitian(A, atol=1e-8, rtol=1e-12):
    """
    Returns True if the matrix `A` is Hermitian (up to the given
    tolerance) and False otherwise.

    The arguments `atol` and `rtol` have the same meaning as in
    `numpy.allclose`.

    """
    if isinstance(A, np.ndarray):
        # Note: just using an absolute tolerance and checking for
        # the maximum difference is about twice as efficient, so
        # maybe we should avoid the relative tolerance in the future.
        return np.allclose(A, np.conj(A.T), atol=atol, rtol=rtol)
    elif isinstance(A, scipy.sparse.linalg.LinearOperator):
        raise NotImplementedError
    else:
        raise NotImplementedError


def check_is_hermitian(A, matrix_name, atol=1e-8, rtol=1e-12):
    """
    Check if `A` is hermitian and print a warning if this is not the case.

    The argument `matrix_name` is only used for printing the warning.

    """
    if not is_hermitian(A):
        mat_diff = np.absolute(A - np.conj(A.T))
        logger.critical("Matrix {} is not Hermitian. Maximum difference "
                        "between A and conj(A^tr): {}, median difference: {}, "
                        "mean difference: {} (maximum entry of A: {}, "
                        "median entry: {}, mean entry: {})".format(
                matrix_name, mat_diff.max(), np.median(mat_diff), np.mean(mat_diff),
                np.max(np.absolute(A)), np.median(np.absolute(A)), np.mean(np.absolute(A))))


def compute_generalised_eigenproblem_matrices(sim, alpha=0.0, frequency_unit=1e9, filename_mat_A=None, filename_mat_M=None, check_hermitian=False):
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
    m0_3xn = m0_array.reshape(3, n)  # this corresponds to the vector 'm0_flat' in Simlib
    m0_column_vector = m0_array.reshape(3, 1, n)
    H0_array = effective_field_for_m(m0_array)
    H0_3xn = H0_array.reshape(3, n)
    h0 = H0_3xn[0]*m0_3xn[0] + H0_3xn[1]*m0_3xn[1] + H0_3xn[2]*m0_3xn[2]

    logger.debug("Computing basis of the tangent space and transition matrices.")
    Q, R, S, Mcross = compute_tangential_space_basis(m0_column_vector)
    Qt = mf_transpose(Q).copy()

    logger.debug("Q.shape: {} ({} MB)".format(Q.shape, Q.nbytes / 1024.**2))

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
        A[:, i] *= -gamma / (2 * pi * frequency_unit)

    logger.debug("Eigenproblem matrix A occupies {:.2f} MB of memory.".format(A.nbytes / 1024.**2))

    # # Compute B, which is -i Mcross 2 pi U / gamma
    # B = np.zeros((2, n, 2, n), dtype=complex)
    # for i in xrange(n):
    #     B[:, i, :, i] = Mcross[:, :, i]
    #     B[:, i, :, i] *= -1j
    # B.shape = (2*n, 2*n)

    M = scipy.sparse.linalg.LinearOperator((2 * n, 2 * n), M_times_w(Mcross, n, alpha), NotImplementedOp(), NotImplementedOp(), dtype=complex)

    if check_hermitian:
        # Sanity check: A and M should be Hermitian matrices
        check_is_hermitian(A, "A")
        #check_is_hermitian(M, "M")

    if filename_mat_A != None:
        dirname_mat_A = os.path.dirname(os.path.abspath(filename_mat_A))
        if not os.path.exists(dirname_mat_A):
            logger.debug("Creating directory '{}' as it does not exist.".format(dirname_mat_A))
            os.makedirs(dirname_mat_A)
        logger.info("Saving generalised eigenproblem matrix 'A' to file '{}'".format(filename_mat_A))
        np.save(filename_mat_A, A)

    if filename_mat_M != None:
        dirname_mat_M = os.path.dirname(os.path.abspath(filename_mat_M))
        if not os.path.exists(dirname_mat_M):
            logger.debug("Creating directory '{}' as it does not exist.".format(dirname_mat_M))
            os.makedirs(dirname_mat_M)
        logger.info("Saving generalised eigenproblem matrix 'M' to file '{}'".format(filename_mat_M))
        np.save(filename_mat_M, M)

    # Restore the original magnetisation.
    # XXX TODO: Is this method safe, or does it leave any trace of the temporary changes we did above?
    sim.set_m(m_orig)

    return A, M, Q, Qt


def compute_normal_modes(D, n_values=10, sigma=0., tol=1e-8, which='LM'):
    logger.debug("Solving eigenproblem. This may take a while...".format(df.toc()))
    df.tic()
    omega, w = scipy.sparse.linalg.eigs(D, n_values, which=which, sigma=0., tol=tol, return_eigenvectors=True)
    logger.debug("Computing the eigenvalues and eigenvectors took {:.2f} seconds".format(df.toc()))

    return omega, w


def compute_normal_modes_generalised(A, M, n_values=10, tol=1e-8, discard_negative_frequencies=False, sigma=None, which='LM',
                                     v0=None, ncv=None, maxiter=None, Minv=None, OPinv=None, mode='normal'):
    logger.debug("Solving eigenproblem. This may take a while...".format(df.toc()))
    df.tic()

    if discard_negative_frequencies:
        n_values *= 2

    # XXX TODO: The following call seems to increase memory consumption quite a bit. Why?!?
    #
    # We have to swap M and A when passing them to eigsh since the M matrix has to be positive definite for eigsh!
    omega_inv, w = scipy.sparse.linalg.eigsh(M, k=n_values, M=A, which=which, tol=tol, return_eigenvectors=True, sigma=sigma,
                                             v0=v0, ncv=ncv, maxiter=maxiter, Minv=Minv, OPinv=OPinv, mode=mode)
    logger.debug("Computing the eigenvalues and eigenvectors took {:.2f} seconds".format(df.toc()))

    # The true eigenfrequencies are given by 1/omega_inv because we swapped M and A above and thus computed the inverse eigenvalues.
    omega = 1. / omega_inv

    # Sanity check: the eigenfrequencies should occur in +/- pairs.
    TOL = 1e-3
    positive_freqs = filter(lambda x: x >0, omega)
    negative_freqs = filter(lambda x: x <0, omega)
    freq_pairs = izip(positive_freqs, negative_freqs)
    if (n_values % 2 == 0 and len(positive_freqs) != len(negative_freqs)) or \
            (n_values % 2 == 0 and len(positive_freqs) - len(negative_freqs) not in [0, 1]) or \
            any([abs(x + y) > TOL for (x, y) in freq_pairs]):
        logger.warning("The eigenfrequencies should occur in +/- pairs, but this "
                       "does not seem to be the case (with TOL={})! Please "
                       "double-check that the results make sense!".format(TOL))

    # Find the indices that sort the frequency by absolute value,
    # with the positive frequencies occurring before the negative ones (where.
    sorted_indices = sorted(np.arange(len(omega)),
                            key=lambda i: (np.round(abs(omega[i]), decimals=4), -np.sign(omega[i]), abs(omega[i])))

    if discard_negative_frequencies:
        # Discard indices corresponding to negative frequencies
        sorted_indices = filter(lambda i: omega[i] >= 0.0, sorted_indices)

    omega = omega[sorted_indices]
    w = w[:, sorted_indices]  # XXX TODO: can we somehow avoid copying the columns to save memory?!?

    return omega, w


def export_normal_mode_animation(sim, freq, w, filename, num_cycles=1, num_snapshots_per_cycle=20, scaling=0.2, dm_only=False):
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

    scaling : float

        If `dm_only` is False, this determines the maximum size of the
        oscillation (relative to the magnetisation vector) in the
        visualisation. If `dm_only` is True, this has no effect.

    dm_only :  bool (optional)

        If False (the default), plots `m0 + scaling*dm(t)`, where m0 is the
        average magnetisation and dm(t) the (spatially varying)
        oscillation corresponding to the frequency of the normal mode.
        If True, only `dm(t)` is plotted.

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

    t_end = num_cycles * 2 * pi / freq
    timesteps = np.linspace(0, t_end, num_cycles * num_snapshots_per_cycle, endpoint=False)
    m_osc = np.zeros(3*n)
    t0 = time()
    f = df.File(filename, 'compressed')
    for (i, t) in enumerate(timesteps):
        logger.debug("Saving animation snapshot for timestep {} ({}/{})".format(t, i, num_cycles * num_snapshots_per_cycle))
        if dm_only is False:
            m_osc = (m0_array + scaling * a * np.cos(t * freq + phi)).reshape(-1)
        else:
            m_osc = (scaling * a * np.cos(t * freq + phi)).reshape(-1)
        #save_vector_field(m_osc, os.path.join(dirname, basename + '_{:04d}.vtk'.format(i)))
        func.vector().set_local(m_osc)
        f << func
    t1 = time()
    logger.debug("Saving the data to file '{}' took {} seconds".format(filename, t1 - t0))


def plot_spatially_resolved_normal_mode(sim, w, slice_z='z_max', components='xyz',
                                        figure_title=None, yshift_title=0.0,
                                        plot_powers=True, plot_phases=True, num_phase_colorbar_ticks=5,
                                        cmap_powers=plt.cm.jet, cmap_phases=plt.cm.hsv, vmin_powers=None,
                                        show_axis_labels=True, show_axis_frames=True, show_colorbars=True, figsize=None,
                                        outfilename=None, dpi=None):
    """
    Plot the normal mode profile across a slice of the sample.

    Remark: Due to a bug in matplotlib (see [1]), when saving the
    `matplotlib.Figure` object returned from this function the title
    and left annotations will likely be cut off. Therefore it is
    recommended to save the plot by specifying the argument
    `outfilename`.

    [1] http://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box



    *Arguments*

    sim:

        The simulation object for which the eigenmode was computed.

    w:

        The eigenvector representing the normal mode (for example,
        one of the columns of the second return value of
        `compute_normal_modes_generalised`).

    slice_z:

        Currently only the values 'z_min' and 'z_max' are supported
        (which correspond to the bottom/top layer of the mesh).
        Hopefully in the future `slice_z` will be allowed to have any
        numerical value.

    num_phase_colorbar_ticks:

        The number of tick labels for the phase colorbars. Currently
        this must be either 3 or 5 (default: 5).

    outfilename:

        If given, the plot will be saved to a file with this name
        (default: None).

    dpi:

        The resolution of the saved plot (ignored if `outfilename` is None).


    *Returns*

    The `matplotlib.Figure` containing the plot.

    """
    coords = sim.mesh.coordinates()

    if slice_z == 'z_min':
        slice_z = min(coords[:, 2])
    elif slice_z == 'z_max':
        slice_z = max(coords[:, 2])
    else:
        raise ValueError("Currently slice_z must have one of the values 'z_min' or 'z_max' (which correspond to the bottom/top layer of the mesh, respectively).")

    # Extract the boundary mesh from the full mesh (which is a 2D mesh
    # consisting only of the triangles on the boundary).
    mesh = sim.mesh
    boundary_mesh = df.BoundaryMesh(mesh, 'exterior')
    entity_map = boundary_mesh.entity_map(0).array()

    # Extract the submesh of the boundary mesh corresponding to the
    # bottom or top layer (whichever slice_z corresponds to).
    class Surface(df.SubDomain):
        def inside(self, pt, on_boundary):
            x, y, z = pt
            return (z >= slice_z - df.DOLFIN_EPS) and (z <= slice_z + df.DOLFIN_EPS)
    sub_domains = df.MeshFunction("size_t", boundary_mesh, 2)
    sub_domains.set_all(0)
    surface = Surface()
    surface.mark(sub_domains, 1)
    surface_layer = df.SubMesh(boundary_mesh, sub_domains, 1)
  
    # Original line that fails
    #parent_vertex_indices = surface_layer.data().array("parent_vertex_indices", 0)

    ## Hans' attempt to fix (only guessing)
    #parent_vertex_indices = surface_layer.data().array("parent_vertex_indices")
    
    
    try:
        # Legacy syntax (for dolfin <= 1.2 or so).
        # TODO: This should be removed in the future once dolfin 1.3 is released!
        parent_vertex_indices = surface_layer.data().mesh_function('parent_vertex_indices').array()
    except RuntimeError:
        # This is the correct syntax now, see:
        # http://fenicsproject.org/qa/185/entity-mapping-between-a-submesh-and-the-parent-mesh
        parent_vertex_indices = surface_layer.data().array('parent_vertex_indices', 0)

    #import IPython
    #IPython.embed()

    ######################################################################
    # XXX TODO: This function should probably not need access to the
    #           simulation object. Instead, the following calculations
    #           should be done externally, and w_3d should be passed
    #           as an argument directly to this function.
    m0_array = sim.m.copy()  # we assume that sim is relaxed!!
    Q, R, S, Mcross = compute_tangential_space_basis(m0_array.reshape(3, 1, -1))
    Qt = mf_transpose(Q).copy()
    

    n = sim.mesh.num_vertices()
    w_3d = mf_mult(Q, w.reshape((2, 1, n)))
    
    w_x = w_3d[0, 0, :]
    w_y = w_3d[1, 0, :]
    w_z = w_3d[2, 0, :]
    ######################################################################

    powers = {'x': np.absolute(w_x) ** 2,
              'y': np.absolute(w_y) ** 2,
              'z': np.absolute(w_z) ** 2}
    phases = {'x': np.angle(w_x),
              'y': np.angle(w_y),
              'z': np.angle(w_z)}

    V = df.FunctionSpace(sim.mesh, 'CG', 1)
    f = df.Function(V)
    
    ##FIXME: Weiwei: comment this two lines first, just to pass through the test
    #V_surface = df.FunctionSpace(surface_layer, 'CG', 1)
    #f_surface = df.Function(V_surface)

    def restrict_to_submesh(a):
        # Remark: We can't use df.interpolate here to interpolate the
        # function values from the full mesh on the submesh because it
        # sometimes crashes (probably due to rounding errors), even if we
        # set df.parameters["allow_extrapolation"]=True as they recommend
        # in the error message.
        #
        # Therefore we manually interpolate the function values here using
        # the vertex mappings determined above. This works fine if the
        # dofs are not re-ordered, but will probably cause problems in
        # parallel (or with dof reordering enabled).
        f.vector().set_local(a)
        f_array = f.vector().array()
        return f_array[[entity_map[i] for i in parent_vertex_indices]]

    surface_coords = surface_layer.coordinates()
    xvals = surface_coords[:, 0]
    yvals = surface_coords[:, 1]

    # We use the mesh triangulation provided by dolfin in case the
    # mesh has multiple disconnected regions (in which case matplotlib
    # would connect them).
    mesh_triang = tri.Triangulation(xvals, yvals, surface_layer.cells())

    # Determine the number of rows (<=2) and columns (<=3) in the plot
    num_rows = 0
    if plot_powers:
        num_rows +=1
    if plot_phases:
        num_rows +=1
    if num_rows == 0:
        raise ValueError("At least one of the arguments `plot_powers`, `plot_phases` must be True.")
    num_columns = len(components)


    def plot_mode_profile(ax, a, title=None, vmin=None, vmax=None, cmap=None, ticks=None, ticklabels=None):
        ax.set_aspect('equal')
        vals = restrict_to_submesh(a)
        trimesh = ax.tripcolor(mesh_triang, vals, shading='gouraud', cmap=cmap)
        ax.set_title(title)
        if show_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "5%", pad="3%")
            if vmin == None:
                vmin = min(vals)
            if vmax == None:
                vmax = max(vals)
            trimesh.set_clim(vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(trimesh, cax=cax, format=FormatStrFormatter('%g'),
                                ticks=ticks)
            if ticklabels != None:
                cbar.ax.set_yticklabels(ticklabels)
        if not show_axis_labels:
            ax.set_xticks([])
            ax.set_yticks([])
        if not show_axis_frames:
            ax.axis('off')

 
    fig = plt.figure(figsize=figsize)

    cnt = 1
    if plot_powers:
        for comp in components:
            ax = fig.add_subplot(num_rows, num_columns, cnt)
            plot_mode_profile(ax, powers[comp], title='m_{}'.format(comp), vmin=vmin_powers, cmap=cmap_powers)
            cnt += 1

    if plot_phases:
        if num_phase_colorbar_ticks == 3:
            ticks=[-pi, 0, pi]
            ticklabels=[u'-\u03C0', u'0', u'\u03C0']
        elif num_phase_colorbar_ticks == 5:
            ticks=[-pi, -pi/2, 0, pi/2, pi]
            ticklabels=[u'-\u03C0', u'-\u03C0/2', u'0', u'\u03C0/2', u'\u03C0']
        else:
            raise ValueError("'num_phase_colorbar_ticks' must be either 3 or 5. Got: {}".format(num_phase_colorbar_ticks))

        for comp in components:
            ax = fig.add_subplot(num_rows, num_columns, cnt)
            plot_mode_profile(ax, phases[comp], title='m_{}'.format(comp),
                              ticks=ticks, ticklabels=ticklabels,
                              vmin=-pi, vmax=+pi,
                              cmap=cmap_phases)
            cnt += 1

    bbox_extra_artists = []
    if figure_title != None:
        txt = plt.text(0.5, 1.0 + yshift_title, figure_title,
                        horizontalalignment='center',
                        fontsize=20,
                        transform = fig.transFigure)
        bbox_extra_artists.append(txt)
    num_axes = len(fig.axes)
    ax_annotate_powers = fig.axes[0]
    ax_annotate_phases= fig.axes[(num_axes // 2) if plot_powers else 0]
    if plot_powers:
        txt_power = plt.text(-0.1, 0.5, 'Power',
                             fontsize=16,
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation='vertical',
                             #transform=fig.transFigure)
                             transform=ax_annotate_powers.transAxes)
        bbox_extra_artists.append(txt_power)
    #
    #ax_topleft.text(0, 0, 'Power', ha='left', va='center', rotation=90)
    #
    #from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
    #box = TextArea("Power", textprops=dict(color="k", fontsize=20))
    #anchored_box = AnchoredOffsetbox(loc=3,
    #                                 child=box, pad=0.,
    #                                 frameon=False,
    #                                 bbox_to_anchor=(-0.1, 0.5),
    #                                 bbox_transform=ax.transAxes,
    #                                 borderpad=0.,
    #                                 )
    #ax_topleft.add_artist(anchored_box)
    #bbox_extra_artists.append(anchored_box)

    if plot_phases:
        txt_phase = plt.text(-0.1, 0.5, 'Phase',
                             fontsize=16,
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation='vertical',
                             #transform=fig.transFigure)
                             transform=ax_annotate_phases.transAxes)
        bbox_extra_artists.append(txt_phase)

    if outfilename != None:
        fig.savefig(outfilename, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight', dpi=dpi)

    return fig
