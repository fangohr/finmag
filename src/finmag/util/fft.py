from __future__ import division
from scipy.interpolate import InterpolatedUnivariateSpline
from finmag.util.helpers import probe
from glob import glob
from time import time
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
import logging
import matplotlib.cm as cm
from numpy import sin, cos, pi

logger = logging.getLogger("finmag")


def FFT_m(filename, t_step, t_ini=0, subtract_values=None):
    """
    Given a data file (e.g. in .ndt format), compute and return the Fourier
    transforms of the x, y and z components of the magnetisation m. The
    data is first resampled at regularly spaced intervals.

    *Arguments*

    filename:

        The data file. Each line must contain four numbers, representing
        the time step and the x, y, z components of m, respectively.

    t_step:

        Interval between consecutive time steps in the resampled data.

    t_ini:

        Initial time for the resampled data (all input data before
        this time is discarded). Defaults to zero.

    subtract_values:  None | 3-tuple of floats | 'first' | 'average'

        If specified, the given values are subtracted from the data
        before computing the Fourier transform. This can be used to
        avoid potentially large peaks at zero frequency. If a 3-tuple
        is given then it is interpreted as the three values to
        subtract from mx, my and mz, respectively. If 'first' or
        'average' is given, the first/average values of mx, my, mz are
        determined and subtracted.

    """
    # Load the data; extract time steps and magnetisation
    data = np.loadtxt(filename)
    ts, mx, my, mz = data.transpose()

    if subtract_values == 'first':
        mx -= mx[0]
        my -= my[0]
        mz -= mz[0]
    elif subtract_values == 'average':
        mx -= mx.mean()
        my -= my.mean()
        mz -= mz.mean()
    elif subtract_values != None:
        try:
            (sx, sy, sz) = subtract_values
            mx -= sx
            my -= sy
            mz -= sz
        except:
            raise ValueError("Unsupported value for 'subtract_values': {}".format(subtract_values))

    f_sample = 1/t_step  # sampling frequency
    t_max = ts[-1]

    # Interpolating functions for mx, my, mz
    f_mx = InterpolatedUnivariateSpline(ts, mx)
    f_my = InterpolatedUnivariateSpline(ts, my)
    f_mz = InterpolatedUnivariateSpline(ts, mz)

    # Sample the interpolating functions at regularly spaced time steps
    t_sampling = np.arange(t_ini, t_max, t_step)
    mx_resampled = [f_mx(t) for t in t_sampling]
    my_resampled = [f_my(t) for t in t_sampling]
    mz_resampled = [f_mz(t) for t in t_sampling]

    fft_mx = abs(np.fft.rfft(mx_resampled))
    fft_my = abs(np.fft.rfft(my_resampled))
    fft_mz = abs(np.fft.rfft(mz_resampled))
    n = len(fft_mx)

    fft_freq = np.fft.fftfreq(len(mx_resampled), t_step)[:n]

    return fft_freq, fft_mx, fft_my, fft_mz


def plot_FFT_m(filename, t_step, t_ini=0.0, subtract_values=None,
               components="xyz", xlim=None, figsize=None):
    """
    Plot the frequency spectrum of the components of the magnetisation m.

    The arguments `t_ini`, `t_step` and `subtract_values` have the
    same meaning as in the FFT_m function.

    `components` can be a string or a list containing the components
    to plot. Default: 'xyz'.

    Returns the matplotlib Axis instance containing the plot.
    """
    if not set(components).issubset("xyz"):
        raise ValueError("Components must only contain 'x', 'y' and 'z'. "
                         "Got: {}".format(components))

    fft_freq, fft_mx, fft_my, fft_mz = FFT_m(filename, t_step, t_ini, subtract_values)
    fft_freq_GHz = fft_freq / 1e9
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if 'x' in components: ax.plot(fft_freq_GHz, fft_mx, '.-', label=r'FFT of $m_x$')
    if 'y' in components: ax.plot(fft_freq_GHz, fft_my, '.-', label=r'FFT of $m_y$')
    if 'z' in components: ax.plot(fft_freq_GHz, fft_mz, '.-', label=r'FFT of $m_z$')
    ax.set_xlabel('fGHz')
    ax.set_ylabel('Amplitude')
    fmin = int(min(fft_freq) / 1e9)
    fmax = int(max(fft_freq) / 1e9)
    ax.set_xticks(np.arange(fmin, fmax))
    ax.set_xlim(xlim)
    plt.legend()
    ax.grid()

    return ax


def fft_at_probing_points(dolfin_funcs, pts):
    """
    Given a list of dolfin Functions (on the same mesh) representing
    field values at different time steps, as well as the x-, y- and
    z-coordinates of some probing points, compute and return the
    discrete Fourier transforms (over time) of these functions at each
    point.


    *Arguments*

    dolfin_funcs:  list of dolfin.Function

        List of functions representing (say) the field values at
        different time steps.

    pts:     points: numpy.array

        An array of points where the FFT should be computed. Can
        have arbitrary shape, except that the last axis must have
        dimension 3. For example, if pts.shape == (10,20,5,3) then
        the FFT is computeed at all points on a regular grid of
        size 10 x 20 x 5.


    *Returns*

    A numpy.array `res` of the same shape as X, Y and Z, but with an
    additional first axis which contains the coefficients if the Fourier
    transform. For example, res[:, ...] represents the Fourier transform
    over time as probed at the point (X[...], Y[...], Z[...]).

    """
    vals_probed = np.ma.masked_array([probe(f, pts) for f in dolfin_funcs])
    #vals_fft = np.ma.masked_array(np.fft.fft(vals_probed, axis=0),
    #                              mask=np.ma.getmask(vals_probed))
    #freqs = np.fft.fftfreq(
    n = (len(dolfin_funcs) // 2) + 1
    vals_fft = np.ma.masked_array(np.fft.rfft(vals_probed, axis=0),
                                  mask=np.ma.getmask(vals_probed[:n, ...]))

    return vals_fft


def plot_spatially_resolved_normal_modes(m_vals_on_grid, idx_fourier_coeff,
                                         t_step=None, figsize=None,
                                         yshift_title=1.5, cmap=cm.jet):
    """
    XXX Warning: The interface for this function hasn't matured yet,
                 so be prepared for it to change in the future.

    Given the time evolution of the magnetisation (probed on a regular
    grid), compute and plot the normal mode shapes corresponding to a
    certain frequency. More precisely, this plots the absolute values
    and the phase of the Fourier coefficients at each probing point
    for each of m_x, m_y and m_z.

    m_vals_on_grid : numpy.array of shape (nt, nx, ny, nz, 3)
        Array containing the magnetisation values probed at regular
        time steps on a regular grid. Here `nt` is the number of time
        steps and `nx`, `ny`, `nz` determine the size of the probing
        grid. Thus The 3-vector m_vals_on_grid[t0, x0, y0, z0]
        contains the magnetisation at time `t0` probed at the grid
        point with coordinates (x0, y0, z0).

    idx_fourier_coeff : int

        Index of the Fourier coefficient for which to compute the normal
        modes. This should be between 0 and (number of files - 1).

    t_step : float (optional)

        The interval between subsequent time steps of the probed
        magnetisation values. This is only relevant to print the
        frequency of the normal mode in the figure title.

    figsize : pair of float

        The size of the resulting figure.

    yshift_title : float

        Amount by which the title should be shifted up (this can be
        used to tweak the figure if the title overlaps with one of the
        colorbars, say).

    cmap :

        The colormap to use.

    *Returns*

    The matplotlib figure containing.
    """
    n = (m_vals_on_grid.shape[0] // 2) + 1
    fft_vals = np.ma.masked_array(np.fft.rfft(m_vals_on_grid, axis=0),
                                  mask=np.ma.getmask(m_vals_on_grid[:n, ...]))
    fig = plt.figure(figsize=figsize)
    axes = []
    for k in [0, 1, 2]:
        ax = fig.add_subplot(2, 3, k+1)
        ax.set_title('m_{}'.format('xyz'[k]))
        im = ax.imshow(abs(fft_vals[idx_fourier_coeff, :, :, 0, k]), origin='lower', cmap=cmap)
        fig.colorbar(im)
        axes.append(ax)

        ax = fig.add_subplot(2, 3, k+3+1)
        axes.append(ax)
        ax.set_title('m_{} (phase)'.format('xyz'[k]))
        im = ax.imshow(np.angle(fft_vals[idx_fourier_coeff, :, :, 0, k], deg=True), origin='lower', cmap=cmap)
        fig.colorbar(im)
    if t_step != None:
        # XXX TODO: Which value of nn is the correct one?
        #nn = n
        nn = len(m_vals_on_grid)
        fft_freqs = np.fft.fftfreq(nn, t_step)[:nn]
        figure_title = "Mode shapes for frequency f={:.2f} GHz".format(fft_freqs[idx_fourier_coeff] / 1e9)
        plt.text(0.5, yshift_title, figure_title,
             horizontalalignment='center',
             fontsize=20,
             transform = axes[2].transAxes)
    else:
        logger.warning("Omitting figure title because no t_step argument was specified.")

    return fig


def filter_frequency_component(signal, k, t_start, t_end, ts_sampling=None):
    """
    Filter the given signal by only keeping the frequency component
    corresponding to the k-th Fourier coefficient.

    XXX TODO: This is probably a bad interface. We should require a
    frequency as the input and compute the index automatically.

    *Arguments*

    signal : numpy array

        Must be a 2d array, where the first index represents time and
        the second index the data. Thus `signal[i, :]` is the signal
        at time `i`.

    k : int

        The index of the Fourier coefficient which should be used for
        filtering.

    t_start, t_end : float

        First and last time step of the signal. Note that this function
        assumes that the time steps are evenly spaced.

    ts_sampling : numpy array

        The time steps at which the filtered signal should be evaluated.

    """
    n = len(signal)

    # Fourier transform the signal
    t0 = time()
    rfft_vals = np.fft.rfft(signal, axis=0)
    t1 = time()
    logger.debug("Computing the Fourier transform took {:.2g} seconds".format(t1-t0))
    #rfft_freqs = np.arange(n // 2 + 1) / (dt*n)

    # Only keep the Fourier coefficients for the given frequency component
    A_k = rfft_vals[k]

    # Since the DFT formula know nothing about the true timesteps at which the
    # signal is given, we need to rescale the sampling timesteps so that they
    # lie in the interval [0, 2*pi*k]
    if ts_sampling is None:
        ts_rescaled = (2 * pi * k * np.arange(n) / n)
    else:
        ts_rescaled = (ts_sampling - t_start) / (t_end - t_start) * 2 * pi * k * (n - 1) / n

    # 'Transpose' the 1D vector so that the linear combination below
    # produces the correct 2D output format.
    ts_rescaled = ts_rescaled[:, np.newaxis]

    signal_filtered = 2.0/n * (A_k.real * cos(ts_rescaled) - A_k.imag * sin(ts_rescaled))
    return signal_filtered


def export_normal_mode_animation(npy_files, outfilename, mesh, ts, k, dm_only=False):
    """
    Read in a bunch of .npy files (containing the magnetisation
    sampled at regular time steps) and export an animation of the
    normal mode corresponding to a specific frequency.

    npy_files :  string (including shell wildcards) or list of filenames

        The list of files containing the magnetisation values sampled
        at the mesh vertices. There should be one file per stime step.

    outfilename :  string

        Name of the .pvd file to which the animation is exported.

    mesh :  dolfin.Mesh or string

        The mesh (or name of the .xml.gz file containing the mesh) on
        which the magnetisation was sampled.

    t_step :  float

        The interval between subsequent time steps.

    idx:  int

        Index of the frequency for which the normal mode is to be plotted.

    dm_only :  bool (optional)

        If False (the default), plots `m0 + dm(t)`, where m0 is the
        average magnetisation and dm(t) the (spatially varying)
        oscillation corresponding to the frequency of the normal mode.
        If True, only `dm(t)` is plotted.
    """
    if isinstance(npy_files, str):
        files = sorted(glob(npy_files))
    else:
        files = list(npy_files)

    if isinstance(mesh, str):
        mesh = df.Mesh(mesh)

    N = len(files)  # number of timesteps
    num_nodes = len(mesh.coordinates())

    # Read in the signal
    signal = np.empty([N, 3*num_nodes])
    for (i, filename) in enumerate(files):
        signal[i, :] = np.load(filename)
    logger.debug("Signal occupies {} MB of memory".format(signal.nbytes / 1024**2))

    # Fourier transform the signal
    t0 = time()
    fft_vals = np.fft.rfft(signal, axis=0)
    t1 = time()
    logger.debug("Computing the Fourier transform took {:.2g} seconds".format(t1-t0))
    fft_freqs = np.fft.fftfreq(N, d=t_step)[:len(fft_vals)]

    # XXX TODO: The full invere Fourier transform should be replaced by the explicit formula for the single coefficient to save memory and computational effort
    fft_vals_filtered = np.zeros(fft_vals.shape, dtype=fft_vals.dtype)
    fft_vals_filtered[idx] = fft_vals[idx]
    t0 = time()
    signal_inv_filtered = np.fft.irfft(fft_vals_filtered, N, axis=0)
    t1 = time()
    logger.debug("Computing the inverse Fourier transform took {:.2f} seconds.".format(t1-t0))

    # Determine a sensible scaling factor so that the oscillations are visible but not too large.
    # N.B.: Even though it looks convoluted, computing the maximum value
    # in this iterated way is actually much faster than doing it directly.
    maxval = max(np.max(signal_inv_filtered, axis=0))
    logger.debug("Maximum value of the signal: {}".format(maxval))
    scaling_factor = 0.3 / maxval
    logger.debug("Scaling factor: {}".format(scaling_factor))

    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    func = df.Function(V)
    if dm_only == True:
        signal_normal_mode = scaling_factor * signal_inv_filtered
    else:
        signal_normal_mode = signal.mean(axis=0).T + scaling_factor * signal_inv_filtered

    logger.debug("Saving normal mode to file '{}'.".format(outfilename))
    t0 = time()
    f = df.File(outfilename, 'compressed')
    # XXX TODO: We need the strange temporary array 'aaa' here because if we write the values directly into func.vector() then they end up being different (as illustrated below)!!!
    aaa = np.empty(3*num_nodes)
    #for i in xrange(len(ts)):
    for i in xrange(20):
        if i % 20 == 0:
            print "i={} ".format(i),
        aaa[:] = signal_normal_mode[i][:]
        func.vector()[:] = aaa
        f << func
    t1 = time()
    logger.debug("Saving the data took {} seconds".format(t1 - t0))
