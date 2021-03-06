from __future__ import division
from scipy.interpolate import InterpolatedUnivariateSpline
from finmag.util.helpers import probe
from finmag.util.fileio import Tablereader
from glob import glob
from time import time
import numpy as np
import dolfin as df
import logging
from numpy import sin, cos, pi

logger = logging.getLogger("finmag")


def _aux_fft_m(filename, t_step=None, t_ini=None, t_end=None, subtract_values='first', vertex_indices=None):
    """
    Helper function to compute the Fourier transform of magnetisation
    data, which is either read from a single .ndt file (for spatially
    averaged magnetisation) or from a series of .npy files (for
    spatially resolved data). If necessary, the data is first
    resampled at regularly spaced time intervals.

    """
    # Load the data; extract time steps and magnetisation
    if filename.endswith('.ndt'):
        data = Tablereader(filename)
        ts = data['time']
        mx = data['m_x']
        my = data['m_y']
        mz = data['m_z']
    elif filename.endswith('.npy'):
        if t_ini == None or t_end == None or t_step == None:
            raise ValueError(
                "If 'filename' represents a series of .npy files then t_ini, t_end and t_step must be given explicitly.")
        num_steps = int(np.round((t_end - t_ini) / t_step)) + 1
        ts = np.linspace(t_ini, t_end, num_steps)
        npy_files = sorted(glob(filename))
        N = len(npy_files)
        if (N != len(ts)):
            raise RuntimeError(
                "Number of timesteps (= {}) does not match number of .npy files found ({}). Aborting.".format(len(ts), N))
        logger.debug("Found {} .npy files.".format(N))

        if vertex_indices != None:
            num_vertices = len(vertex_indices)
        else:
            num_vertices = len(np.load(npy_files[0])) // 3
            vertex_indices = np.arange(num_vertices)
        mx = np.zeros((N, num_vertices))
        my = np.zeros((N, num_vertices))
        mz = np.zeros((N, num_vertices))

        for (i, npyfile) in enumerate(npy_files):
            a = np.load(npyfile)
            aa = a.reshape(3, -1)
            mx[i, :] = aa[0, vertex_indices]
            my[i, :] = aa[1, vertex_indices]
            mz[i, :] = aa[2, vertex_indices]
    else:
        raise ValueError(
            "Expected a single .ndt file or a wildcard pattern referring to a series of .npy files. Got: {}.".format(filename))

    # If requested, subtract the first value of the time series
    # (= relaxed state), or the average, or some other value.
    if subtract_values == 'first':
        mx -= mx[0]
        my -= my[0]
        mz -= mz[0]
    elif subtract_values == 'average':
        mx -= mx.mean(axis=0)
        my -= my.mean(axis=0)
        mz -= mz.mean(axis=0)
    elif subtract_values != None:
        try:
            (sx, sy, sz) = subtract_values
            mx -= sx
            my -= sy
            mz -= sz
        except:
            raise ValueError(
                "Unsupported value for 'subtract_values': {}".format(subtract_values))

    # Try to guess sensible values of t_ini, t_end and t_step if none
    # were specified.
    if t_step is None:
        t_step = ts[1] - ts[0]
        if not(np.allclose(t_step, np.diff(ts))):
            raise ValueError("A value for t_step must be explicitly provided "
                             "since timesteps in the file '{}' are not "
                             "equidistantly spaced.".format(filename))
    f_sample = 1. / t_step  # sampling frequency
    if t_ini is None:
        t_ini = ts[0]
    if t_end is None:
        t_end = ts[-1]

    # Resample the magnetisation if it was recorded at uneven
    # timesteps (or not at the timesteps specified for the Fourier
    # transform).
    eps = 1e-8
    num_steps = int(np.round((t_end - t_ini) / t_step)) + 1
    ts_resampled = np.linspace(t_ini, t_end, num_steps)
    if (ts.shape == ts_resampled.shape and np.allclose(ts, ts_resampled, atol=0, rtol=1e-7)):
        #logger.debug("Data already given at the specified regular intervals. No need to resample.")
        mx_resampled = mx
        my_resampled = my
        mz_resampled = mz
    else:
        logger.debug("Resampling data at specified timesteps.")

        # Interpolating functions for mx, my, mz
        f_mx = InterpolatedUnivariateSpline(ts, mx)
        f_my = InterpolatedUnivariateSpline(ts, my)
        f_mz = InterpolatedUnivariateSpline(ts, mz)

        # Sample the interpolating functions at regularly spaced time steps
        mx_resampled = np.array([f_mx(t) for t in ts_resampled])
        my_resampled = np.array([f_my(t) for t in ts_resampled])
        mz_resampled = np.array([f_mz(t) for t in ts_resampled])

    fft_mx = np.fft.rfft(mx_resampled, axis=0)
    fft_my = np.fft.rfft(my_resampled, axis=0)
    fft_mz = np.fft.rfft(mz_resampled, axis=0)

    # When using np.fft.fftfreq, the last frequency sometimes becomes
    # negative; to avoid this we compute the frequencies by hand.
    n = len(fft_mx)
    freqs = np.arange(n) / (t_step * len(ts_resampled))

    return freqs, fft_mx, fft_my, fft_mz


def filter_frequency_component(signal, k, t_start, t_end, ts_sampling=None):
    """
    Filter the given signal by only keeping the frequency component
    corresponding to the k-th Fourier coefficient.

    XXX TODO: This is probably not the best interface. We should require
              a frequency as the input and compute the index automatically.

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
    logger.debug(
        "Computing the Fourier transform took {:.2g} seconds".format(t1 - t0))
    #rfft_freqs = np.arange(n // 2 + 1) / (dt*n)

    # Only keep the Fourier coefficients for the given frequency component
    A_k = rfft_vals[k]

    # Since the DFT formula know nothing about the true timesteps at which the
    # signal is given, we need to rescale the sampling timesteps so that they
    # lie in the interval [0, 2*pi*k]
    if ts_sampling is None:
        ts_rescaled = (2 * pi * k * np.arange(n) / n)
    else:
        ts_rescaled = (ts_sampling - t_start) / \
            (t_end - t_start) * 2 * pi * k * (n - 1) / n

    # 'Transpose' the 1D vector so that the linear combination below
    # produces the correct 2D output format.
    ts_rescaled = ts_rescaled[:, np.newaxis]

    signal_filtered = 2.0 / n * \
        (A_k.real * cos(ts_rescaled) - A_k.imag * sin(ts_rescaled))
    return signal_filtered


def compute_power_spectral_density(filename, t_step=None, t_ini=None, t_end=None, subtract_values='first', restrict_to_vertices=None):
    """
    Compute the power spectral densities (= squares of the absolute
    values of the Fourier coefficients) of the x, y and z components
    of the magnetisation m, where the magnetisation data is either
    read from a .ndt file (for spatially averages magnetisation; not
    recommended, see below) or from a series of data files in .npy
    format (recommended). If necessary, the data is first resampled at
    regularly spaced time intervals.

    Note that this performs a real-valued Fourier transform (i.e. it
    uses np.fft.rfft internally) and thus does not return Fourier
    coefficients belonging to negative frequencies.

    If `filename` is the name of a single .ndt file then the Fourier
    transform of the average magneisation is computed. Note that this
    is *not* recommended since it may not detect modes that have
    certain symmetries which are averaged out by this method. A better
    way is to pass in a series of .npy files, which takes the
    spatially resolved magnetisation into account.


    *Arguments*

    filename:

        The .ndt file or .npy files containing the magnetisation values.
        In the second case a pattern should be given (e.g. 'm_ringdown*.npy').

    t_step:

        Interval between consecutive time steps in the resampled data.
        If the timesteps in the .ndt file are equidistantly spaced,
        this distance is used as the default value.

    t_ini:

        Initial time for the resampled data (all input data before
        this time is discarded). Defaults to the first time step saved
        in the .ndt data file.

    t_end:

        Last time step for the resampled data (all input data after
        this time is discarded). Defaults to the last time step saved
        in the .ndt data file.

    subtract_values:  None | 3-tuple of floats | 'first' | 'average'

        If specified, the given values are subtracted from the data
        before computing the Fourier transform. This can be used to
        avoid potentially large peaks at zero frequency. If a 3-tuple
        is given then it is interpreted as the three values to
        subtract from mx, my and mz, respectively. If 'first' or
        'average' is given, the first/average values of mx, my, mz are
        determined and subtracted.

    restrict_to_vertices:

        This argument only has an effect when computing the spectrum
        from spatially resolved magnetisation data (i.e., from a
        series of .npy files). If `restrict_to_vertices` is `None`
        (the default), all mesh vertices are taken into account.
        Otherwise `restrict_to_vertices` should be a list of vertex
        indices. The spectrum then is then computed for the
        magnetisation dynamics at these particular vertices only.


    *Returns*

    Returns a tuple (freqs, psd_mx, psd_my, psd_mz), where psd_mx,
    psd_my, psd_mz are the power spectral densities of the x/y/z-component
    of the magnetisation and freqs are the corresponding frequencies.

    """
    if not filename.endswith('.npy') and restrict_to_vertices != None:
        logger.warning("Ignoring argument 'restrict_to_vertices' because it "
                       "can only be used when reading the spatially resolved "
                       "magnetisation from a list of .npy files.")

    freqs, fft_mx, fft_my, fft_mz = \
        _aux_fft_m(filename, t_step=t_step, t_ini=t_ini, t_end=t_end,
                   subtract_values=subtract_values, vertex_indices=restrict_to_vertices)

    psd_mx = np.absolute(fft_mx) ** 2
    psd_my = np.absolute(fft_my) ** 2
    psd_mz = np.absolute(fft_mz) ** 2

    if filename.endswith('.npy'):
        # Compute the power spectra and then do the spatial average
        psd_mx = psd_mx.sum(axis=-1)
        psd_my = psd_my.sum(axis=-1)
        psd_mz = psd_mz.sum(axis=-1)

    return freqs, psd_mx, psd_my, psd_mz


def find_peak_near_frequency(f_approx, fft_freqs, fft_vals):
    """
    Given the Fourier spectrum of one or multiple magnetisation
    components, find the peak closest to the given frequency.

    This is a helper function for interactive use that allows to
    quickly determine the exact location of a peak for which an
    approximate location is known (e.g from a plot).


    *Example*

    >>> fft_freqs, fft_mx, fft_my, fft_mz = FFT_m('simulation.ndt', t_step=1e-11)
    >>> # Let's say that we have identified a peak near 5.4 GHz in fft_my (e.g. from a plot)
    >>> idx = find_peak_near_frequency(5.4e9, fft_freqs, fft_my)


    *Arguments*

    f_approx :  float

        The frequency near which a peak is to be found.

    fft_freqs :  array

        An array of frequencies (as returned by FFT_m, for example).
        The values are assumed to be ordered from smallest to largest.

    fft_vals :  array or list of arrays

        The Fourier transform of one magnetisation component (m_x, m_y or m_z).

    *Returns*

    A pair `(freq, idx)`, where `idx` is the index of the exact peak
    in the array fft_freqs and `freq` is the associated frequency,
    i.e. freq=fft_freqs[idx].

    """
    try:
        from scipy.signal import argrelmax
    except ImportError:
        raise NotImplementedError(
            "Need scipy >= 0.11, please install the latest version via: 'sudo pip install -U scipy'")

    if not len(fft_freqs) == len(fft_vals):
        raise ValueError("The arrays `fft_freqs` and `fft_vals` "
                         "must have the same length, "
                         "but {} != {}".format(len(fft_freqs), len(fft_vals)))

    fft_freqs = np.asarray(fft_freqs)
    fft_vals = np.asarray(fft_vals)
    N = len(fft_freqs) - 1  # last valid index

    peak_indices = list(argrelmax(fft_vals)[0])

    # Check boundary extrema
    if fft_vals[0] > fft_vals[1]:
        peak_indices.insert(0, 0)
    if fft_vals[N - 1] < fft_vals[N]:
        peak_indices.append(N)

    closest_peak_idx = peak_indices[
        np.argmin(np.abs(fft_freqs[peak_indices] - f_approx))]
    logger.debug("Found peak at {:.3f} GHz (index: {})".format(
        fft_freqs[closest_peak_idx] / 1e9, closest_peak_idx))
    return fft_freqs[closest_peak_idx], closest_peak_idx


def _plot_spectrum(freqs, psd_mx, psd_my, psd_mz, components="xyz", log=False,
                   xlim=None, ylim=None, ticks=21, figsize=None, title="", outfilename=None):
    """
    Internal helper function to plot certain components of the spectrum.
    This is only separated out from plot_power_spectral_density so that
    it can be re-used elsewhere, e.g. in the NormalModeSimulation class.

    """
    import matplotlib.pyplot as plt
    freqs_GHz = freqs / 1e9
    if log:
        psd_mx = np.log(psd_mx)
        psd_my = np.log(psd_my)
        psd_mz = np.log(psd_mz)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    if 'x' in components:
        ax.plot(freqs_GHz, psd_mx, '.-', label=r'$m_x$')
    if 'y' in components:
        ax.plot(freqs_GHz, psd_my, '.-', label=r'$m_y$')
    if 'z' in components:
        ax.plot(freqs_GHz, psd_mz, '.-', label=r'$m_z$')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power spectral density{}'.format(' (log)' if log else ''))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.linspace(xmin, xmax, ticks))
    ax.legend(loc='best')
    ax.grid()

    if title:
        ax.set_title(title)

    if outfilename is not None:
        fig.savefig(outfilename)

    return fig


def plot_power_spectral_density(filename, t_step=None, t_ini=None, t_end=None, subtract_values='first', components="xyz",
                                log=False, xlim=None, ylim=None, ticks=21, figsize=None, title="", outfilename=None, restrict_to_vertices=None):
    """
    Plot the power spectral density of the components of the magnetisation m.

    The arguments `t_step`, `t_ini`, `t_end` and `subtract_values` have the
    same meaning as in the function `FFT_m`.

    *Arguments*

    components:  string | list

        A string or list containing the components to plot. Default: 'xyz'.

    log:  bool

        If True (the default), the y-axis is plotted on a log scale.

    figsize:  pair of floats

        The size of the resulting figure.

    title:   string

        The figure title.


    *Returns*

    The matplotlib Figure instance containing the plot. If
    `outfilename` is not None, it also saves the plot to the
    specified file.

    """
    if not set(components).issubset("xyz"):
        raise ValueError("Components must only contain 'x', 'y' and 'z'. "
                         "Got: {}".format(components))

    freqs, psd_mx, psd_my, psd_mz = \
        compute_power_spectral_density(filename, t_step, t_ini=t_ini, t_end=t_end,
                                       subtract_values=subtract_values, restrict_to_vertices=restrict_to_vertices)

    return _plot_spectrum(freqs, psd_mx, psd_my, psd_mz, components=components,
                          log=log, xlim=xlim, ylim=ylim, ticks=ticks,
                          figsize=figsize, title=title, outfilename=outfilename)


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
    # vals_fft = np.ma.masked_array(np.fft.fft(vals_probed, axis=0),
    #                              mask=np.ma.getmask(vals_probed))
    # freqs = np.fft.fftfreq(
    n = (len(dolfin_funcs) // 2) + 1
    vals_fft = np.ma.masked_array(np.fft.rfft(vals_probed, axis=0),
                                  mask=np.ma.getmask(vals_probed[:n, ...]))

    return vals_fft


def plot_spatially_resolved_normal_modes(m_vals_on_grid, idx_fourier_coeff,
                                         t_step=None, figsize=None, yshift_title=1.5,
                                         show_colorbars=True, cmap=None):
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

    show_colorbars : bool

        Whether to show a colorbar in each subplot (default: True).

    cmap :

        The colormap to use.

    *Returns*

    The matplotlib figure containing.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if cmap is None:
        cmap = cm.jet
    n = (m_vals_on_grid.shape[0] // 2) + 1
    fft_vals = np.ma.masked_array(np.fft.rfft(m_vals_on_grid, axis=0),
                                  mask=np.ma.getmask(m_vals_on_grid[:n, ...]))
    fig = plt.figure(figsize=figsize)
    axes = []
    for k in [0, 1, 2]:
        ax = fig.add_subplot(2, 3, k + 1)
        ax.set_title('m_{}'.format('xyz'[k]))
        im = ax.imshow(
            abs(fft_vals[idx_fourier_coeff, :, :, 0, k]), origin='lower', cmap=cmap)
        if show_colorbars:
            fig.colorbar(im)
        axes.append(ax)

        ax = fig.add_subplot(2, 3, k + 3 + 1)
        axes.append(ax)
        ax.set_title('m_{} (phase)'.format('xyz'[k]))
        im = ax.imshow(np.angle(
            fft_vals[idx_fourier_coeff, :, :, 0, k], deg=True), origin='lower', cmap=cmap)
        if show_colorbars:
            fig.colorbar(im)
    if t_step != None:
        # XXX TODO: Which value of nn is the correct one?
        #nn = n
        nn = len(m_vals_on_grid)
        fft_freqs = np.fft.fftfreq(nn, t_step)[:nn]
        figure_title = "Mode shapes for frequency f={:.2f} GHz".format(
            fft_freqs[idx_fourier_coeff] / 1e9)
        plt.text(0.5, yshift_title, figure_title,
                 horizontalalignment='center',
                 fontsize=20,
                 transform=axes[2].transAxes)
    else:
        logger.warning(
            "Omitting figure title because no t_step argument was specified.")
    plt.tight_layout()

    return fig


def export_normal_mode_animation_from_ringdown(npy_files, outfilename, mesh, t_step, k, scaling=0.2, dm_only=False, num_cycles=1, num_frames_per_cycle=20):
    """
    Read a bunch of .npy files (containing the magnetisation sampled
    at regular time steps) and export an animation of the normal mode
    corresponding to a specific frequency.

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

    k:  int

        Index of the frequency for which the normal mode is to be plotted.

    scaling : float

        If `dm_only` is False, this determines the maximum size of the
        oscillation (relative to the magnetisation vector) in the
        visualisation. If `dm_only` is True, this has no effect.

    dm_only :  bool (optional)

        If False (the default), plots `m0 + scaling*dm(t)`, where m0 is the
        average magnetisation and dm(t) the (spatially varying)
        oscillation corresponding to the frequency of the normal mode.
        If True, only `dm(t)` is plotted.

    num_cycles :  int

        The number of cycles to be animated (default: 1).

    num_frames_per_cycle :  int

        The number of snapshot per cycle to be exported (default: 20). Thus the
        total number of exported frames is (num_frames_per_cycle * num_cycles).

    """
    files = sorted(glob(npy_files)) if isinstance(
        npy_files, str) else list(npy_files)
    if len(files) == 0:
        logger.error("Cannot produce normal mode animation. No input .npy "
                     "files found matching '{}'".format(npy_files))
        return

    if isinstance(mesh, str):
        mesh = df.Mesh(mesh)

    N = len(files)  # number of timesteps
    num_nodes = mesh.num_vertices()

    # Read in the magnetisation dynamics from each .npy file and store
    # it as successive time steps in the array 'signal'.
    signal = np.empty([N, 3 * num_nodes])
    for (i, filename) in enumerate(files):
        signal[i, :] = np.load(filename)
    logger.debug("Array with magnetisation dynamics occupies "
                 "{} MB of memory".format(signal.nbytes / 1024 ** 2))

    # Fourier transform the signal
    t0 = time()
    fft_vals = np.fft.rfft(signal, axis=0)
    t1 = time()
    logger.debug(
        "Computing the Fourier transform took {:.2g} seconds".format(t1 - t0))
    fft_freqs = np.fft.fftfreq(N, d=t_step)[:len(fft_vals)]

    # Only keep the k-th Fourier coefficient at each mesh node
    # (combined in the array A_k).
    A_k = fft_vals[k]
    abs_k = np.abs(A_k)[np.newaxis, :]
    theta_k = np.angle(A_k)[np.newaxis, :]

    num_frames = num_frames_per_cycle * num_cycles
    signal_filtered = np.empty([num_frames, 3 * num_nodes])

    # frequency associated with the k-th Fourier coefficient
    omega = fft_freqs[k]
    cycle_length = 1.0 / omega
    timesteps = np.linspace(
        0, num_cycles * cycle_length, num_frames, endpoint=False)[:, np.newaxis]
    t_end = (N - 1) * t_step

    # Compute 'snapshots' of the oscillation and store them in signal_filtered
    #
    # TODO: Write a unit test for this formula, just to be 100% sure
    #       that it is correct!
    signal_filtered = 2.0 / N * abs_k * \
        cos(k * 2 * pi * timesteps / t_end + theta_k)

    # Determine a sensible scaling factor so that the oscillations are
    # visible but not too large. (Note that, even though it looks
    # convoluted, computing the maximum value in this iterated way is
    # actually much faster than doing it directly.)
    maxval = max(np.max(signal_filtered, axis=0))
    logger.debug("Maximum value of the signal: {}".format(maxval))

    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    func = df.Function(V)
    func.rename('m', 'magnetisation')
    if dm_only == True:
        signal_normal_mode = 1 / maxval * signal_filtered
    else:
        signal_normal_mode = signal.mean(
            axis=0).T + scaling / maxval * signal_filtered

    # XXX TODO: Should check for an existing file and ask for confirmation
    # whether it should be overwritten!
    logger.debug(
        "Saving normal mode animation to file '{}'.".format(outfilename))
    t0 = time()
    f = df.File(outfilename, 'compressed')
    # XXX TODO: We need the strange temporary array 'aaa' here because
    #           if we write the values directly into func.vector()
    #           then they end up being different (as illustrated in
    #           the code that is commented out)!!!
    aaa = np.empty(3 * num_nodes)
    # for i in xrange(len(ts)):
    # for i in xrange(20):
    for i in xrange(num_frames):
        # if i % 20 == 0:
        #    print "i={} ".format(i),
        #    import sys
        #    sys.stdout.flush()
        aaa[:] = signal_normal_mode[i][:]
        func.vector()[:] = aaa
        f << func
    t1 = time()
    logger.debug("Saving the data took {} seconds".format(t1 - t0))


# Shorter aliases that are easier to type
compute_psd = compute_power_spectral_density
plot_psd = plot_power_spectral_density
