from __future__ import division
from scipy.interpolate import InterpolatedUnivariateSpline
from finmag.util.helpers import probe
from finmag.util.fileio import Tablereader
from glob import glob
from time import time
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
import logging
import matplotlib.cm as cm
from numpy import sin, cos, pi

logger = logging.getLogger("finmag")


def _aux_fft_m(filename, t_step=None, t_ini=None, t_end=None, subtract_values='average'):
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
            raise ValueError("If 'filename' represents a series of .npy files then t_ini, t_end and t_step must be given explicitly.")
        num_steps = int(np.round((t_end - t_ini) / t_step)) + 1
        ts = np.linspace(t_ini, t_end, num_steps)
        npy_files = sorted(glob(filename))
        N = len(npy_files)
        if (N != len(ts)):
            raise RuntimeError("Number of timesteps (= {}) does not match number of .npy files found ({}). Aborting.".format(len(ts), N))
        logger.debug("Found {} .npy files.".format(N))

        num_timesteps = len(np.load(npy_files[0])) // 3
        mx = np.zeros((N, num_timesteps))
        my = np.zeros((N, num_timesteps))
        mz = np.zeros((N, num_timesteps))

        for (i, npyfile) in enumerate(npy_files):
            a = np.load(npyfile)
            aa = a.reshape(3, -1)
            mx[i, :] = aa[0]
            my[i, :] = aa[1]
            mz[i, :] = aa[2]
    else:
        raise ValueError("Expected a single .ndt file or a wildcard pattern referring to a series of .npy files. Got: {}.".format(filename))

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
            raise ValueError("Unsupported value for 'subtract_values': {}".format(subtract_values))

    # Try to guess sensible values of t_ini, t_end and t_step if none
    # were specified.
    if t_step is None:
        t_step = ts[1] - ts[0]
        if not(np.allclose(t_step, np.diff(ts))):
            raise ValueError("A value for t_step must be explicitly provided "
                             "since timesteps in the file '{}' are not "
                             "equidistantly spaced.".format(filename))
    f_sample = 1. / t_step  # sampling frequency
    if t_ini is None: t_ini = ts[0]
    if t_end is None: t_end = ts[-1]

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
    freqs = np.arange(n) / (t_step*len(ts_resampled))

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


def power_spectral_density(filename, t_step=None, t_ini=None, t_end=None, subtract_values='average'):
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


    *Returns*

    Returns a tuple (freqs, psd_mx, psd_my, psd_mz), where psd_mx,
    psd_my, psd_mz are the power spectral densities of the x/y/z-component
    of the magnetisation and freqs are the corresponding frequencies.

    """

    freqs, fft_mx, fft_my, fft_mz = \
        _aux_fft_m(filename, t_step=t_step, t_ini=t_ini, t_end=t_end, subtract_values=subtract_values)

    psd_mx = np.absolute(fft_mx)**2
    psd_my = np.absolute(fft_my)**2
    psd_mz = np.absolute(fft_mz)**2

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
        raise NotImplementedError("Need scipy >= 0.11, please install the latest version via: 'sudo pip install -U scipy'")

    if not len(fft_freqs) == len(fft_vals):
        raise ValueError("The arrays `fft_freqs` and `fft_vals` "
                         "must have the same length, "
                         "but {} != {}".format(len(fft_freqs), len(fft_vals)))

    fft_freqs = np.asarray(fft_freqs)
    fft_vals = np.asarray(fft_vals)
    N = len(fft_freqs) - 1  # last valid index

    peak_indices = list(argrelmax(fft_vals)[0])

    # Check boundary extrema
    if fft_vals[0] > fft_vals[1]: peak_indices.insert(0, 0)
    if fft_vals[N-1] < fft_vals[N]: peak_indices.append(N)

    closest_peak_idx = peak_indices[np.argmin(np.abs(fft_freqs[peak_indices] - f_approx))]
    logger.debug("Found peak at {:.3f} GHz (index: {})".format(fft_freqs[closest_peak_idx] / 1e9, closest_peak_idx))
    return fft_freqs[closest_peak_idx], closest_peak_idx
