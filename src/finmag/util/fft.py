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

    psd_mx = np.absolute(np.fft.rfft(mx_resampled, axis=0))**2
    psd_my = np.absolute(np.fft.rfft(my_resampled, axis=0))**2
    psd_mz = np.absolute(np.fft.rfft(mz_resampled, axis=0))**2
    n = len(psd_mx)

    if filename.endswith('.npy'):
        # Compute the power spectra and then do the spatial average
        psd_mx = psd_mx.sum(axis=-1)
        psd_my = psd_my.sum(axis=-1)
        psd_mz = psd_mz.sum(axis=-1)

    # When using np.fft.fftfreq, the last frequency sometimes becomes
    # negative; to avoid this we compute the frequencies by hand.
    freqs = np.arange(n) / (t_step*len(ts_resampled))

    return freqs, psd_mx, psd_my, psd_mz
