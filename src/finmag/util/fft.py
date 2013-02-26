from __future__ import division
from scipy.interpolate import InterpolatedUnivariateSpline
from finmag.util.helpers import probe
import numpy as np
import matplotlib.pyplot as plt

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
               components="xyz", figsize=None):
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
