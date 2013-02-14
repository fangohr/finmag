from __future__ import division
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt

def FFT_m(filename, t_step=1e-11, t_ini=0):
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
    """
    # Load the data; extract time steps and magnetisation
    data = np.loadtxt(filename)
    ts, mx, my, mz = data.transpose()

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


def plot_FFT_m(filename, t_step=1e-11, t_ini=0.0, components="xyz"):
    """
    Plot the frequency spectrum of the components of the magnetisation m.

    The arguemtns `t_ini` and `t_step` have the same meaning as in the
    FFT_m function.

    `components` can be a string or a list containing the components
    to plot. Default: 'xyz'.

    Returns the matplotlib Axis instance containing the plot.
    """
    if not set(components).issubset("xyz"):
        raise ValueError("Components must only contain 'x', 'y' and 'z'. "
                         "Got: {}".format(components))

    fft_freq, fft_mx, fft_my, fft_mz = FFT_m(filename, t_step, t_ini)
    fft_freq_GHz = fft_freq / 1e9
    fig = plt.figure()
    ax = fig.gca()
    if 'x' in components: ax.plot(fft_freq_GHz, fft_mx, '.-', label=r'FFT of $m_x$')
    if 'y' in components: ax.plot(fft_freq_GHz, fft_my, '.-', label=r'FFT of $m_y$')
    if 'z' in components: ax.plot(fft_freq_GHz, fft_mz, '.-', label=r'FFT of $m_z$')
    ax.set_xlabel('fGHz')
    ax.set_ylabel('Amplitude')
    ax.set_xticks(np.arange(int(fft_freq[-1] / 1e9)))
    plt.legend()
    ax.grid()

    return ax
