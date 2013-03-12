from __future__ import division
from fft import *

import numpy as np
from numpy import sin, cos, pi, exp, real, conj
import matplotlib.pyplot as plt
import logging


def test_analytical_inverse_DFT():
    """
    Construct a simple signal as a superposition of two sinusoidal
    oscillations with different frequencies. Then filter this signal
    in two ways and check that we get the same result:

    1) Set all Fourier coefficients except one to zero and do an
       inverse Fourier transform.

    2) Compute the inverse Fourier transform with a single coefficient
       "analytically" (by applying the formula at [1] to the case of
       just a single summand and writing the complex exponential as a
       sum of a single sine and cosine).

       [1] http://docs.scipy.org/doc/numpy/reference/routines.fft.html
    """
    n = 1000
    tmin = 0.23*pi
    tmax = 4.23*pi
    dt = (tmax - tmin) / (n - 1)

    # Time steps of the signal
    ts = np.linspace(tmin, tmax, n)

    # Define a simple signal that is a superposition of two waves
    signal = sin(ts) + 2*cos(3*ts)

    # Plot the signal and its sin/cos components
    plt.figure()
    plt.plot(signal, 'x-', label='signal')
    plt.plot(sin(ts), label='sin(t)')
    plt.plot(cos(3*ts), label='cos(3t)')
    plt.legend()
    plt.savefig('fft_test_01_signal.pdf')

    # Perform a (real-valued) Fourier transform. Also store the
    # frequencies corresponding to the Fourier coefficients.
    rfft_vals = np.fft.rfft(signal)
    rfft_freqs = np.arange(n // 2 + 1) / (dt*n)

    # Determine indices of the two peaks
    idx_peaks = sorted(abs(rfft_vals).argsort()[-2:])
    assert(idx_peaks == [2, 6])  # sanity check that the peaks are as expected

    # For each peak coefficient, filter the signal both using the
    # inverse DFT and manually/analytically.
    for k in idx_peaks:
        # Filter the signal using the inverse DFT
        rfft_vals_filtered = np.zeros_like(rfft_vals)
        rfft_vals_filtered[k] = rfft_vals[k]
        signal_filtered = np.fft.irfft(rfft_vals_filtered)

        # Manually construct a filtered signal in various ways
        A_k = rfft_vals[k]  # Fourier coefficient at the peak
        B_k = A_k.real
        C_k = A_k.imag
        print "Fourier coefficient at index k={} is: {}".format(k, A_k)

        tt = 2 * pi * k * np.arange(n) / n
        signal_analytical_1 = np.squeeze(filter_frequency_component(signal, k, tmin, tmax))
        signal_analytical_2 = 2.0/n * (B_k * cos(tt) - C_k * sin(tt))
        signal_analytical_3 = real(1.0/n * (A_k * exp(1j*tt) + conj(A_k) * exp(-1j*tt)))

        base_oscillation = sin(ts) if (k == 2) else 2*cos(3*ts)

        print "Maximum deviation of filtered signal from the base sinusoidal oscillation: {}".format(max(abs(base_oscillation - signal_filtered)))
        assert np.allclose(base_oscillation, signal_filtered, atol=0.05, rtol=0)
        assert np.allclose(signal_filtered, signal_analytical_1, atol=1e-11, rtol=0)
        assert np.allclose(signal_filtered, signal_analytical_2, atol=1e-11, rtol=0)
        assert np.allclose(signal_filtered, signal_analytical_3, atol=1e-11, rtol=0)

        plt.figure()
        plt.plot(ts, base_oscillation, '-', label='sin(t)')
        plt.plot(ts, signal_filtered, 'x', label='filtered (iDFT)')
        plt.plot(ts, signal_analytical_1, '-', label='filtered (analytical #1)')
        plt.plot(ts, signal_analytical_2, '.', label='filtered (analytical #1)')
        plt.legend()
        plt.savefig('fft_test_02_filtered_signal_for_k_{}.pdf'.format(k))
