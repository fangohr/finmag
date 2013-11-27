from __future__ import division
from fft import *
import numpy as np
from numpy import sqrt, sin, cos, pi, exp, real, conj
import matplotlib.pyplot as plt
import subprocess as sp
import pytest
import os

from finmag.util.consts import gamma


def test_analytical_inverse_DFT():
    """
    We construct a simple signal as a superposition of two sinusoidal
    oscillations with different frequencies. Then filter this signal
    in the following two ways and check that we get the same result.

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


def test_wrong_file_suffix_for_power_spectral_density():
    with pytest.raises(ValueError):
        power_spectral_density('datafile.foo')


def test_power_spectral_density_from_averaged_magnetisation(tmpdir):
    """
    Write a time series of artificial magnetisation data (created from
    a damped harmonic oscillator) to a .ndt file and use it to compute
    the power spectral densities. Then compare them with the manually
    computed ones.

    """
    os.chdir(str(tmpdir))
    RTOL = 1e-10

    ##
    ## Step 1: Construct a time series of artificial magnetisation data and
    ## save it to a .ndt file.
    ##
    H = 1e6  # external field in A/m
    omega = gamma * H  # precession frequency
    alpha = 0.5  # some sort of damping constant
    print "Precessional frequency: {} GHz".format(omega / 1e9)

    t_step = 1e-11
    t_ini = 0
    t_end = 10e-9

    ts = np.arange(t_ini, t_end, t_step)
    print len(ts)

    # Use damped harmonic oscillator to create fake magnetisation dynamics
    mx = exp(-ts * 1e8 / alpha) * sin(omega * ts)
    my = exp(-ts * 1e8 / alpha) * cos(omega * ts)
    mz = 1 - sqrt(mx**2 + my**2)
    data = np.array([ts, mx, my, mz]).T

    # Plot the dynamics for debugging purposes
    fig = plt.figure(figsize=(20, 5))
    ax = fig.gca()
    ax.plot(ts, mx)
    ax.plot(ts, my)
    ax.plot(ts, mz)
    fig.savefig('m_vs_t.png')

    # Save the data to a .ndt file. The sed commands add the two header lines
    # which are required by the file format.
    ndt_filename = 'fake_relaxation.ndt'
    np.savetxt(ndt_filename, data)
    sp.check_call("sed -i '1 i # time  m_x  m_y  m_z' ./fake_relaxation.ndt", shell=True)
    sp.check_call("sed -i '2 i # <s>   <>   <>   <>' ./fake_relaxation.ndt", shell=True)

    ##
    ## Step 2: compute the PSDs of a resampled time series, both by
    ## hand and using power_spectral_density() and check that the
    ## results are the same.
    ##
    t_step_res = 2e-11
    t_ini_res = 1e-10
    t_end_res = 9.9e-9
    ts_resampled = np.arange(t_ini_res, t_end_res, t_step_res)
    N = len(ts_resampled) // 2 + 1  # expected length of real-valued FFT

    # Compute time series based on resampled timesteps
    mx_res = exp(-ts_resampled * 1e8 / alpha) * sin(omega * ts_resampled)
    my_res = exp(-ts_resampled * 1e8 / alpha) * cos(omega * ts_resampled)
    mz_res = 1 - sqrt(mx_res**2 + my_res**2)

    # Compute 'analytical' power spectral densities of resampled time series
    psd_mx_res_expected = abs(np.fft.rfft(mx_res))**2
    psd_my_res_expected = abs(np.fft.rfft(my_res))**2
    psd_mz_res_expected = abs(np.fft.rfft(mz_res))**2

    # Compute Fourier transform of resampled time series using FFT_m
    freqs_res, psd_mx_res, psd_my_res, psd_mz_res = \
        power_spectral_density(ndt_filename, t_step_res, t_ini=t_ini_res, t_end=t_end_res, subtract_values=None)

    # Compare both results
    assert(np.allclose(psd_mx_res, psd_mx_res_expected, atol=0, rtol=RTOL))
    assert(np.allclose(psd_my_res, psd_my_res_expected, atol=0, rtol=RTOL))
    assert(np.allclose(psd_mz_res, psd_mz_res_expected, atol=0, rtol=RTOL))

    # Also check that the frequency range is as expected
    freqs_np = np.fft.fftfreq(len(ts_resampled), d=t_step_res)[:N]
    assert(np.allclose(freqs_res, freqs_np, atol=0, rtol=RTOL))


def test_power_spectral_density_from_spatially_resolved_magnetisation(tmpdir):
    """
    First we write some 'forged' spatially resolved magnetisation
    dynamics to a bunch of .npy files (representing the time series).
    The oscillation is exactly the same at every location, so that we
    don't lose any information in the averaging process and can
    compare with the analytical solution as in the previous test.

    """
    os.chdir(str(tmpdir))
    RTOL = 1e-10

    ##
    ## Step 1: Construct a time series of artificial magnetisation
    ## data and save it to a bunch of .npy files.
    ##
    H = 1e6  # external field in A/m
    omega = gamma * H  # precession frequency
    alpha = 0.5  # some sort of damping constant
    print "Precessional frequency: {} GHz".format(omega / 1e9)

    t_step = 1e-11
    t_ini = 0
    t_end = 10e-9

    ts = np.arange(t_ini, t_end, t_step)
    num_timesteps = len(ts)
    print "Number of timesteps: {}".format(num_timesteps)

    # Use damped harmonic oscillator to create fake magnetisation dynamics
    mx = exp(-ts * 1e8 / alpha) * sin(omega * ts)
    my = exp(-ts * 1e8 / alpha) * cos(omega * ts)
    mz = 1 - sqrt(mx**2 + my**2)

    # Write the data to a series of .npy files
    num_vertices = 42 # in a real application this would be the number of mesh vertices
    a = np.zeros((3, num_vertices))
    for i in xrange(num_timesteps):
        a[0, :] = mx[i]
        a[1, :] = my[i]
        a[2, :] = mz[i]
        filename = 'm_ringdown_{:06d}.npy'.format(i)
        np.save(filename, a.ravel())

    ##
    ## Step 2: compute the FFT of a resampled time series, both by
    ## hand and using FFT_m.
    ##
    ## XXX TODO: Resampling timesteps is not supported when using .npy
    ## files. Either simplify the code below, or implement saving to
    ## .h5 files so that it's easier to implement resampling for
    ## spatially resolved data, too.
    ##
    t_step_res = t_step
    t_ini_res = t_ini
    t_end_res = t_end
    ts_resampled = np.arange(t_ini_res, t_end_res, t_step_res)

    # Compute time series based on resampled timesteps
    mx_res = exp(-ts_resampled * 1e8 / alpha) * sin(omega * ts_resampled)
    my_res = exp(-ts_resampled * 1e8 / alpha) * cos(omega * ts_resampled)
    mz_res = 1 - sqrt(mx_res**2 + my_res**2)

    # Compute 'analytical' Fourier transform of resampled time series and
    # determine the power of the spectrum for each component. We also need
    # to multiply by the number of mesh nodes because the numerical algorithm
    # sums up all contributions at the individual nodes (but we can just
    # multiply because they are all identical by construction).
    psd_mx_expected = num_vertices * np.absolute(np.fft.rfft(mx_res))**2
    psd_my_expected = num_vertices * np.absolute(np.fft.rfft(my_res))**2
    psd_mz_expected = num_vertices * np.absolute(np.fft.rfft(mz_res))**2

    # Compute Fourier transform of resampled time series using FFT_m
    freqs_computed, psd_mx_computed, psd_my_computed, psd_mz_computed = \
        power_spectral_density('m_ringdown*.npy', t_step_res, t_ini=t_ini_res, t_end=t_end_res, subtract_values=None)

    # Check that the analytically determined power spectra are the same as the computed ones.
    assert(np.allclose(psd_mx_expected, psd_mx_computed, atol=0, rtol=RTOL))
    assert(np.allclose(psd_my_expected, psd_my_computed, atol=0, rtol=RTOL))
    assert(np.allclose(psd_mz_expected, psd_mz_computed, atol=0, rtol=RTOL))

    # Plot the spectra for debugging
    fig = plt.figure(figsize=(20, 5))
    ax = fig.gca()
    ax.plot(freqs_computed, psd_mx_expected, label='psd_mx_expected')
    ax.plot(freqs_computed, psd_my_expected, label='psd_my_expected')
    ax.plot(freqs_computed, psd_mz_expected, label='psd_mz_expected')
    ax.plot(freqs_computed, psd_mx_computed, label='psd_mx_computed')
    ax.plot(freqs_computed, psd_my_computed, label='psd_my_computed')
    ax.plot(freqs_computed, psd_mz_computed, label='psd_mz_computed')
    ax.legend(loc='best')
    fig.savefig('psd_m_McMichaelStiles.png')


def test_find_peak_near_frequency(tmpdir, debug=False):
    """
    Check that `find_peak_near_frequency` works as expected, including
    special cases, boundary cases etc.

    """

    fft_freqs = [0, 1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9]
    fft_mx = [1, 4, 3, 2, 1.5, 1.3, 2.5, 1.1, 1.0]
    fft_my = [1, 4, 3, 2, 1.5, 1.3, 1.1, 1.7, 1.0]
    fft_mz = [2, 1, 1, 1, 1, 1, 1, 1, 1.5]

    if debug == True:
        # Plot the arrays for debugging
        os.chdir(str(tmpdir))
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(fft_freqs, fft_mx, label='fft_mx')
        ax.plot(fft_freqs, fft_my, label='fft_my')
        ax.plot(fft_freqs, fft_mz, label='fft_mz')
        ax.legend()
        fig.savefig('fft_vals.png')

    assert find_peak_near_frequency(1.5e9, fft_freqs, fft_mx) == (1e9, 1)
    #assert find_peak_near_frequency(1.5e9, fft_freqs, [fft_mx, fft_my]) == (1, 1e9)
    assert find_peak_near_frequency(5e9, fft_freqs, fft_mx) == (6e9, 6)
    assert find_peak_near_frequency(5e9, fft_freqs, fft_my) == (7e9, 7)
    assert find_peak_near_frequency(3.7e9, fft_freqs, fft_mx) == (6e9, 6)
    #assert find_peak_near_frequency(4e9, fft_freqs, [fft_mx, fft_my]) == None  # no simultaneous peak

    # Just to check special cases, boundary cases etc.
    assert find_peak_near_frequency(1e9, fft_freqs, fft_mx) == (1e9, 1)
    assert find_peak_near_frequency(0.9e9, fft_freqs, fft_mx) == (1e9, 1)
    assert find_peak_near_frequency(1.1e9, fft_freqs, fft_mx) == (1e9, 1)
    assert find_peak_near_frequency(-0.1e9, fft_freqs, fft_mx) == (1e9, 1)
    assert find_peak_near_frequency(20e9, fft_freqs, fft_mx) == (6e9, 6)

    assert find_peak_near_frequency(-0.5e9, fft_freqs, fft_mz) == (0e9, 0)
    assert find_peak_near_frequency(0.5e9, fft_freqs, fft_mz) == (0e9, 0)
    assert find_peak_near_frequency(1e9, fft_freqs, fft_mz) == (0e9, 0)
    assert find_peak_near_frequency(6e9, fft_freqs, fft_mz) == (8e9, 8)
    assert find_peak_near_frequency(8e9, fft_freqs, fft_mz) == (8e9, 8)
    assert find_peak_near_frequency(9e9, fft_freqs, fft_mz) == (8e9, 8)

    with pytest.raises(ValueError):
        # An error should be raised if fft_vals doesn't have the same
        # length as fft_freqs.
        find_peak_near_frequency(2.5e9, fft_freqs, fft_vals=[0, 1])
