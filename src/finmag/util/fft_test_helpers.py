import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
from numpy import sqrt, sin, cos, pi, exp, real, conj


def create_test_ndt_file(dirname, t_step, t_ini, t_end, omega, alpha, debug=True):
    """
    Create a .ndt file with a time series of artificial magnetisation
    data for testing purposes.

    """
    print "Precessional frequency: {} GHz".format(omega / 1e9)

    ts = np.arange(t_ini, t_end, t_step)
    print len(ts)

    # Use damped harmonic oscillator to create fake magnetisation dynamics
    mx = exp(-ts * 1e8 / alpha) * sin(omega * ts)
    my = exp(-ts * 1e8 / alpha) * cos(omega * ts)
    mz = 1 - sqrt(mx**2 + my**2)
    data = np.array([ts, mx, my, mz]).T

    if debug:
        # Plot the dynamics for debugging purposes
        fig = plt.figure(figsize=(20, 5))
        ax = fig.gca()
        ax.plot(ts, mx)
        ax.plot(ts, my)
        ax.plot(ts, mz)
        fig.savefig(os.path.join(dirname, 'm_vs_t.png'))

    # Save the data to a .ndt file. The sed commands add the two header lines
    # which are required by the file format.
    ndt_filename = os.path.join(dirname, 'fake_relaxation.ndt')
    np.savetxt(ndt_filename, data)
    sp.check_call("sed -i '1 i # time  m_x  m_y  m_z' ./fake_relaxation.ndt", shell=True)
    sp.check_call("sed -i '2 i # <s>   <>   <>   <>' ./fake_relaxation.ndt", shell=True)

    return ndt_filename


def create_test_npy_files(dirname, t_step, t_ini, t_end, omega, alpha, num_vertices):
    """
    Construct a time series of artificial magnetisation data and save
    it to a bunch of .npy files.
    """
    print "Precessional frequency: {} GHz".format(omega / 1e9)

    ts = np.arange(t_ini, t_end, t_step)
    num_timesteps = len(ts)
    print "Number of timesteps: {}".format(num_timesteps)

    # Use damped harmonic oscillator to create fake magnetisation dynamics
    mx = exp(-ts * 1e8 / alpha) * sin(omega * ts)
    my = exp(-ts * 1e8 / alpha) * cos(omega * ts)
    mz = 1 - sqrt(mx**2 + my**2)

    # Write the data to a series of .npy files
    a = np.zeros((3, num_vertices))
    for i in xrange(num_timesteps):
        a[0, :] = mx[i]
        a[1, :] = my[i]
        a[2, :] = mz[i]
        filename = os.path.join(dirname, 'm_ringdown_{:06d}.npy'.format(i))
        np.save(filename, a.ravel())

    npy_files = os.path.join(dirname, 'm_ringdown_*.npy')
    return npy_files


def create_test_npy_files_with_two_regions(dirname, t_step, t_ini, t_end, omega1, alpha1, num_vertices1, omega2, alpha2, num_vertices2):
    """
    Construct a time series of artificial magnetisation data and save
    it to a bunch of .npy files.
    """
    print "Precessional frequency in region 1: {} GHz".format(omega1 / 1e9)
    print "Precessional frequency in region 2: {} GHz".format(omega2 / 1e9)

    ts = np.arange(t_ini, t_end, t_step)
    num_timesteps = len(ts)
    print "Number of timesteps: {}".format(num_timesteps)

    # Use damped harmonic oscillator to create fake magnetisation dynamics
    mx1 = exp(-ts * 1e8 / alpha1) * sin(omega1 * ts)
    my1 = exp(-ts * 1e8 / alpha1) * cos(omega1 * ts)
    mz1 = 1 - sqrt(mx1**2 + my1**2)

    mx2 = exp(-ts * 1e8 / alpha2) * sin(omega2 * ts)
    my2 = exp(-ts * 1e8 / alpha2) * cos(omega2 * ts)
    mz2 = 1 - sqrt(mx2**2 + my2**2)

    # Write the data to a series of .npy files
    N = num_vertices1 + num_vertices2
    a = np.zeros((3, N))
    for i in xrange(num_timesteps):
        # Write values in region 1
        a[0, :num_vertices1] = mx1[i]
        a[1, :num_vertices1] = my1[i]
        a[2, :num_vertices1] = mz1[i]

        # Write values in region 2
        a[0, num_vertices1:] = mx2[i]
        a[1, num_vertices1:] = my2[i]
        a[2, num_vertices1:] = mz2[i]

        filename = os.path.join(dirname, 'm_ringdown_{:06d}.npy'.format(i))
        np.save(filename, a.ravel())

    npy_files = os.path.join(dirname, 'm_ringdown_*.npy')
    return npy_files
