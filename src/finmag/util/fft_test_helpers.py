import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
from numpy import sqrt, sin, cos, pi, exp, real, conj


def create_test_ndt_file(dirname, omega, alpha, t_step, t_ini, t_end, debug=True):
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


def create_test_npy_files(dirname, omega, alpha, t_step, t_ini, t_end, num_vertices):
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
