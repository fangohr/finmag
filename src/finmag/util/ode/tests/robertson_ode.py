# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import numpy as np
import scipy.integrate

n_rhs_evals = 0
n_jac_evals = 0

def robertson_reset_n_evals():
    global n_rhs_evals, n_jac_evals
    n_rhs_evals = 0
    n_jac_evals = 0

def robertson_rhs(t, y):
    global n_rhs_evals
    n_rhs_evals += 1
    return np.array([
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
            3e7 * y[1] * y[1]
    ])


def robertson_jacobean(t, y):
    global n_jac_evals
    n_jac_evals += 1
    jac = np.zeros((3, 3))
    # jac[i,j] = d f[i] / d y[j]
    jac[0, 0] = -0.04
    jac[0, 1] = 1e4 * y[2]
    jac[0, 2] = 1e4 * y[1]
    jac[1, 0] = 0.04
    jac[1, 1] = -1e4 * y[2] - 6e7 * y[1]
    jac[1, 2] = -1e4 * y[1]
    jac[2, 0] = 0
    jac[2, 1] = 6e7 * y[1]
    jac[2, 2] = 0
    return jac

ROBERTSON_Y0 = np.array([1., 0., 0])

if __name__=="__main__":
    import matplotlib.pyplot as plt

    ts = np.logspace(-8, 8, base=10, num=200)
    ys = np.zeros((3, ts.size))
    for i, t in enumerate(ts):
        integrator = scipy.integrate.ode(robertson_rhs, jac=robertson_jacobean)
        integrator.set_initial_value(ROBERTSON_Y0)
        integrator.set_integrator("vode", method="bdf", nsteps=10000)
        ys[:, i] = integrator.integrate(t)

    ys[1] *= 1e4

    vals = []
    for i in xrange(3):
        vals.append(ts)
        vals.append(ys[i])
        vals.append('')

    plt.semilogx(*vals)
    plt.legend(["$y_1$", "$10^4 y_2$", "$y_3$"])
    plt.show()

