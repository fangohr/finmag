# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import finmag.native.sundials as native_ode
import numpy as np

class cvode(object):
    def __init__(self, f):
        native_ode.cvode(native_ode.CV_ADAMS, native_ode.CV_FUNCTIONAL)

def scipy_to_cvode_rhs(f):
    def cvode_rhs(t, y, ydot):
        ydot[:] = f(t, y)
        return 0
    return cvode_rhs

def scipy_to_cvode_jtimes(jac):
    def cvode_jtimes(v, Jv, t, y, fy, tmp):
        Jv[:] = np.dot(jac(t, y), v)
        return 0

    return cvode_jtimes
