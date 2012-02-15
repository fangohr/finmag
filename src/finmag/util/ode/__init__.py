# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import finmag.native.ode as native_ode

class cvode(object):
    def __init__(self, f):
        native_ode.sundials_cvode(native_ode.CV_ADAMS, native_ode.CV_FUNCTIONAL)
