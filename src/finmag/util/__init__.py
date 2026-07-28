# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

try:
    from .plot import *
except ImportError:
    # Plotting depends on optional third-party packages such as matplotlib.
    # Keep finmag importable when the core simulation stack is present.
    pass
