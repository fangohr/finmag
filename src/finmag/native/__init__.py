# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

"""
Provides the symbols from the finmag native extension module.

Importing this module will first compile the finmag extension module, then
load all of its symbols into this module's namespace.
"""
from ..util.native_compiler import make_modules as _make_modules
_make_modules()
