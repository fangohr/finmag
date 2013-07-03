# FinMag 
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
from init import *
from __version__ import __version__

# dolfin 1.1.0 re-orders the degrees of freedom
# to speed up some iterative algorithms. However,
# this means that the ordering doesn't correspond
# to geometrical structures (such as vertices)
# anymore. For now, we disable the re-ordering.
# In the future, we should enable it as it will
# help with parallel execution.
import dolfin as df
df.parameters.reorder_dofs_serial = False
