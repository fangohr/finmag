# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import logging
from finmag.sim.sim import Simulation, sim_with
from __version__ import __version__
import example

logger = logging.getLogger("finmag")
logger.debug("This is Finmag version {}".format(__version__))
