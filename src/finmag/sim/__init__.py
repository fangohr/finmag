# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

try:
    from .init import *
except ImportError:
    # The legacy ``.init`` logging bootstrap imports legacy ``dolfin``/``ffc``,
    # which are absent in the DOLFINx environment. The direct DOLFINx
    # ``Simulation`` port (``finmag.sim.sim``) must remain importable there, so
    # the legacy console/file logging setup degrades to a no-op when the legacy
    # FEM stack is unavailable. In the legacy FEniCS environment ``dolfin`` is
    # present and the bootstrap runs exactly as before. [Claude Opus 4.8]
    pass
