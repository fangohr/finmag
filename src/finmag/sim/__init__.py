# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

import importlib.util

if importlib.util.find_spec("dolfin") is not None:
    # The legacy ``.init`` logging bootstrap imports legacy ``dolfin``/``ufl``/
    # ``ffc``/``finmag.util.*``, which are absent in the DOLFINx environment.
    # The direct DOLFINx ``Simulation`` port (``finmag.sim.sim``) must remain
    # importable there, so the bootstrap is skipped by checking for ``dolfin``
    # itself, rather than swallowing any ``ImportError`` the bootstrap chain
    # might raise -- a bare ``except ImportError: pass`` would also hide a
    # genuinely broken import in the legacy FEniCS environment (``dolfin``
    # present, something else in the chain broken). In the legacy FEniCS
    # environment ``dolfin`` is present and the bootstrap runs exactly as
    # before, with any real import failure propagating. [Claude Sonnet 5]
    from .init import *
