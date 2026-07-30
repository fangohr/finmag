# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

try:  # master: from .init import *
    from .init import *
except ImportError:
    # not ported (D33): `.init` chains to `finmag.util.helpers`, which still
    # `import dolfin` at module scope (register C02). Guarded here (rather
    # than only in test_mesh.py) because this package `__init__.py` runs
    # before ANY submodule of `finmag.util.oommf` can be imported --
    # including dolfin-free submodules like `mesh.py` -- so an unguarded
    # failure here would make every test file living under this package
    # uncollectable, not just the ones that actually need the calculator.
    pass
