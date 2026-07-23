# FinMag
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

"""Stdlib-only logging helpers for the public Finmag boundary.

``set_logging_level`` is part of ``finmag``'s public API but its historical
home, :mod:`finmag.util.helpers`, imports legacy ``dolfin`` at module scope.
SR1 P2.1 lifts this one function into a module with no FEM dependency at all
so ``finmag.set_logging_level`` resolves in the DOLFINx environment;
:mod:`finmag.util.helpers` re-exports it so the legacy spelling
``from finmag.util.helpers import set_logging_level`` keeps working.
[Claude Opus 4.8]
"""

import logging


logger = logging.getLogger("finmag")

SUPPORTED_LOGGING_LEVELS = [
    'EXTREMEDEBUG', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


def set_logging_level(level):
    """
    Set the level for finmag log messages.

    *Arguments*

    level: string

       One of the levels supported by Python's `logging` module.
       Supported values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' and
       the finmag specific level 'EXTREMEDEBUG'.
    """
    if level not in SUPPORTED_LOGGING_LEVELS:
        raise ValueError("Logging level must be one of: 'DEBUG', 'INFO', "
                         "'WARNING', 'ERROR', 'CRITICAL'")
    logger.setLevel(level)
