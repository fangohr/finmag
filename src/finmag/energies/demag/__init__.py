"""Demag public surface for the DOLFINx port.

The Fredkin-Koehler solver (:class:`~finmag.energies.demag.fk_demag.FKDemag`)
is ported directly to DOLFINx (Task 11b). The remaining demag variants
(``Treecode``/``GCR`` solvers, 2D demag, and periodic ``MacroGeometry``) are
not ported and raise ``NotImplementedError`` by name when requested; importing
this package no longer pulls legacy ``dolfin``. [Claude Opus 4.8]
"""

import logging

from .fk_demag import FKDemag

log = logging.getLogger("finmag")

KNOWN_SOLVERS = {"FK": FKDemag}

_DEFERRED_SOLVERS = {
    "Treecode": "the treecode/PBC BEM native slice",
    "GCR": "the Garcia-Cervera-Roma solver",
}


def Demag(solver="FK", *args, **kwargs):
    """Create a demag interaction. Only the Fredkin-Koehler solver is ported."""
    if solver in KNOWN_SOLVERS:
        log.debug("Creating Demag object with solver '{}'.".format(solver))
        return KNOWN_SOLVERS[solver](*args, **kwargs)
    if solver in _DEFERRED_SOLVERS:
        raise NotImplementedError(
            "Demag solver '{}' is deferred in the DOLFINx port ({}); "
            "only the 'FK' Fredkin-Koehler solver is available.".format(
                solver, _DEFERRED_SOLVERS[solver]))
    raise NotImplementedError(
        "Solver '{}' not implemented. Valid choice: 'FK'.".format(solver))


def Demag2D(*args, **kwargs):
    """2D demag is deferred in the DOLFINx port."""
    raise NotImplementedError(
        "Demag2D (2D thin-film demag) is not ported to DOLFINx; use the "
        "'FK' solver via Demag() for 3D geometries.")


def MacroGeometry(*args, **kwargs):
    """Periodic macro-geometry demag is deferred in the DOLFINx port."""
    raise NotImplementedError(
        "MacroGeometry (periodic-boundary demag) is deferred to the separate "
        "PBC/treecode native slice.")
