"""Demag public surface for the DOLFINx port.

The Fredkin-Koehler solver (:class:`~finmag.energies.demag.fk_demag.FKDemag`)
is ported directly to DOLFINx (Task 11b). Task 23 ports the treecode-accelerated
FK solver (``Demag(solver='Treecode')``, :class:`~finmag.energies.demag.treecode_bem.TreecodeBEM`)
and the periodic macro-geometry demag (:class:`~finmag.energies.demag.fk_demag_pbc.MacroGeometry`,
via ``Demag(macrogeometry=...)``) on top of the native ``treecode_bem`` Cython
BEM kernels. The remaining variants (``GCR`` solver, 2D ``Demag2D``) are not
ported and raise ``NotImplementedError`` by name. Importing this package does
not pull legacy ``dolfin`` (the treecode kernels are imported lazily on setup).
[Claude Opus 4.8]
"""

import logging

from .fk_demag import FKDemag
from .fk_demag_pbc import MacroGeometry
from .treecode_bem import TreecodeBEM

log = logging.getLogger("finmag")

KNOWN_SOLVERS = {"FK": FKDemag, "Treecode": TreecodeBEM}

_DEFERRED_SOLVERS = {
    "GCR": "the Garcia-Cervera-Roma solver",
}


def Demag(solver="FK", *args, **kwargs):
    """Create a demag interaction.

    ``solver='FK'`` (default) uses the ported Fredkin-Koehler solver;
    ``solver='Treecode'`` uses the treecode-accelerated FK solver. Pass
    ``macrogeometry=MacroGeometry(...)`` for periodic (macro-geometry) demag.
    """
    if solver in KNOWN_SOLVERS:
        log.debug("Creating Demag object with solver '{}'.".format(solver))
        return KNOWN_SOLVERS[solver](*args, **kwargs)
    if solver in _DEFERRED_SOLVERS:
        raise NotImplementedError(
            "Demag solver '{}' is deferred in the DOLFINx port ({}); "
            "available solvers: {}.".format(
                solver, _DEFERRED_SOLVERS[solver],
                sorted(KNOWN_SOLVERS.keys())))
    raise NotImplementedError(
        "Solver '{}' not implemented. Valid choices: {}.".format(
            solver, sorted(KNOWN_SOLVERS.keys())))


def Demag2D(*args, **kwargs):
    """2D thin-film demag is deferred in the DOLFINx port.

    ``Demag2D`` builds an auxiliary extruded 3D mesh (legacy ``MeshEditor``) and
    interpolates via legacy ``Expression`` objects; neither has a DOLFINx
    equivalent on the current foundations, and it does not use the treecode
    kernels. Deferred by name (Task 29 review item). [Claude Opus 4.8]
    """
    raise NotImplementedError(
        "Demag2D (2D thin-film demag) is not ported to DOLFINx; use the "
        "'FK' or 'Treecode' solver via Demag() for 3D geometries.")
