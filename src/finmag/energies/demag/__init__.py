import logging
from fk_demag import FKDemag
from fk_demag_2d import Demag2D
from solver_gcr import FemBemGCRSolver
from treecode_bem import TreecodeBEM

log = logging.getLogger("finmag")
KNOWN_SOLVERS = {'FK': FKDemag, 'GCR': FemBemGCRSolver, 'Treecode': TreecodeBEM}


def Demag(solver='FK', *args, **kwargs):
    if not solver in KNOWN_SOLVERS:
        log.error("Tried to create a Demag object with unknown solver '{}'".format(solver))
        raise NotImplementedError(
            "Solver '{}' not implemented. Valid choices: one of '{}'.".format(solver, KNOWN_SOLVERS.keys()))

    log.debug("Creating Demag object with solver '{}'.".format(solver))
    return KNOWN_SOLVERS[solver](*args, **kwargs)
