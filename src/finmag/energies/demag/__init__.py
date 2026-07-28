import logging
from .fk_demag import FKDemag
from .fk_demag_pbc import MacroGeometry
from .fk_demag_2d import Demag2D
try:
    from .treecode_bem import TreecodeBEM
    TREECODE_BEM_IMPORT_ERROR = None
except Exception as error:
    TreecodeBEM = None
    TREECODE_BEM_IMPORT_ERROR = error

log = logging.getLogger("finmag")
KNOWN_SOLVERS = {
    'FK': FKDemag}
if TreecodeBEM is not None:
    KNOWN_SOLVERS['Treecode'] = TreecodeBEM


def Demag(solver='FK', *args, **kwargs):
    if not solver in KNOWN_SOLVERS:
        log.error(
            "Tried to create a Demag object with unknown solver '{}'".format(solver))
        raise NotImplementedError(
            "Solver '{}' not implemented. Valid choices: one of '{}'.".format(solver, KNOWN_SOLVERS.keys()))

    log.debug("Creating Demag object with solver '{}'.".format(solver))
    return KNOWN_SOLVERS[solver](*args, **kwargs)
