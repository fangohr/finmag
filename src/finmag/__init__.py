# FinMag
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk

"""Public Finmag package boundary.

The legacy implementation is loaded one public object at a time so importing
the package itself remains independent of a particular FEM runtime.
"""

from importlib import import_module
import logging

from .__version__ import __version__


logger = logging.getLogger("finmag")
logger.propagate = False
logging.EXTREMEDEBUG = 5
logging.addLevelName(logging.EXTREMEDEBUG, "EXTREMEDEBUG")
logger.extremedebug = lambda message: logger.log(logging.EXTREMEDEBUG, message)


# These are intentional compatibility exports, rather than every incidental
# name that the historical wildcard initializer happened to expose. [Codex GPT-5.6]
_LAZY_EXPORTS = {
    # The core Simulation/sim_with are now the direct DOLFINx port and no
    # longer need the legacy dolfin bridge. [Claude Opus 4.8]
    "Simulation": ("finmag.sim.sim", "Simulation", False),
    "sim_with": ("finmag.sim.sim", "sim_with", False),
    "Field": ("finmag.field", "Field", False),
    "MacroGeometry": ("finmag.energies.demag", "MacroGeometry", False),
    "NormalModeSimulation": (
        "finmag.sim.normal_mode_sim",
        "NormalModeSimulation",
        True,
    ),
    "normal_mode_simulation": (
        "finmag.sim.normal_mode_sim",
        "normal_mode_simulation",
        True,
    ),
    "set_logging_level": ("finmag.util.helpers", "set_logging_level", True),
    "configuration": ("finmag.util.configuration", None, False),
    "versions": ("finmag.util.versions", None, False),
    "example": ("finmag.example", None, True),
    "energies": ("finmag.energies", None, False),
}

__all__ = [
    "Simulation",
    "sim_with",
    "Field",
    "MacroGeometry",
    "NormalModeSimulation",
    "normal_mode_simulation",
    "set_logging_level",
    "configuration",
    "versions",
    "example",
    "energies",
    "timings_report",
    "__version__",
    "logger",
    "logging",
]


def _prepare_legacy_dolfin(df=None):
    """Apply the degree-of-freedom ordering expected by legacy Finmag."""
    if df is None:
        df = import_module("dolfin")

    parameters = df.parameters
    if hasattr(parameters, "reorder_dofs_serial"):
        parameters.reorder_dofs_serial = False
    elif "reorder_dofs_serial" in parameters:
        parameters["reorder_dofs_serial"] = False
    return df


def __getattr__(name):
    try:
        module_name, attribute_name, requires_legacy_dolfin = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    if requires_legacy_dolfin:
        _prepare_legacy_dolfin()
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def timings_report(n=10):
    """Return a report of the functions where Finmag spent the most time."""
    from aeon import timer

    return timer.report(n)
