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
import importlib.util
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
    # SR1 P2.1: ``set_logging_level`` was lifted out of the (still unported)
    # ``finmag.util.helpers`` into a stdlib-only module, so it now resolves
    # without legacy dolfin. The flag nevertheless stays ``True`` (like
    # ``example`` below, and unlike the ``_LEGACY_ONLY_FEATURES`` names): when
    # legacy dolfin IS installed, reaching this public name must keep applying
    # the historical dof-ordering preparation, because legacy scripts of the
    # form ``import finmag; finmag.set_logging_level(...); df.FunctionSpace(...)``
    # relied on that side effect. Dropping it here would silently change dof
    # ordering in the legacy lane. [Claude Opus 4.8]
    "set_logging_level": (
        "finmag.util.logging_helpers",
        "set_logging_level",
        True,
    ),
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


# SR1 P2.1: public names that still *require* legacy dolfin and have no
# DOLFINx port yet. Without legacy dolfin installed the lazy boundary raises a
# curated ``NotImplementedError`` naming the feature, instead of letting a raw
# ``ModuleNotFoundError: No module named 'dolfin'`` escape from somewhere deep
# inside the import chain. Names flagged ``requires_legacy_dolfin=True`` that
# are NOT listed here (currently ``example``) still get the legacy dof-ordering
# preparation when dolfin is present, but resolve normally when it is absent.
# [Claude Opus 4.8]
_LEGACY_ONLY_FEATURES = {
    "NormalModeSimulation": (
        "finmag.NormalModeSimulation is not ported to DOLFINx yet: the "
        "normal-mode eigenvalue machinery (finmag.sim.normal_mode_sim) is a "
        "deferred legacy-dolfin surface."
    ),
    "normal_mode_simulation": (
        "finmag.normal_mode_simulation is not ported to DOLFINx yet: the "
        "normal-mode eigenvalue machinery (finmag.sim.normal_mode_sim) is a "
        "deferred legacy-dolfin surface."
    ),
}


def _legacy_dolfin_is_available():
    return importlib.util.find_spec("dolfin") is not None


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
        if _legacy_dolfin_is_available():
            _prepare_legacy_dolfin()
        elif name in _LEGACY_ONLY_FEATURES:
            raise NotImplementedError(_LEGACY_ONLY_FEATURES[name])
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
