# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Hans Fangohr

# SR1 P2.1: ``bar``/``barmini``/``nanowire`` are ported to DOLFINx and import
# eagerly; ``sphere_inside_airbox`` and ``normal_modes`` still need genuinely
# unported machinery (``finmag.util.meshes.sphere_inside_box``,
# ``finmag.util.helpers.scalar_valued_dg_function``, the normal-mode solvers
# and a legacy dolfin-XML mesh) and are therefore resolved lazily, raising a
# curated ``NotImplementedError`` that names the deferred feature instead of
# leaking a raw ``ModuleNotFoundError: No module named 'dolfin'``.
#
# They are deliberately resolved at this boundary rather than stubbed out in
# their own modules, so that the deferral is confined to the dolfin-free
# environment. Note this is forward-looking insurance, not a contract exercised
# today: importing ``finmag.example`` already required ``dolfinx`` before this
# slice (via ``sphere_inside_airbox`` -> ``finmag.field``), so the legacy
# FEniCS lane cannot import this package in either state. Do not claim the
# legacy resolution path is verified until an environment provides both
# runtimes. [Claude Opus 4.8]

import importlib
import importlib.util

from .bar import bar, barmini
from .nanowire import nanowire


_LEGACY_ONLY_EXPORTS = {
    "sphere_inside_airbox": (
        "finmag.example.sphere_inside_airbox",
        "sphere_inside_airbox",
        "finmag.example.sphere_inside_airbox is not ported to DOLFINx yet: it "
        "depends on the deferred legacy-dolfin helpers "
        "finmag.util.meshes.sphere_inside_box and "
        "finmag.util.helpers.scalar_valued_dg_function.",
    ),
    "normal_modes": (
        "finmag.example.normal_modes",
        None,
        "finmag.example.normal_modes is not ported to DOLFINx yet: it depends "
        "on the deferred normal-mode machinery and a legacy dolfin-XML mesh.",
    ),
}

__all__ = ["bar", "barmini", "nanowire"] + sorted(_LEGACY_ONLY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name, message = _LEGACY_ONLY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name))

    if importlib.util.find_spec("dolfin") is None:
        raise NotImplementedError(message)

    module = importlib.import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
