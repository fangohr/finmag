"""Lazy public exports for Finmag interaction classes."""

from importlib import import_module


# Keep the legacy public interaction surface without importing every backend
# and optional native module when the package is first discovered. [Codex GPT-5.6]
_LAZY_EXPORTS = {
    "Demag": ("finmag.energies.demag", "Demag", True),
    "Demag2D": ("finmag.energies.demag", "Demag2D", True),
    "MacroGeometry": ("finmag.energies.demag", "MacroGeometry", True),
    "EnergyBase": ("finmag.energies.energy_base", "EnergyBase", False),
    "Exchange": ("finmag.energies.exchange", "Exchange", False),
    "UniaxialAnisotropy": (
        "finmag.energies.anisotropy",
        "UniaxialAnisotropy",
        False,
    ),
    "CubicAnisotropy": (
        "finmag.energies.cubic_anisotropy",
        "CubicAnisotropy",
        True,
    ),
    "Zeeman": ("finmag.energies.zeeman", "Zeeman", False),
    "TimeZeeman": ("finmag.energies.zeeman", "TimeZeeman", False),
    "DiscreteTimeZeeman": (
        "finmag.energies.zeeman",
        "DiscreteTimeZeeman",
        False,
    ),
    "OscillatingZeeman": (
        "finmag.energies.zeeman",
        "OscillatingZeeman",
        False,
    ),
    "TimeZeemanPython": (
        "finmag.energies.zeeman",
        "TimeZeemanPython",
        False,
    ),
    "DMI": ("finmag.energies.dmi", "DMI", True),
    "DMI_interfacial": ("finmag.energies.dmi", "DMI_interfacial", True),
    "ThinFilmDemag": (
        "finmag.energies.thin_film_demag",
        "ThinFilmDemag",
        True,
    ),
    "FixedEnergyDW": ("finmag.energies.dw_fixed_energy", "FixedEnergyDW", True),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name, requires_legacy_dolfin = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    if requires_legacy_dolfin:
        from finmag import _prepare_legacy_dolfin

        _prepare_legacy_dolfin()
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
