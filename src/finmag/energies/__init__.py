"""Lazy public exports for Finmag interaction classes."""

from importlib import import_module


# Keep the legacy public interaction surface without importing every backend
# and optional native module when the package is first discovered. [Codex GPT-5.6]
_LAZY_EXPORTS = {
    "Demag": ("finmag.energies.demag", "Demag"),
    "Demag2D": ("finmag.energies.demag", "Demag2D"),
    "MacroGeometry": ("finmag.energies.demag", "MacroGeometry"),
    "EnergyBase": ("finmag.energies.energy_base", "EnergyBase"),
    "Exchange": ("finmag.energies.exchange", "Exchange"),
    "UniaxialAnisotropy": ("finmag.energies.anisotropy", "UniaxialAnisotropy"),
    "CubicAnisotropy": ("finmag.energies.cubic_anisotropy", "CubicAnisotropy"),
    "Zeeman": ("finmag.energies.zeeman", "Zeeman"),
    "TimeZeeman": ("finmag.energies.zeeman", "TimeZeeman"),
    "DiscreteTimeZeeman": ("finmag.energies.zeeman", "DiscreteTimeZeeman"),
    "OscillatingZeeman": ("finmag.energies.zeeman", "OscillatingZeeman"),
    "TimeZeemanPython": ("finmag.energies.zeeman", "TimeZeemanPython"),
    "DMI": ("finmag.energies.dmi", "DMI"),
    "DMI_interfacial": ("finmag.energies.dmi", "DMI_interfacial"),
    "ThinFilmDemag": ("finmag.energies.thin_film_demag", "ThinFilmDemag"),
    "FixedEnergyDW": ("finmag.energies.dw_fixed_energy", "FixedEnergyDW"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    from finmag import _prepare_legacy_dolfin

    _prepare_legacy_dolfin()
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
