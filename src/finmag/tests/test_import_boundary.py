"""Focused checks for Finmag's lazy package import boundary."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent


def _run_isolated(code):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_plain_import_loads_no_fem_or_native_modules():
    # A clean subprocess makes sys.modules an exact import-side-effect witness. [Codex GPT-5.6]
    _run_isolated(
        """
import sys
import finmag

expected = {
    "Simulation", "sim_with", "Field", "MacroGeometry",
    "NormalModeSimulation", "normal_mode_simulation", "set_logging_level",
    "configuration", "versions", "example", "energies", "timings_report",
    "__version__", "logger", "logging",
}
assert expected == set(finmag.__all__)
assert "dolfin" not in sys.modules
assert "dolfinx" not in sys.modules
assert not any(
    name == "finmag.native" or name.startswith("finmag.native.")
    for name in sys.modules
)
assert finmag.__version__
assert finmag.logging.EXTREMEDEBUG == 5
assert callable(finmag.logger.extremedebug)
"""
    )


@pytest.mark.skipif(
    importlib.util.find_spec("dolfinx") is None,
    reason="This coexistence check requires the DOLFINx environment.",
)
def test_import_after_dolfinx_needs_no_initializer_bridge():
    _run_isolated(
        """
import sys
import dolfinx

assert "finmag" not in sys.modules
import finmag

assert not hasattr(dolfinx, "parameters")
assert "dolfin" not in sys.modules
assert not any(
    name == "finmag.native" or name.startswith("finmag.native.")
    for name in sys.modules
)
"""
    )


def test_energies_package_is_lazy():
    _run_isolated(
        """
import sys
import finmag.energies as energies

expected = {
    "Demag", "Demag2D", "MacroGeometry", "EnergyBase", "Exchange",
    "UniaxialAnisotropy", "CubicAnisotropy", "Zeeman", "TimeZeeman",
    "DiscreteTimeZeeman", "OscillatingZeeman", "TimeZeemanPython", "DMI",
    "DMI_interfacial", "ThinFilmDemag", "FixedEnergyDW",
}
assert expected == set(energies.__all__)
assert "dolfin" not in sys.modules
assert "dolfinx" not in sys.modules
assert not any(
    name == "finmag.native" or name.startswith("finmag.native.")
    for name in sys.modules
)
"""
    )


@pytest.mark.skipif(
    importlib.util.find_spec("dolfinx") is None,
    reason="The ported energy export check requires the DOLFINx environment.",
)
def test_ported_energy_exports_bypass_legacy_dolfin():
    _run_isolated(
        """
import sys
from finmag.energies import (
    EnergyBase, Exchange, TimeZeeman, UniaxialAnisotropy, Zeeman,
)

assert EnergyBase.__module__ == "finmag.energies.energy_base"
assert Exchange.__module__ == "finmag.energies.exchange"
assert UniaxialAnisotropy.__module__ == "finmag.energies.anisotropy"
assert Zeeman.__module__ == "finmag.energies.zeeman"
assert TimeZeeman.__module__ == "finmag.energies.zeeman"
try:
    TimeZeeman((1.0, 0.0, 0.0))
except NotImplementedError as error:
    assert "deferred" in str(error)
else:
    raise AssertionError("TimeZeeman did not report its explicit deferral")
assert "dolfin" not in sys.modules
assert not any(
    name == "finmag.native" or name.startswith("finmag.native.")
    for name in sys.modules
)
"""
    )


def test_unported_energy_access_reports_the_real_missing_dependency():
    if importlib.util.find_spec("dolfin") is not None:
        pytest.skip("This check is for the DOLFINx environment without legacy dolfin.")

    result = subprocess.run(
        [sys.executable, "-c", "import finmag.energies as energies; energies.DMI"],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SRC_ROOT)},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No module named 'dolfin'" in result.stderr


def test_unported_public_access_reports_the_real_missing_dependency():
    if importlib.util.find_spec("dolfin") is not None:
        pytest.skip("This check is for the DOLFINx environment without legacy dolfin.")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import finmag; finmag.Simulation",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SRC_ROOT)},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No module named 'dolfin'" in result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("dolfin") is None
    or importlib.util.find_spec("dolfinx") is None,
    reason=(
        "Downstream legacy public names cross the now-DOLFINx Field boundary; "
        "they remain deferred until their direct port tasks."
    ),
)
def test_legacy_public_names_resolve_to_their_defining_modules():
    _run_isolated(
        """
import dolfin as df
df.parameters["reorder_dofs_serial"] = True

import finmag
assert df.parameters["reorder_dofs_serial"] is True

from finmag import (
    MacroGeometry, NormalModeSimulation, Simulation, configuration,
    example, normal_mode_simulation, set_logging_level, sim_with, versions,
)
import finmag.energies as energies

assert Simulation.__module__ == "finmag.sim.sim"
assert sim_with.__module__ == "finmag.sim.sim"
assert MacroGeometry.__module__ == "finmag.energies.demag.fk_demag_pbc"
assert NormalModeSimulation.__module__ == "finmag.sim.normal_mode_sim"
assert normal_mode_simulation.__module__ == "finmag.sim.normal_mode_sim"
assert set_logging_level.__module__ == "finmag.util.helpers"
assert configuration.__name__ == "finmag.util.configuration"
assert versions.__name__ == "finmag.util.versions"
assert example.__name__ == "finmag.example"
assert finmag.timings_report.__module__ == "finmag"
assert isinstance(finmag.__version__, str)
assert finmag.logger.name == "finmag"

expected_energy_modules = {
    "CubicAnisotropy": "finmag.energies.cubic_anisotropy",
    "DMI": "finmag.energies.dmi",
    "DMI_interfacial": "finmag.energies.dmi",
    "Demag": "finmag.energies.demag",
    "Demag2D": "finmag.energies.demag.fk_demag_2d",
    "DiscreteTimeZeeman": "finmag.energies.zeeman",
    "EnergyBase": "finmag.energies.energy_base",
    "Exchange": "finmag.energies.exchange",
    "FixedEnergyDW": "finmag.energies.dw_fixed_energy",
    "MacroGeometry": "finmag.energies.demag.fk_demag_pbc",
    "OscillatingZeeman": "finmag.energies.zeeman",
    "ThinFilmDemag": "finmag.energies.thin_film_demag",
    "TimeZeeman": "finmag.energies.zeeman",
    "TimeZeemanPython": "finmag.energies.zeeman",
    "UniaxialAnisotropy": "finmag.energies.anisotropy",
    "Zeeman": "finmag.energies.zeeman",
}
assert {
    name: getattr(energies, name).__module__
    for name in energies.__all__
} == expected_energy_modules
assert df.parameters["reorder_dofs_serial"] is False
"""
    )


@pytest.mark.skipif(
    importlib.util.find_spec("dolfin") is None
    or importlib.util.find_spec("dolfinx") is not None,
    reason="This check requires the legacy-only FEniCS environment.",
)
def test_direct_field_import_is_dolfinx_only_and_has_no_legacy_bridge():
    _run_isolated(
        """
import dolfin as df
df.parameters["reorder_dofs_serial"] = True

try:
    from finmag.field import Field
except ModuleNotFoundError as error:
    assert error.name == "dolfinx"
else:
    raise AssertionError("DOLFINx-only Field unexpectedly loaded without dolfinx")

assert df.parameters["reorder_dofs_serial"] is True
"""
    )
