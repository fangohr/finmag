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


def test_sim_init_guard_only_skips_bootstrap_when_dolfin_absent(tmp_path):
    """``finmag.sim``'s legacy ``.init`` bootstrap is skipped by checking
    ``importlib.util.find_spec("dolfin") is None``, not a bare
    ``except ImportError: pass``. A genuine breakage inside the bootstrap
    chain while ``dolfin`` itself is present (i.e. its spec is found) must
    propagate rather than being silently swallowed. [Claude Sonnet 5]"""
    fake_root = tmp_path / "fake_dolfin_present"
    fake_root.mkdir()
    (fake_root / "dolfin.py").write_text(
        "raise ImportError('unrelated breakage inside the dolfin package')\n"
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(fake_root) + os.pathsep + str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", "import finmag.sim"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unrelated breakage inside the dolfin package" in result.stderr


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
    "DiscreteTimeZeeman", "OscillatingZeeman", "TimeZeemanPython",
    "DipolarField", "DMI",
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
    DMI, CubicAnisotropy, EnergyBase, Exchange, FixedEnergyDW, TimeZeeman,
    ThinFilmDemag, UniaxialAnisotropy, Zeeman,
)

assert EnergyBase.__module__ == "finmag.energies.energy_base"
assert Exchange.__module__ == "finmag.energies.exchange"
assert UniaxialAnisotropy.__module__ == "finmag.energies.anisotropy"
assert Zeeman.__module__ == "finmag.energies.zeeman"
assert TimeZeeman.__module__ == "finmag.energies.zeeman"
assert DMI.__module__ == "finmag.energies.dmi"
assert CubicAnisotropy.__module__ == "finmag.energies.cubic_anisotropy"
assert ThinFilmDemag.__module__ == "finmag.energies.thin_film_demag"
assert FixedEnergyDW.__module__ == "finmag.energies.dw_fixed_energy"
# Task 15: TimeZeeman is now ported (no longer a by-name deferral). A
# constant-array field_expression with no t_off raises ValueError (there
# would be no time update at all), matching the ported input-contract
# safety check transcribed from legacy.
try:
    TimeZeeman((1.0, 0.0, 0.0))
except ValueError as error:
    assert "t_off" in str(error)
else:
    raise AssertionError(
        "TimeZeeman((1.0, 0.0, 0.0)) without t_off should raise ValueError")
# Task 19: ThinFilmDemag is now ported directly; FixedEnergyDW is a curated
# by-name NotImplementedError deferral (not a raw ModuleNotFoundError).
assert hasattr(ThinFilmDemag(), "name")
try:
    FixedEnergyDW()
except NotImplementedError as error:
    assert "FixedEnergyDW" in str(error)
else:
    raise AssertionError("FixedEnergyDW() should raise NotImplementedError")
assert "dolfin" not in sys.modules
assert not any(
    name == "finmag.native" or name.startswith("finmag.native.")
    for name in sys.modules
)
"""
    )


def test_array_helpers_imports_dolfin_free_and_matches_master_formula():
    """``finmag.util.array_helpers.spherical_to_cartesian`` (SR1 S1b, register
    D31 exercise): a stdlib+numpy-only module split out of
    ``finmag.util.helpers`` (whose module scope imports legacy ``dolfin``) so
    ``examples/magnetic_grain/suess_2001.py`` -- which only needs the pure
    coordinate conversion -- can import it in the DOLFINx environment. Values
    are checked against master's ``(r, theta, phi) -> (x, y, z)`` formula
    (``x = r sin(theta) cos(phi)``, ``y = r sin(theta) sin(phi)``,
    ``z = r cos(theta)``), not just "does it import". [Claude Sonnet 5]"""
    _run_isolated(
        """
import sys
import numpy as np
from finmag.util.array_helpers import (
    cartesian_to_spherical, spherical_to_cartesian,
)

assert "dolfin" not in sys.modules
assert "dolfinx" not in sys.modules

# +z pole: theta=0 regardless of phi.
np.testing.assert_allclose(
    spherical_to_cartesian((1.0, 0.0, 0.0)), (0.0, 0.0, 1.0), atol=1e-12)
# +x axis: r=1, theta=pi/2, phi=0.
np.testing.assert_allclose(
    spherical_to_cartesian((1.0, np.pi / 2, 0.0)), (1.0, 0.0, 0.0), atol=1e-12)
# +y axis: r=1, theta=pi/2, phi=pi/2.
np.testing.assert_allclose(
    spherical_to_cartesian((1.0, np.pi / 2, np.pi / 2)), (0.0, 1.0, 0.0),
    atol=1e-12)
# Round-trip through the inverse.
v = (2.0, 0.3, 1.1)
np.testing.assert_allclose(
    cartesian_to_spherical(spherical_to_cartesian(v)), v, atol=1e-12)
"""
    )

    # The legacy re-export spelling still resolves to the same function object
    # (not a copy), matching the D31 fix pattern used for logging_helpers.
    # ``finmag.util.helpers`` imports legacy ``dolfin`` at module scope, so
    # this half only runs where ``dolfin`` is actually importable (the
    # DOLFINx-only environment can't even load ``helpers.py`` -- that gap is
    # exactly what this fix routes callers around).
    if importlib.util.find_spec("dolfin") is not None:
        _run_isolated(
            """
from finmag.util.array_helpers import spherical_to_cartesian as new
from finmag.util.helpers import spherical_to_cartesian as legacy_spelling
assert legacy_spelling is new
"""
        )


def test_no_unported_optional_energies_remain():
    """Historical note: before Task 19, ``energies.ThinFilmDemag`` was the
    last ``requires_legacy_dolfin=True`` optional energy, and accessing it
    without legacy ``dolfin`` installed surfaced the raw
    ``ModuleNotFoundError`` for 'dolfin' (see this test's previous version in
    git history for that exact probe). Task 19 ports ``ThinFilmDemag``
    directly and converts ``FixedEnergyDW`` to a curated by-name
    ``NotImplementedError`` deferral (not a raw import error) -- so no
    optional energy class remains ``requires_legacy_dolfin=True`` any more.
    This test asserts that fact directly instead of probing a
    ``ModuleNotFoundError`` that no longer occurs. [Claude Sonnet 5]"""
    from finmag.energies import _LAZY_EXPORTS

    assert not any(flag for (_module, _attr, flag) in _LAZY_EXPORTS.values())


def test_unported_public_access_raises_a_named_not_implemented_error():
    """Historical note: before SR1 P2.1, accessing an unported legacy-dolfin
    public name such as ``finmag.NormalModeSimulation`` in the DOLFINx
    environment surfaced the raw ``ModuleNotFoundError: No module named
    'dolfin'`` (see this test's previous version in git history for that exact
    probe). The rationale then was that the boundary must report the *real*
    missing dependency rather than swallow it. SR1 P2.1 keeps that honesty but
    upgrades the diagnosis: the lazy boundary now checks
    ``importlib.util.find_spec("dolfin")`` itself and raises a curated
    ``NotImplementedError`` that NAMES the deferred feature, which is strictly
    more informative than an import traceback pointing at an internal module.
    The legacy lane (where ``dolfin`` IS installed) still resolves the real
    object, so this test only applies without legacy dolfin.
    [Claude Opus 4.8]"""
    if importlib.util.find_spec("dolfin") is not None:
        pytest.skip("This check is for the DOLFINx environment without legacy dolfin.")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import finmag; finmag.NormalModeSimulation",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SRC_ROOT)},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr
    assert "NormalModeSimulation" in result.stderr
    assert "No module named 'dolfin'" not in result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("dolfinx") is None,
    reason="The ported Simulation export check requires the DOLFINx environment.",
)
def test_ported_public_names_bypass_legacy_dolfin():
    _run_isolated(
        """
import sys
from finmag import Simulation, sim_with

assert Simulation.__module__ == "finmag.sim.sim"
assert sim_with.__module__ == "finmag.sim.sim"
assert "dolfin" not in sys.modules
assert not any(
    name == "finmag.native" or name.startswith("finmag.native.")
    for name in sys.modules
)
"""
    )


@pytest.mark.skipif(
    importlib.util.find_spec("dolfinx") is None,
    reason="The curated MacroGeometry stub check requires the DOLFINx environment.",
)
def test_macro_geometry_top_level_export_is_ported_and_dolfin_free():
    """``finmag.MacroGeometry`` and ``finmag.energies.MacroGeometry`` resolve to
    the same ported class (Task 23), constructing a working tiling object
    without pulling legacy ``dolfin`` at import (the native ``treecode_bem``
    kernels it drives are imported lazily on demag setup, not here).
    [Claude Sonnet 5], [Claude Opus 4.8]"""
    _run_isolated(
        """
import sys
import finmag
import finmag.energies as energies

assert finmag.MacroGeometry is energies.MacroGeometry
assert finmag.MacroGeometry.__module__ == "finmag.energies.demag.fk_demag_pbc"

mg = finmag.MacroGeometry(nx=3, ny=1, dx=10.0, dy=10.0)
assert len(mg.compute_Ts(None)) == 3

assert "dolfin" not in sys.modules
assert "finmag.native.treecode_bem" not in sys.modules
"""
    )


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
assert set_logging_level.__module__ == "finmag.util.logging_helpers"
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
    "Demag2D": "finmag.energies.demag",
    "DipolarField": "finmag.energies.zeeman",
    "DiscreteTimeZeeman": "finmag.energies.zeeman",
    "EnergyBase": "finmag.energies.energy_base",
    "Exchange": "finmag.energies.exchange",
    "FixedEnergyDW": "finmag.energies.dw_fixed_energy",
    "MacroGeometry": "finmag.energies.demag",
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
