"""SR1 P2.1: the ``finmag.example`` / ``set_logging_level`` import boundary.

These checks pin down that the small "standard simulation" examples that the
manual leans on (``bar``, ``barmini``, ``nanowire``) and the public
``set_logging_level`` helper work in the DOLFINx environment with no legacy
``dolfin`` anywhere, and that the two genuinely unported example surfaces
(``sphere_inside_airbox`` and ``normal_modes``) fail with a curated,
feature-naming ``NotImplementedError`` instead of a raw
``ModuleNotFoundError: No module named 'dolfin'``.

Everything runs in a clean subprocess so ``sys.modules`` is an exact witness
of what was actually imported. [Claude Opus 4.8]
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent

requires_dolfinx = pytest.mark.skipif(
    importlib.util.find_spec("dolfinx") is None,
    reason="The ported example checks require the DOLFINx environment.",
)
requires_no_legacy_dolfin = pytest.mark.skipif(
    importlib.util.find_spec("dolfin") is not None,
    reason="This check is for the DOLFINx environment without legacy dolfin.",
)


def _run_isolated(code, cwd=None, check=True):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC_ROOT)
    # The examples write .log/.ndt files next to the cwd, so give the caller a
    # chance to point that at a tmp_path. [Claude Opus 4.8]
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# the example package itself
# --------------------------------------------------------------------------

@requires_dolfinx
def test_example_package_imports_without_legacy_dolfin():
    _run_isolated(
        """
import sys
import finmag.example as example

assert "dolfin" not in sys.modules
for name in ("bar", "barmini", "nanowire"):
    assert callable(getattr(example, name)), name
assert example.bar.__module__ == "finmag.example.bar"
assert example.barmini.__module__ == "finmag.example.bar"
assert example.nanowire.__module__ == "finmag.example.nanowire"
assert "dolfin" not in sys.modules
"""
    )


@requires_dolfinx
def test_plain_finmag_import_still_pulls_no_fem_runtime():
    """``finmag.example`` becoming importable without legacy dolfin must not
    make plain ``import finmag`` eagerly drag in a FEM runtime."""
    _run_isolated(
        """
import sys
import finmag

assert "dolfin" not in sys.modules
assert "dolfinx" not in sys.modules
assert "finmag.example" not in sys.modules
assert not any(
    name == "finmag.native" or name.startswith("finmag.native.")
    for name in sys.modules
)
"""
    )


# --------------------------------------------------------------------------
# ported examples
# --------------------------------------------------------------------------

@requires_dolfinx
def test_barmini_constructs_and_integrates_without_legacy_dolfin(tmp_path):
    _run_isolated(
        """
import sys
import numpy as np
import finmag

sim = finmag.example.barmini(name="barmini_p21")
assert "dolfin" not in sys.modules

assert sim.alpha == 0.5
assert abs(sim.mesh.geometry.x[:, 0].max() - 3.0) < 1e-12
assert abs(sim.mesh.geometry.x[:, 2].max() - 10.0) < 1e-12
assert sim.t == 0.0

sim.run_until(1e-12)
assert abs(sim.t - 1e-12) < 1e-18

m = sim.m_field.get_ordered_numpy_array_xyz().reshape((-1, 3))
norms = np.linalg.norm(m, axis=1)
assert np.allclose(norms, 1.0, atol=1e-8), norms.min()

assert "dolfin" not in sys.modules
""",
        cwd=tmp_path,
    )


@requires_dolfinx
def test_barmini_mark_regions_without_legacy_dolfin(tmp_path):
    _run_isolated(
        """
import sys
import finmag

sim = finmag.example.barmini(mark_regions=True, name="barmini_regions_p21")
assert set(sim.region_ids.keys()) == {"top", "bottom"}
assert "dolfin" not in sys.modules
""",
        cwd=tmp_path,
    )


@requires_dolfinx
def test_bar_and_nanowire_construct_without_legacy_dolfin(tmp_path):
    _run_isolated(
        """
import sys
import finmag

sim = finmag.example.bar(name="bar_p21")
assert abs(sim.mesh.geometry.x[:, 2].max() - 100.0) < 1e-12
assert sim.mesh.topology.index_map(0).size_global == 16 * 16 * 51
assert sim.alpha == 0.5

wire = finmag.example.nanowire(name="nanowire_p21")
assert abs(wire.mesh.geometry.x[:, 0].max() - 100.0) < 1e-12
assert wire.mesh.topology.index_map(0).size_global == 31 * 4 * 2

assert "dolfin" not in sys.modules
""",
        cwd=tmp_path,
    )


# --------------------------------------------------------------------------
# set_logging_level
# --------------------------------------------------------------------------

def test_set_logging_level_is_dolfin_free_and_still_validates():
    _run_isolated(
        """
import logging
import sys
import finmag

assert finmag.set_logging_level.__module__ == "finmag.util.logging_helpers"

finmag.set_logging_level("DEBUG")
assert logging.getLogger("finmag").level == logging.DEBUG
finmag.set_logging_level("EXTREMEDEBUG")
assert logging.getLogger("finmag").level == logging.EXTREMEDEBUG

try:
    finmag.set_logging_level("NOT_A_LEVEL")
except ValueError as error:
    assert "CRITICAL" in str(error)
else:
    raise AssertionError("invalid logging level should raise ValueError")

assert "dolfin" not in sys.modules
assert "dolfinx" not in sys.modules
"""
    )


def test_helpers_still_reexports_set_logging_level():
    """The legacy lane imports ``set_logging_level`` from
    ``finmag.util.helpers``; the move to a stdlib-only module keeps that
    spelling working."""
    if importlib.util.find_spec("dolfin") is None:
        pytest.skip("finmag.util.helpers itself still needs legacy dolfin.")
    from finmag.util import helpers, logging_helpers

    assert helpers.set_logging_level is logging_helpers.set_logging_level


# --------------------------------------------------------------------------
# curated deferrals
# --------------------------------------------------------------------------

@requires_no_legacy_dolfin
@pytest.mark.parametrize(
    "expression, feature",
    [
        ("finmag.NormalModeSimulation", "NormalModeSimulation"),
        ("finmag.normal_mode_simulation", "normal_mode_simulation"),
        ("finmag.example.sphere_inside_airbox", "sphere_inside_airbox"),
        ("finmag.example.normal_modes", "normal_modes"),
    ],
)
def test_deferred_surfaces_raise_named_not_implemented_error(expression, feature):
    result = _run_isolated(
        """
import finmag
try:
    {expression}
except NotImplementedError as error:
    print("MESSAGE:" + str(error))
else:
    raise AssertionError("{expression} should raise NotImplementedError")
""".format(expression=expression),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert feature in result.stdout
    assert "not ported" in result.stdout.lower() or "deferred" in result.stdout.lower()
    assert "No module named 'dolfin'" not in result.stdout
    assert "No module named 'dolfin'" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
