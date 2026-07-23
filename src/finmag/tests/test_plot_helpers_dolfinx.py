"""Import-boundary and headless-render checks for the backend-neutral
NumPy/Matplotlib plot helpers (``finmag.util.plot_helpers``).

``plot_helpers`` is pure numpy + matplotlib (plus ``Tablereader``) and must be
importable and usable in the DOLFINx environment, where legacy ``dolfin`` is
absent. Historically it was blocked by a broad ``from .helpers import *`` that
pulled ``finmag.util.helpers`` (module-scope ``import dolfin``). These tests
pin that the module imports with NO ``dolfin`` in ``sys.modules`` and that a
representative helper actually renders a real artifact headlessly. The
``mayavi``-bound ``quiver`` helper must stay deferred (it may not be silently
half-enabled).

Each check runs in a clean subprocess so ``sys.modules`` is an exact
import-side-effect witness, mirroring ``test_import_boundary.py``. [Claude Opus 4.8]
"""

import os
from pathlib import Path
import subprocess
import sys


SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent


def _run_isolated(code):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC_ROOT)
    # A headless Agg backend so pyplot never needs an X display.
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_synthetic_ndt(path):
    """Write a minimal valid .ndt file readable by ``Tablereader``.

    Format: a ``# `` comment-prefixed header line of column names, a matching
    units line, then whitespace-separated data rows (time, m_x, m_y, m_z).
    """
    import numpy as np

    ts = np.linspace(0.0, 1.0, 11)
    lines = ["# time m_x m_y m_z", "# <> <> <> <>"]
    for t in ts:
        mx = np.cos(2 * np.pi * t)
        my = np.sin(2 * np.pi * t)
        mz = 0.0
        lines.append("  {:.6g} {:.6g} {:.6g} {:.6g}".format(t, mx, my, mz))
    Path(path).write_text("\n".join(lines) + "\n")


def test_plot_helpers_imports_without_dolfin():
    """``import finmag.util.plot_helpers`` succeeds and pulls no legacy
    ``dolfin``; its backend-neutral public helpers are all present."""
    _run_isolated(
        """
import sys
import matplotlib
matplotlib.use("Agg")

import finmag.util.plot_helpers as ph

assert "dolfin" not in sys.modules, sorted(
    n for n in sys.modules if "dolfin" in n)

for name in (
    "surface_2d", "surface_3d", "plot_ndt_columns", "plot_dynamics",
    "plot_dynamics_3d", "plot_hysteresis_loop", "boxplot",
):
    assert callable(getattr(ph, name)), name
"""
    )


def test_plot_dynamics_renders_ndt_headlessly(tmp_path):
    """``plot_dynamics`` reads a synthetic .ndt via ``Tablereader``, returns a
    real matplotlib ``Figure``, and writes a non-empty PNG -- with no
    ``dolfin`` imported on the render path."""
    ndt = tmp_path / "dynamics.ndt"
    _write_synthetic_ndt(ndt)
    png = tmp_path / "dynamics.png"

    _run_isolated(
        """
import sys, os
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import finmag.util.plot_helpers as ph

fig = ph.plot_dynamics({ndt!r}, outfile={png!r})
assert isinstance(fig, Figure), type(fig)
assert os.path.exists({png!r})
assert os.path.getsize({png!r}) > 0

assert "dolfin" not in sys.modules, sorted(
    n for n in sys.modules if "dolfin" in n)
""".format(ndt=str(ndt), png=str(png))
    )
    # The artifact really lands on disk (parent-visible).
    assert png.exists() and png.stat().st_size > 0


def test_plot_hysteresis_loop_renders_headlessly(tmp_path):
    """``plot_hysteresis_loop`` (which uses the inlined
    ``create_missing_directory_components``) writes a non-empty PNG into a
    not-yet-existing subdirectory, with no ``dolfin`` on the path."""
    png = tmp_path / "nested" / "loop.png"

    _run_isolated(
        """
import sys, os
import matplotlib
matplotlib.use("Agg")

import finmag.util.plot_helpers as ph

H = [-100.0, -50.0, 0.0, 50.0, 100.0, 50.0, 0.0, -50.0, -100.0]
m = [-1.0, -0.9, 0.0, 0.9, 1.0, 0.95, 0.1, -0.85, -1.0]
ph.plot_hysteresis_loop(H, m, filename={png!r})

assert os.path.exists({png!r})
assert os.path.getsize({png!r}) > 0

assert "dolfin" not in sys.modules, sorted(
    n for n in sys.modules if "dolfin" in n)
""".format(png=str(png))
    )
    assert png.exists() and png.stat().st_size > 0


def test_quiver_stays_deferred():
    """The ``mayavi``-bound ``quiver`` helper must NOT be silently enabled: it
    still fails by raising ``ModuleNotFoundError`` for its unavailable
    ``mayavi``/``dolfin`` backend when called."""
    _run_isolated(
        """
import numpy as np
import matplotlib
matplotlib.use("Agg")

import finmag.util.plot_helpers as ph

try:
    ph.quiver(np.zeros(9), np.zeros((3, 3)))
except ModuleNotFoundError as error:
    assert error.name in ("mayavi", "dolfin"), error.name
else:
    raise AssertionError(
        "quiver must stay deferred (mayavi/dolfin absent), not run")
"""
    )
