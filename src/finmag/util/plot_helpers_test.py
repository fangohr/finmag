"""Import-boundary and headless-render checks for the backend-neutral
NumPy/Matplotlib plot helpers (``finmag.util.plot_helpers``).

This file now lives at its master path
``src/finmag/util/plot_helpers_test.py`` (formerly
``src/finmag/tests/test_plot_helpers_dolfinx.py``), so
``git diff b5015c5a..HEAD -- src/finmag/util/plot_helpers_test.py``
shows the port diff directly.

``plot_helpers`` is pure numpy + matplotlib (plus ``Tablereader``) and must be
importable and usable in the DOLFINx environment, where legacy ``dolfin`` is
absent. Historically it was blocked by a broad ``from .helpers import *`` that
pulled ``finmag.util.helpers`` (module-scope ``import dolfin``). These tests
pin that the module imports with NO ``dolfin`` in ``sys.modules`` and that a
representative helper actually renders a real artifact headlessly. The
``mayavi``-bound ``quiver`` helper must stay deferred (it may not be silently
half-enabled).

The first test below (above the NEW banner) is the master
``plot_helpers_test.py`` test, transcribed faithfully in-process (it drives a
real ``barmini`` simulation, so it cannot run in the isolated subprocesses
used by the checks below). The remaining checks each run in a clean
subprocess so ``sys.modules`` is an exact import-side-effect witness,
mirroring ``test_import_boundary.py``. [Claude Opus 4.8]
"""

import os
from pathlib import Path
import subprocess
import sys

# dolfinx: force a headless backend up front. Nothing in the test-suite
# configures MPLBACKEND globally, and the transcribed master test below
# renders in-process (not in an isolated subprocess), so it needs the Agg
# backend selected before `finmag.util.plot_helpers` (or anything else) has
# a chance to import pyplot with an interactive backend. [Claude Opus 4.8]
import matplotlib
matplotlib.use("Agg")

import pytest

from finmag.example import barmini
from finmag.util.plot_helpers import *


SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent


# GENUINE GAP (found while porting, not introduced by this test): legacy
# ``Simulation.add()`` (``b5015c5a:src/finmag/sim/sim.py`` lines 449-458)
# registers an ``E_<interaction.name>`` energy column and an
# ``H_<interaction.name>_x/y/z`` averaged-field column in the .ndt
# ``Tablewriter`` for every interaction added. The ported
# ``Simulation.add()`` (``src/finmag/sim/sim.py`` line 380) only forwards to
# ``self.llg.effective_field.add(...)`` and never calls
# ``self.tablewriter.add_entity(...)`` for the new interaction, so a
# barmini-class simulation's .ndt file carries only
# time/m/steps/last_step_dt/dmdt -- no ``E_Demag``/``H_Exchange_x`` columns
# -- even though Exchange and (FK) Demag are both attached by ``barmini()``.
# This is NOT something this test file can fix (implementation modules under
# ``src/finmag/`` are out of scope for this port); the master invocation
# below is transcribed and invoked FAITHFULLY (verbatim columns/args) and
# marked ``xfail(strict=True)`` so the missing .ndt-column registration
# stays visible and the test flips to an error (XPASS) the moment
# ``Simulation.add()`` is fixed upstream. [Claude Opus 4.8]
@pytest.mark.xfail(strict=True, reason=(
    "GENUINE GAP: ported Simulation.add() does not register per-interaction "
    "E_<name>/H_<name>_x/y/z .ndt columns the way legacy add() does, so "
    "plot_ndt_columns(columns=[..., 'E_Demag', 'H_Exchange_x']) raises "
    "KeyError from Tablereader.__getitem__."))
@pytest.mark.requires_X_display
def test_plot_ndt_columns_and_plot_dynamics(tmpdir):
    """
    Simply check that we can call the command `plot_ndt_columns` with some arguments
    """
    os.chdir(str(tmpdir))
    sim = barmini()
    sim.schedule('save_ndt', every=1e-12)
    sim.run_until(1e-11)
    plot_ndt_columns('barmini.ndt', columns=['m_x', 'm_y', 'm_z', 'E_Demag', 'H_Exchange_x'],
                     outfile='barmini.png', title="Some awesome title",
                     show_legend=True, legend_loc='center', figsize=(10, 4))

    plot_dynamics('barmini.ndt', components='xz',
                  outfile='barmini2.png', xlim=(0, 0.8e-11), ylim=(-1, 1))

    assert(os.path.exists('barmini.png'))
    assert(os.path.exists('barmini2.png'))


# ===== NEW under DOLFINx (no master ancestor) =====


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


def test_plot_ndt_columns_renders_with_nondefault_args_headlessly(tmp_path):
    """Direct invocation of ``plot_ndt_columns`` (not via the ``plot_dynamics``
    wrapper) with the same non-default keyword arguments the master test used
    (``title``, ``show_legend``, ``legend_loc``, ``figsize``), restoring a
    real invocation of ``plot_ndt_columns`` that is independent of the
    E_Demag/H_Exchange .ndt-column gap documented above the NEW banner:
    only ``m_x``/``m_y``/``m_z`` columns (present in the synthetic .ndt) are
    requested here. Audit finding (1)."""
    ndt = tmp_path / "columns.ndt"
    _write_synthetic_ndt(ndt)
    png = tmp_path / "columns.png"

    _run_isolated(
        """
import sys, os
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import finmag.util.plot_helpers as ph

fig = ph.plot_ndt_columns(
    {ndt!r}, columns=['m_x', 'm_y', 'm_z'], outfile={png!r},
    title="Some awesome title", show_legend=True, legend_loc='center',
    figsize=(10, 4))
assert isinstance(fig, Figure), type(fig)
assert os.path.exists({png!r})
assert os.path.getsize({png!r}) > 0

assert "dolfin" not in sys.modules, sorted(
    n for n in sys.modules if "dolfin" in n)
""".format(ndt=str(ndt), png=str(png))
    )
    assert png.exists() and png.stat().st_size > 0


def test_plot_dynamics_renders_with_nondefault_args_headlessly(tmp_path):
    """``plot_dynamics`` with the same non-default arguments the master test
    used for its second call (``components='xz'``, ``xlim``, ``ylim``),
    independent of the E_Demag/H_Exchange .ndt-column gap documented above
    the NEW banner: ``components='xz'`` only needs ``m_x``/``m_z``, both
    present in the synthetic .ndt. Audit finding (2)."""
    ndt = tmp_path / "dynamics_nondefault.ndt"
    _write_synthetic_ndt(ndt)
    png = tmp_path / "dynamics_nondefault.png"

    _run_isolated(
        """
import sys, os
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import finmag.util.plot_helpers as ph

fig = ph.plot_dynamics({ndt!r}, components='xz', outfile={png!r},
                       xlim=(0, 0.8), ylim=(-1, 1))
assert isinstance(fig, Figure), type(fig)
assert os.path.exists({png!r})
assert os.path.getsize({png!r}) > 0

assert "dolfin" not in sys.modules, sorted(
    n for n in sys.modules if "dolfin" in n)
""".format(ndt=str(ndt), png=str(png))
    )
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
