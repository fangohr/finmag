"""SR1 P4-helpers: the dolfin-free logging helpers import boundary.

``finmag.util.helpers`` imports legacy ``dolfin`` at module scope, so the whole
module is unimportable in the DOLFINx environment. SR1 P2.1 already lifted
``set_logging_level`` into the stdlib-only :mod:`finmag.util.logging_helpers`.
This slice does the SAME extraction for the pure-stdlib logging helpers that
selected scripts reach for -- ``start_logging_to_file`` (per-simulation /
per-run logfile wiring) plus its two logging-introspection companions
``logging_status_str`` and ``logging_handler_str`` -- so they resolve without
legacy dolfin, while ``finmag.util.helpers`` keeps re-exporting them for the
legacy spelling.

The ``shutdown`` / ``instances_*`` / ``close_logfile`` surface named by the
scoping investigation is deliberately NOT covered here: those are
``Simulation`` *methods* in the legacy ``sim.py`` (coupled to the tablewriter /
scheduler cyclic-reference teardown and ``Simulation.instances`` bookkeeping),
not ``finmag.util.helpers`` functions, and no standalone
``instances`` / ``get_instances`` / ``finmag_instances`` helper exists anywhere
in the legacy tree. There is no ``finmag.util.helpers`` symbol to extract for
them, so they stay out of this helpers-extraction slice.

Everything runs in a clean subprocess so ``sys.modules`` is an exact witness of
what was actually imported. [Claude Opus 4.8]
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
    reason="The dolfin-free logging-helper checks require the DOLFINx environment.",
)


def _run_isolated(code, cwd=None, check=True):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# importability, dolfin-free
# --------------------------------------------------------------------------

@requires_dolfinx
def test_logging_helpers_import_without_legacy_dolfin():
    """The extracted helpers resolve from the stdlib-only module with no
    legacy ``dolfin`` anywhere in ``sys.modules``."""
    _run_isolated(
        """
import sys
from finmag.util.logging_helpers import (
    start_logging_to_file, logging_status_str, logging_handler_str,
)

assert start_logging_to_file.__module__ == "finmag.util.logging_helpers"
assert logging_status_str.__module__ == "finmag.util.logging_helpers"
assert logging_handler_str.__module__ == "finmag.util.logging_helpers"

assert "dolfin" not in sys.modules
assert "dolfinx" not in sys.modules
"""
    )


# --------------------------------------------------------------------------
# start_logging_to_file: real behaviour (attach handler + write records)
# --------------------------------------------------------------------------

@requires_dolfinx
def test_start_logging_to_file_attaches_handler_and_writes_records(tmp_path):
    """``start_logging_to_file`` adds a file handler to the ``finmag`` logger,
    returns it, and finmag log records land in the given file -- all without
    legacy dolfin."""
    result = _run_isolated(
        """
import logging
import sys
from finmag.util.logging_helpers import start_logging_to_file, logging_handler_str

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)

before = list(logger.handlers)
handler = start_logging_to_file("run.log", mode="w", level=logging.DEBUG)

# The returned object is a logging handler now attached to the finmag logger.
assert isinstance(handler, logging.Handler)
assert handler in logger.handlers
assert handler not in before

# logging_handler_str names the backing file for a file handler.
assert logging_handler_str(handler).endswith("run.log")

logger.info("HELLO_MARKER_P4HELPERS")
handler.flush()

assert "dolfin" not in sys.modules
print("OK")
""",
        cwd=tmp_path,
    )
    assert "OK" in result.stdout
    logfile = tmp_path / "run.log"
    assert logfile.exists()
    assert "HELLO_MARKER_P4HELPERS" in logfile.read_text()


@requires_dolfinx
def test_start_logging_to_file_creates_missing_directories(tmp_path):
    """Legacy contract: the helper creates any missing parent directories for
    the logfile before opening it."""
    result = _run_isolated(
        """
import logging
import os
import sys
from finmag.util.logging_helpers import start_logging_to_file

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)

path = os.path.join("nested", "deeper", "run.log")
assert not os.path.exists("nested")
handler = start_logging_to_file(path, mode="w")
logger.info("NESTED_MARKER")
handler.flush()
assert os.path.exists(path)
assert "dolfin" not in sys.modules
print("OK")
""",
        cwd=tmp_path,
    )
    assert "OK" in result.stdout
    nested = tmp_path / "nested" / "deeper" / "run.log"
    assert nested.exists()
    assert "NESTED_MARKER" in nested.read_text()


@requires_dolfinx
def test_closing_the_logfile_handler_stops_records(tmp_path):
    """The handler returned by ``start_logging_to_file`` can be torn down the
    way the legacy ``Simulation.close_logfile`` did -- close its stream and
    remove it from the ``finmag`` logger -- after which new records no longer
    reach the file. This exercises the teardown contract that the (deferred,
    Simulation-bound) ``shutdown`` path relied on, using only the extracted
    helper's return value."""
    result = _run_isolated(
        """
import logging
import sys
from finmag.util.logging_helpers import start_logging_to_file

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)

handler = start_logging_to_file("run.log", mode="w")
logger.info("BEFORE_CLOSE")
handler.flush()

# Legacy close_logfile behaviour: close the stream, drop the handler.
handler.stream.close()
logger.removeHandler(handler)
assert handler not in logger.handlers

logger.info("AFTER_CLOSE")
assert "dolfin" not in sys.modules
print("OK")
""",
        cwd=tmp_path,
    )
    assert "OK" in result.stdout
    contents = (tmp_path / "run.log").read_text()
    assert "BEFORE_CLOSE" in contents
    assert "AFTER_CLOSE" not in contents


# --------------------------------------------------------------------------
# logging_status_str / logging_handler_str: real behaviour
# --------------------------------------------------------------------------

@requires_dolfinx
def test_logging_status_and_handler_str_report_the_finmag_logger(tmp_path):
    _run_isolated(
        """
import logging
import sys
from finmag.util.logging_helpers import (
    start_logging_to_file, logging_status_str, logging_handler_str,
)

# A plain stream handler stringifies to its stream.
stream_handler = logging.StreamHandler()
assert logging_handler_str(stream_handler) == str(stream_handler.stream)

logger = logging.getLogger("finmag")
logger.setLevel(logging.DEBUG)
file_handler = start_logging_to_file("status.log", mode="w")

status = logging_status_str()
assert isinstance(status, str)
assert "Current logging status" in status
# The freshly attached file handler shows up by its backing filename.
assert "status.log" in status

assert "dolfin" not in sys.modules
""",
        cwd=tmp_path,
    )


# --------------------------------------------------------------------------
# legacy re-export spelling
# --------------------------------------------------------------------------

def test_helpers_still_reexports_logging_file_helpers():
    """The legacy lane imports these helpers from ``finmag.util.helpers``; the
    move to the stdlib-only module keeps that spelling working and pointing at
    the same objects. ``finmag.util.helpers`` itself still needs legacy dolfin,
    so this only runs in the legacy environment (mirroring the P2.1
    ``set_logging_level`` re-export guard)."""
    if importlib.util.find_spec("dolfin") is None:
        pytest.skip("finmag.util.helpers itself still needs legacy dolfin.")
    from finmag.util import helpers, logging_helpers

    assert helpers.start_logging_to_file is logging_helpers.start_logging_to_file
    assert helpers.logging_status_str is logging_helpers.logging_status_str
    assert helpers.logging_handler_str is logging_helpers.logging_handler_str
