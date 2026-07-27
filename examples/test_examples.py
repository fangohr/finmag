"""Task 30 gate: run the converted ``examples/`` scripts on the ported package.

This is the practical-parity witness. Each converted example is a self-contained
script that (a) builds a real legacy micromagnetic workflow on the DOLFINx port
and (b) embeds its own validation -- comparison against checked-in reference
data (exchange_demag), analytic solutions (macrospin, demag field/energy),
published muMAG reference values (std_prob_3), or physical-sanity invariants
(|m|=1, finite energies, correct qualitative dynamics). The gate runs each
script as a SUBPROCESS with the installed ``finmag`` (closest to real user
usage) and asserts it exits 0 -- i.e. its own validation passed.

Fast set (this gate): the reference-data / analytic witnesses plus the light
physical-sanity examples, kept to a few minutes total. The heavy muMAG /
long-dynamics examples (std_prob_4, cubic_anisotropy/sim STT, magnetic_grain,
the full std_prob_3 bisection, the 250-point hysteresis loop) run only when
FINMAG_EXAMPLE_FULL=1; see each script's header for the full-resolution command.

[Claude Opus 4.8]
"""
import os
import sys
import subprocess

import pytest

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
FULL = os.environ.get("FINMAG_EXAMPLE_FULL") == "1"

# (relpath, timeout_seconds). Ordered cheapest-first so failures surface fast.
FAST_EXAMPLES = [
    ("demag/test_field.py", 120),
    ("demag/test_energy.py", 120),
    ("macrospin/test_macrospin.py", 120),
    ("macrospin/test_macrospin_alpha_rtol.py", 120),
    ("varying_alpha/run.py", 120),
    ("scheduling/sim_with_scheduling.py", 120),
    ("edge_damping/damping.py", 120),
    ("exchange_1D/1d_run.py", 120),
    ("spatially-varying-anisotropy/run.py", 180),
    ("precession/run.py", 180),
    ("time-dependent-applied-field/test_appfield.py", 180),
    ("cubic_anisotropy/hysteresis.py", 240),
    ("std_prob_3/run.py", 240),
    ("exchange_demag/test_exchange_demag.py", 300),
]

SLOW_EXAMPLES = [
    ("std_prob_4/test_std_prob_4.py", 1800),
    ("cubic_anisotropy/sim.py", 3600),
    ("magnetic_grain/suess_2001.py", 3600),
]


def _run_example(relpath, timeout):
    script = os.path.join(EXAMPLES_DIR, relpath)
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(script),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg"),
    )
    if result.returncode != 0:
        raise AssertionError(
            "example {!r} failed (exit {}):\n{}".format(
                relpath, result.returncode,
                result.stdout.decode("utf-8", "replace")))


@pytest.mark.parametrize("relpath,timeout", FAST_EXAMPLES,
                         ids=[e[0] for e in FAST_EXAMPLES])
def test_example_runs(relpath, timeout):
    _run_example(relpath, timeout)


@pytest.mark.skipif(not FULL, reason="set FINMAG_EXAMPLE_FULL=1 to run the "
                    "heavy muMAG / long-dynamics examples")
@pytest.mark.parametrize("relpath,timeout", SLOW_EXAMPLES,
                         ids=[e[0] for e in SLOW_EXAMPLES])
def test_slow_example_runs(relpath, timeout):
    _run_example(relpath, timeout)
