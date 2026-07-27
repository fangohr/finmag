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
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
FULL = os.environ.get("FINMAG_EXAMPLE_FULL") == "1"

# (relpath, timeout_seconds). Ordered cheapest-first so failures surface fast.
#
# SR1 S2 (2026-07-27) timeout scaling. Evidence base: SR1 Task 2 measured run
# (this session) plus its Opus review, and the P0.2 baseline in
# docs/superpowers/master-pixi-parity-manifest.md. Entries not called out
# below passed comfortably in the P0.2 clean FULL run (14 fast passed, total
# 42:14) and keep their timeouts unchanged.
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
    # provisional x3 (current_timeout x 3): P0.2 recorded this entry hitting
    # its 240s wrapper ceiling with no progress-rate data captured (manifest
    # "harness evidence, not a physics failure"); factor is
    # provisional-pending-measurement -- the SR1 S2 FULL acceptance run
    # measures the real duration.
    ("cubic_anisotropy/hysteresis.py", 720),
    # provisional x3, same basis as cubic_anisotropy/hysteresis.py above
    # (P0.2: 240s wrapper ceiling, no rate data recorded).
    ("std_prob_3/run.py", 720),
    ("exchange_demag/test_exchange_demag.py", 300),
]

SLOW_EXAMPLES = [
    # provisional x3, same basis as the two FAST entries above (P0.2: 1800s
    # wrapper ceiling, no rate data recorded).
    ("std_prob_4/test_std_prob_4.py", 5400),
    # measured 3551s (~59.2 min) on 2026-07-27
    # (/tmp/cubic_anisotropy_sim_run2.log: background command started
    # 2026-07-27T17:20:06.90Z, log's own "EXIT:0" line written at
    # 2026-07-27T18:19:17.56Z -- read directly off the log file's own
    # timestamps, not the ~55 min figure floated before this measurement).
    # timeout = ceil(measured x 2) rounded up to the nearest 300s = 7200s.
    ("cubic_anisotropy/sim.py", 7200),
    # NOT measured to completion (SR1 Task 2): field 1 of 3 alone ran
    # ~50m48s before the session ended it, and Opus review's
    # reviewer-validated extrapolation from that partial run (~72.6 min per
    # simulated ns on the 419-vertex mesh x 3 field strengths x failsafe 2ns
    # each with early-break on switching) puts field 1 alone at ~6750-8700s
    # and the full 3-field run at "several hours". Recorded deviation from
    # the plan's measured x2 rule: timeout set to 21600s (6h) as
    # extrapolation x margin, not a measured x2 scaling. The real duration
    # is to be recorded from this SR1 S2 acceptance run itself; no separate
    # pre-measurement run was attempted.
    ("magnetic_grain/suess_2001.py", 21600),
]


def _clean_ignored_outputs(example_dir):
    """Remove git-ignored byproducts left in an example's own directory by a
    previous run, before executing it again.

    The FULL lane is effectively one-shot per checkout: some examples write
    incremental output files (e.g. cubic_anisotropy/sim.py's ``save_m``,
    via ``FieldSaver``) that refuse to overwrite themselves on a second run
    and raise ``IOError``. Those outputs (``*.npy``, ``*.ndt``, ``*.h5``,
    ``*.xdmf``, plot files, ...) are all git-ignored (see ``.gitignore``),
    so they are invisible to ``git status`` and the failure is silent until
    someone re-runs the suite in the same checkout.

    We scope ``git clean -fdX`` to the single example's own directory.
    ``-X`` restricts removal to paths matched by ``.gitignore`` -- git
    refuses to remove anything that is tracked or that is untracked but NOT
    ignored, so this can never delete a tracked file or a non-ignored
    untracked file, even if a stray file happens to sit in the same
    directory. This is deliberately chosen over an explicit per-example
    glob list: the glob list would need hand-maintenance as examples grow
    new output kinds, while ``-fdX`` only ever touches what ``.gitignore``
    already declares disposable.
    """
    subprocess.run(
        ["git", "clean", "-fdX", "--", example_dir],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,  # best-effort: a cleanup hiccup shouldn't mask the example's own result
    )


def _run_example(relpath, timeout):
    script = os.path.join(EXAMPLES_DIR, relpath)
    example_dir = os.path.dirname(script)
    _clean_ignored_outputs(example_dir)
    result = subprocess.run(
        [sys.executable, script],
        cwd=example_dir,
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
