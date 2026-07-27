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
# SR1 S2 (2026-07-27) timeout scaling. Evidence base: SR1 Task 2's measured
# cubic_anisotropy/sim.py run (this session) plus its Opus review; the P0.2
# baseline in docs/superpowers/master-pixi-parity-manifest.md (2026-07-23
# clean FULL worktree run: fast lane 14 passed/3 skipped); this session's
# pre-fix full-lane-p6.log baseline (/home/sam/.claude/jobs/6b8f36a7/tmp/
# full-lane-p6.log, 2026-07-27, dirty tree: 12 passed, 5 failed in
# 2534.34s/0:42:14 -- those 5 failures are exactly the 5 entries this file
# retimes below); and a first acceptance-run attempt (SR1 S2 phase 1, PID
# 23183) that measured a live lower bound on cubic_anisotropy/hysteresis.py
# and, on review, refuted the std_prob_4 estimate before it could even run
# -- see the C1/C2 comments below for both. Entries not called out below
# passed comfortably in both the P0.2 fast lane and this session's post-fix
# fast-lane verification (14 passed, 3 skipped, 394.09s) and keep their
# timeouts unchanged.
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
    # C2 (SR1 S2 relaunch, 2026-07-27): the first attempt's 720s
    # (current_timeout x3, provisional) was measured LIVE and refuted -- the
    # FULL-mode subprocess was killed by this exact timeout at 22:14:34
    # local (started 22:02:33), so 720s is now a measured lower bound, not
    # a duration. Estimate: FULL mode sweeps 250 field points vs FAST
    # mode's 6 (see the script's own
    # `npoints = 250 if FINMAG_EXAMPLE_FULL else 6`), ~41.7x the sweep
    # work; the FAST-lane sweep itself is estimated to cost ~45-60s
    # excluding mesh-build/simulation-startup overhead within its 240s
    # ceiling, so 41.7x that sweep-only cost puts the FULL sweep around
    # 1900-2500s. ceil(measured x 2) on that range, rounded up for margin
    # given the live >720s failure: 6000s. Estimate-based, not a direct
    # measurement -- the relaunched acceptance run records the real
    # duration.
    ("cubic_anisotropy/hysteresis.py", 6000),
    # SR1 S2, 2nd relaunch (2026-07-28): 3600s (bounded provisional guess)
    # was measured live and refuted -- FULL mode is iteration-bounded, not
    # open-ended, and this is now a derived estimate from real artifacts.
    # FULL mode runs `bisect(energy_difference, 8, 8.5, xtol=0.1)`;
    # `energy_difference()` runs ONE vortex + ONE flower relax() (10 sims
    # total: verified empirically -- `scipy.optimize.bisect` with this
    # exact (a, b, xtol) makes exactly 5 calls to `energy_difference`,
    # independent of where the root falls in [8, 8.5], confirmed by
    # instrumenting bisect with 6 different synthetic linear roots). Hard
    # measurement: examples/std_prob_3/{data_m,data_energies}.txt (append
    # mode; first entry = the first bisect call, lfactor=8.0, vortex) show
    # ONE completed vortex relax (divisions=round(8*2)=16) from test start
    # (examples/cubic_anisotropy/hysteresis.txt mtime 23:17:16.40, marking
    # the prior FAST entry's completion) to the vortex row's write
    # (00:09:57.39) = 3161s; the following flower relax (same mesh) had
    # run >=441s without finishing when the 3600s timeout killed it at
    # 00:18:13 -- consistent with, not contradicting, a similar per-sim
    # order of magnitude. Using 3161s as the representative per-sim cost
    # (flower assumed <= vortex order, the conservative/larger choice):
    # 10 sims x 3161s = 31610s; ceil(x2) = 63220s; rounded up to the
    # nearest 300s = 63300s (17.6h ceiling; expected wall time -- not
    # timeout -- is closer to the un-doubled 31610s ~= 8.8h). NOTE for the
    # owner: this is ~35x this script's own header comment ("the legacy
    # behaviour; ~30 min" for the whole bisection) -- flagged as a
    # possible port performance regression on this workload (dense
    # per-relax() cost on a 16^3 mesh), not investigated or fixed here;
    # the timeout below is sized to the DOLFINx port's measured behaviour,
    # not the legacy comment.
    ("std_prob_3/run.py", 63300),
    ("exchange_demag/test_exchange_demag.py", 300),
]

SLOW_EXAMPLES = [
    # C1 (SR1 S2 relaunch, 2026-07-27): the first attempt's 5400s
    # (current_timeout x3, provisional) was refuted before it could even
    # run. full-lane-p6.log (this session's pre-fix FULL baseline,
    # /home/sam/.claude/jobs/6b8f36a7/tmp/full-lane-p6.log) shows this
    # entry TimeoutExpired at 1800s, having reached t=2.1e-10s of the
    # 2.0e-9s target per examples/std_prob_4/dynamics.ndt's last row
    # (corroborated by the 22 dynamics0000NN.pvtu frames it wrote,
    # spanning 18:18:48-18:47:38 local). Linear extrapolation:
    # 1800 * (2.0e-9/2.1e-10) ~= 17143s for the full dynamics trace alone.
    # In that p6 baseline run, examples/std_prob_4/m_0.npy (the cached
    # s-state relaxation output) already existed from an earlier,
    # unrelated run (mtime 11:49 that day), so
    # create_initial_s_state()/`if not os.path.exists(m_0_file)` skipped
    # the relaxation; the new _clean_ignored_outputs() below now deletes
    # m_0.npy (it matches examples/**/*.npy) before every run, so the
    # s-state relaxation always runs from scratch: +~1920s. From-scratch
    # estimate ~= 17143 + 1920 = 19063s. ceil(measured x 2) = 38126s,
    # rounded up to the nearest 300s = 38400s.
    ("std_prob_4/test_std_prob_4.py", 38400),
    # measured 3551s (~59.2 min) on 2026-07-27
    # (/tmp/cubic_anisotropy_sim_run2.log: background command started
    # 2026-07-27T17:20:06.90Z, log's own "EXIT:0" line written at
    # 2026-07-27T18:19:17.56Z -- read directly off the log file's own
    # timestamps, not the ~55 min figure floated before this measurement).
    # timeout = ceil(measured x 2) rounded up to the nearest 300s = 7200s.
    # M4: this entry shares examples/cubic_anisotropy/ with the FAST-lane
    # cubic_anisotropy/hysteresis.py entry above, so
    # _clean_ignored_outputs() run before either one also wipes the
    # other's gitignored byproducts (disk*.h5/.xdmf mesh cache,
    # hysteresis.txt, disksim*.npy/.ndt). Benign: neither script reads the
    # other's output, and mesh regeneration is cheap (seconds).
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

    Best-effort: if ``git`` itself is unavailable (or any other OS-level
    failure occurs invoking it), this is logged as a warning and execution
    continues -- a missing/broken ``git`` must not fail all 17 examples
    before any of them get a chance to run.
    """
    # Scrub any ambient GIT_DIR/GIT_WORK_TREE so a caller's environment
    # can't redirect the clean at a different worktree than REPO_ROOT.
    env = dict(os.environ)
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    try:
        subprocess.run(
            ["git", "-C", REPO_ROOT, "clean", "-fdX", "--", example_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,  # best-effort: a non-zero exit shouldn't mask the example's own result
        )
    except OSError as exc:
        print("WARNING: pre-run cleanup skipped for {!r}: {}".format(
            example_dir, exc))


def _tail_output(data, n=30):
    """Decode subprocess output and return its last ``n`` lines, for
    embedding in a failure message. Used so a timeout leaves behind rate
    data (progress the child printed before being killed) instead of
    silently discarding it -- see the C1/std_prob_3 timeout comments
    above, which had to be set without this information.
    """
    if not data:
        return "(no output captured before timeout)"
    lines = data.decode("utf-8", "replace").splitlines()
    return "\n".join(lines[-n:])


def _run_example(relpath, timeout):
    script = os.path.join(EXAMPLES_DIR, relpath)
    example_dir = os.path.dirname(script)
    _clean_ignored_outputs(example_dir)
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=example_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg"),
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "example {!r} timed out after {}s; last output before kill "
            "(up to 30 lines -- rate data for setting the next "
            "timeout):\n{}".format(
                relpath, timeout, _tail_output(exc.stdout))) from exc
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
