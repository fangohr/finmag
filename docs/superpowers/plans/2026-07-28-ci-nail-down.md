# CI Nail-Down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the full-suite sweep a genuinely green CI gate: fix the 5
real failures (owner decision 2026-07-28 reverses the D24/D26 ratified
DEFERs), convert the 55 collection-error unported master files to the
guarded strict-xfail form, retire the broken legacy workflows, and add
sweep + FULL-lane CI jobs.

**Architecture:** Cheap fixes first (T1 test isolation, T2 fileio port),
then the two src slices (T3 teardown port / D24, T4 point-evaluation bug /
D26), then the mechanical 55-file conversion wave (T5) which flips the
sweep to green, then CI wiring (T6) and closure (T7). Owner decisions to
record: D24/D26 DEFER→fix (new dated decisions superseding the batch
ratification rows), and a new register row **D33** — uniform strict-xfail
mechanism for ALL unported tests, superseding the
collection-errors-as-backlog convention in D30/M5/criterion-4 wording.

**Tech Stack:** pixi env `dolfinx`, pytest 9 (strict xfail), GitHub Actions,
`dev/bin/inventory-dolfinx-suite`, 33-gate `dev/bin/verify-dolfinx-m5`.

## Global Constraints

- Branch `dolfinx-parity` (tag `sr1` stays where it is — this work lands
  AFTER the declaration; do not move the tag).
- 33-gate verifier green after every task. RED-first for every behavioural
  change. No tolerance loosening. Master reference `b5015c5a`.
- Ported/fixed code must match master semantics unless a register row says
  otherwise; every deviation gets a register entry.
- Conversion-wave file edits are MINIMAL-DIFF: py2→py3 syntax at 2to3
  level only (print parens, except-as, integer div where syntactic),
  guarded imports, decorators, nothing else. Function bodies, assertions
  and tolerances stay byte-identical modulo those mechanical fixes.
  Reasons cite register rows/family names.
- xfail markers are `strict=True` EXCEPT where a test could genuinely pass
  today (unknown) — the wave must run each converted file once and use the
  observed outcome: fails → strict; passes → NO xfail (report it — that's
  dropped-coverage treasure); flaky/order-dependent → non-strict with a
  comment (expected rare; report each).
- Commits: one per task (T5 may use one commit per wave batch);
  `Co-Authored-By: <model> <noreply@anthropic.com>`.
- The untracked `dev/benchmarks/` (stopped perf partial) must not be
  committed by any task.

---

### Task 1: De-flake the llg import-hygiene test

**Files:**
- Modify: `src/finmag/tests/test_llg.py:106`
  (`test_ported_llg_does_not_load_legacy_dolfin_or_native`)

The test asserts importing the ported llg does not pull in legacy
`dolfin`/native modules; it passes in isolation and in its gate, but fails
under whole-suite ordering because *earlier tests* legitimately import
those modules into the same process.

- [ ] **Step 1:** Reproduce: `pixi run -e dolfinx python -m pytest -q
  src/finmag/util/fileio_test.py src/finmag/tests/test_llg.py::test_ported_llg_does_not_load_legacy_dolfin_or_native`
  (or any ordering the inventory log shows preceding it) — confirm the
  cross-contamination failure mode; record the exact polluting module.
- [ ] **Step 2:** Rewrite the test to run its check in a SUBPROCESS
  (pattern already exists in `src/finmag/tests/test_example.py` —
  `_run_isolated`-style: `subprocess.run([sys.executable, "-c", ...],
  check=True)` asserting the module set inside the child). Same asserted
  contract, now order-immune.
- [ ] **Step 3:** Verify: the pairing from Step 1 now passes AND
  `pixi run -e dolfinx dolfinx-src-llg-pytest` unchanged (24 passed).
- [ ] **Step 4:** Commit: `De-flake llg import-hygiene test via subprocess
  isolation (CI T1)`.

### Task 2: Port `fileio_test.py` in place

**Files:**
- Modify: `src/finmag/util/fileio_test.py` (master file at its canonical
  path; collects but fails — `test_Table_writer_and_reader` exercises
  Tablewriter/Tablereader ndt round-trip, a surface the port HAS)

- [ ] **Step 1:** Read master's version (`git show
  b5015c5a:src/finmag/util/fileio_test.py`) and the current failure
  (`pixi run -e dolfinx python -m pytest -q src/finmag/util/fileio_test.py
  -x`). Classify each test function: portable now vs needs an unported
  surface.
- [ ] **Step 2:** Minimal-diff port (canonical-paths conventions: dolfin→
  dolfinx API swaps only, tolerances byte-identical, traceability
  docstring noting this closes a never-ported file). Any function needing
  an unported surface gets the S0 treatment: `@pytest.mark.not_ported` +
  `@pytest.mark.xfail(reason="not ported: <surface> (register <row>)",
  strict=True)`.
- [ ] **Step 3:** Add the file to the appropriate gate
  (`dolfinx-src-restart-output-pytest` is the ndt/io home — confirm by
  reading its pixi.toml list) and run that gate.
- [ ] **Step 4:** Commit: `Port fileio_test onto its canonical path (CI T2)`.

### Task 3: Port the Simulation teardown surface (D24 → fixed)

**Files:**
- Modify: `src/finmag/sim/sim.py` — port from `git show
  b5015c5a:src/finmag/sim/sim.py`: `shutdown` (:236), `close_logfile`
  (:1249), `instances_delete_all_others` (:273), `instances_list_all`
  (:283), `instances_delete_all` (:291), `instances_alive_count` (:303),
  adapted to the port's plain `instances` dict (no cyclic-ref machinery —
  master's weakref bookkeeping may simplify; semantics, not mechanism,
  are the contract).
- Modify: `src/finmag/tests/test_cyclic_references_in_sim.py` — the two
  failing master tests become the RED tests; dolfin→dolfinx mechanical
  fixes only if the file needs them to run.
- Modify: `src/finmag/sim/sim_test.py` — the carried D24 tests under the
  NOT PORTED banner (`test_clean_up`,
  `test_removing_logger_handlers_allows_to_create_many_simulation_objects`):
  their strict xfail will FLIP when this lands — verify each against
  master semantics, remove both markers, promote to live (per the
  established promote-if-passing rule). If a promoted test still fails
  for a DIFFERENT reason, STOP and report.
- Modify: `docs/superpowers/acceptance-register.md` D24 row: append dated
  owner decision `2026-07-28: owner reversed DEFER — teardown surface
  ported in <commit>`; keep the ratification history.

- [ ] **Step 1:** RED: run the two `test_cyclic_references_in_sim.py`
  tests; capture failures.
- [ ] **Step 2:** Port the six methods; match master's public behaviour
  (docstrings included); run the two tests → green.
- [ ] **Step 3:** Handle the strict-xfail flips in `sim_test.py` per
  above; run `dolfinx-src-simulation-pytest` (expect passed +N, xfailed
  −N, counts recorded).
- [ ] **Step 4:** Register + one commit: `Port Simulation teardown surface
  (CI T3, D24 fixed)`.

### Task 4: Fix `get_field_as_dolfin_function` point evaluation (D26 → fixed)

**Files:**
- Modify: `src/finmag/sim/sim.py:449-458` and/or `src/finmag/field.py` —
  the plain call `sim.get_field_as_dolfin_function('m')` returns an object
  whose point-call `m(point)` raises `ValueError: UFL conditions cannot be
  evaluated as bool`. Investigate what the returned object is (the
  `llg._m_field.f` dolfinx Function vs a UFL wrapper); the contract
  (master) is: returned object is point-evaluable, `m((x,y,z))` → 3-vector.
  The port already has correct point evaluation machinery
  (`finmag.field.evaluate_at_point` — note register D22's outer-face
  caveat, interior points are correct); route the returned object's
  `__call__` through it or return a callable wrapper preserving Function
  semantics.
- Test: `src/finmag/tests/bugs/test_bug_ndt_file_writing.py::test_ndt_writing_pretest`
  is the RED test (currently failing).
- Modify: `docs/superpowers/acceptance-register.md` D26 row: dated owner
  reversal + fix commit; note the D22 interior-only caveat inherited by
  point calls.

- [ ] **Step 1:** RED: run the pretest, capture the traceback, identify
  the returned object type.
- [ ] **Step 2:** Minimal fix honouring master's contract; the full
  `test_bug_ndt_file_writing.py` file should now run — treat remaining
  failures in that file per their own merits (report, don't paper).
- [ ] **Step 3:** Gates: `dolfinx-src-simulation-pytest`,
  `dolfinx-src-io-utils-pytest`, `dolfinx-src-field-pytest` — unchanged
  counts plus the newly-green file.
- [ ] **Step 4:** Register + commit: `Fix get_field_as_dolfin_function
  point evaluation (CI T4, D26 fixed)`.

### Task 5: Conversion wave — 55 collection-error files → guarded strict xfail

**Files:**
- The authoritative list: derive from the declaration inventory log —
  `grep -a "^ERROR" /home/sam/.claude/jobs/6b8f36a7/tmp/sr1-final-inventory.log |
  sed 's/\x1b\[[0-9;]*m//g;s/ERROR //;s/ - .*//' | sort -u` (55 files;
  minus any T2/T3 already fixed — recompute, don't assume).
- Modify: each listed file (minimal-diff conversion per Global
  Constraints); `pytest.ini` unchanged (marker exists).
- Modify: `docs/superpowers/acceptance-register.md` — new row **D33**:
  "All unported master tests fail via guarded strict xfail (owner
  2026-07-28); supersedes the collection-error convention recorded in
  D30/M5"; update D30 and M5 mechanism sentences with dated markers
  (the zhangli pair gets converted like the rest — M5's witnesses now
  surface as xfails, not collection errors).
- Modify: `docs/SUPPORTED.md` §7 + `docs/superpowers/HANDOVER.md`
  inventory-classes text: the failure mechanisms collapse to two
  (by-name NotImplementedError; strict xfail); the sweep is now expected
  GREEN and its tally shape changes (errors→0, xfailed grows by the
  per-test count of the 55 files).

Conversion recipe per file (bucket by directory; one commit per bucket,
3-5 buckets):
1. 2to3-level syntax fixes only (each annotated `# py3 syntax fix (D33)`).
2. Module-level failing imports → guarded:
   `try:  # master: import dolfin as df\n    import dolfin as df\nexcept ImportError:\n    df = None  # not ported (D33): tests below xfail`.
   Same pattern for other unavailable imports (nmag helpers, etc.).
3. Every test function: `@pytest.mark.not_ported` +
   `@pytest.mark.xfail(reason="not ported: <family> (register <row>)",
   strict=True)` — family/row from the register's Later/Not-now mapping
   (sllg→M16/thermal, neb→deferred list, etc.; the SUPPORTED.md Later
   table is the lookup).
4. Run the file: confirm every test xfails (strict). A PASSING test loses
   its xfail (report as recovered coverage); an ERROR at collection still
   = the guard missed an import — fix the guard.
5. NO gate additions — these files are inventory-lane-only until ported.

- [ ] **Step 1:** Generate the list; bucket it; record in the report.
- [ ] **Step 2-N:** Convert bucket by bucket; after each bucket run
  `pixi run -e dolfinx python -m pytest -q <bucket files>
  --continue-on-collection-errors` → expect 0 errors, all xfailed.
- [ ] **Final step:** Full sweep `dev/bin/inventory-dolfinx-suite` →
  expect `failed=0 errors=0` (passed unchanged from T1-T4 state; xfailed
  grown). Record the tally. Register D33 + docs updates + final bucket
  commit: `Convert unported master tests to guarded strict xfail (CI T5,
  D33)`.

### Task 6: CI workflows

**Files:**
- Delete: `.github/workflows/python3-m1.yml`, `python3-m2.yml`,
  `python3-m3.yml`, `python3-core-suite.yml` (the recorded merge
  blockers; legacy execution = `dev/bin/run-legacy-oracle`, already
  documented).
- Create: `.github/workflows/dolfinx-sweep.yml` — `schedule:` weekly
  (cron `0 3 * * 1`) + `workflow_dispatch:`; job runs pixi install +
  editable install + native build (mirror dolfinx-m5.yml's setup steps)
  then `dev/bin/inventory-dolfinx-suite`; GATES on the script's exit code
  (post-T5 contract: exit 0 AND `failed=0 errors=0` — add a grep
  assertion on the INVENTORY line so a regression fails the job) and
  publishes the tally to `$GITHUB_STEP_SUMMARY`.
- Create: `.github/workflows/dolfinx-full-examples.yml` —
  `workflow_dispatch:` only (runtime hours until the perf work lands;
  flip to scheduled later); runs `FINMAG_EXAMPLE_FULL=1 pixi run -e
  dolfinx dolfinx-src-examples-pytest` with `timeout-minutes: 1200`;
  summary to `$GITHUB_STEP_SUMMARY`.
- Modify: `docs/superpowers/HANDOVER.md` merge-blocker note (workflows now
  retired → blocker cleared) + a short CI section in `README.md`
  (what runs when: m5 per push; sweep weekly; FULL on demand).

- [ ] **Step 1:** Author both workflows by copying dolfinx-m5.yml's setup
  block verbatim (pixi cache etc.) — do not invent new setup.
- [ ] **Step 2:** Validate YAML (`gh workflow list` after push won't run
  here — use `python -c "import yaml,sys;yaml.safe_load(open(f))"` per
  file, and `gh api` schema check if available; otherwise note validation
  is push-time).
- [ ] **Step 3:** Delete the four legacy workflows; grep `.github/` for
  dangling references.
- [ ] **Step 4:** Docs; commit: `CI: sweep + FULL-lane workflows, retire
  legacy lanes (CI T6)`.

### Task 7: Verification + closure

- [ ] **Step 1:** Clean tree; `dev/bin/verify-dolfinx-m5` → 33/33
  (detached + orchestrator watchdog).
- [ ] **Step 2:** `dev/bin/inventory-dolfinx-suite` → `failed=0 errors=0`,
  record the full tally as the new baseline in SUPPORTED.md/HANDOVER
  (replacing 754/5/55/25/47 with dated supersession markers).
- [ ] **Step 3:** Register consistency grep (`pending` → only historical-
  with-markers); capability-status C-rows touched by D24/D26 fixes
  updated.
- [ ] **Step 4:** Commit: `CI nail-down closure: green-sweep baseline (CI
  T7)`. (Tag stays `sr1`; this work is post-declaration by design.)

---

## Self-review notes

- Owner decisions carried: fix-all-5 (T1-T4), convert-55 (T5), both dated
  and register-recorded (D24/D26 reversals, new D33).
- The T3 strict-xfail flip interaction is explicit (promote-if-passing);
  T5's observed-outcome rule prevents blanket-stamping xfail onto tests
  that would pass.
- Sweep-green contract is enforced twice: T5's final step and T7's
  baseline; the CI gate greps the INVENTORY line so future regressions
  fail the weekly job.
- No placeholder steps; exact master line numbers for the teardown port
  verified against `b5015c5a` this session.
