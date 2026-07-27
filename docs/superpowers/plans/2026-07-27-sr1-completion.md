# SR1 Completion (Priority 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the seven spec slices S0–S6 so SR1 can be declared: standard
xfail handling for deferred tests, a fully green FULL example lane with
honest timeouts, a quantitative std_prob_4 anchor, the D29 physics verdict,
a pending-free acceptance register, and the fresh-clone handoff with the
`sr1` tag.

**Architecture:** Spec:
`docs/superpowers/specs/2026-07-27-sr1-completion-design.md` (read it before
any task). Task order: T1(S0) → T2(S1) → T3(S2) → T4(S3) ∥ T5(S4) → T6+T7(S5,
with an owner gate between) → T8(S6). T4 and T5 may run in either order or
interleaved but not concurrently with each other (single working tree).

**Tech Stack:** pixi env `dolfinx` (DOLFINx 0.10, py3.12), pytest 9,
`dev/bin/verify-dolfinx-m5` (33 gates), `dev/bin/inventory-dolfinx-suite`,
`dev/bin/run-legacy-oracle` (frozen commit `ba928093`).

## Global Constraints

- Branch `dolfinx-parity`; master reference `b5015c5a`. Work in the main
  tree; NO worktrees unless a task says so.
- The 33-gate verifier must be green after every task; measure gate counts
  BEFORE changing anything and reproduce them after (modulo deltas the task
  itself predicts and records).
- Deferred functionality fails via STANDARD pytest mechanisms
  (`xfail(strict=True)` with a register-row reason); tests for unported
  functionality are never deleted or muted. Master-verbatim carried code
  keeps master's own markers untouched.
- No tolerance may be loosened to make anything pass. A result outside a
  justified tolerance is a finding that blocks declaration.
- Every behavioural change is RED-first. One reviewed commit per task
  (T2 has two defect sub-commits allowed; T8 has two commits: content +
  tag). Commit style: one-line summary + body,
  `Co-Authored-By: <executing model> <noreply@anthropic.com>`.
- Long runs (> 5 min): launch `setsid nohup … </dev/null &` detached, log
  under `/home/sam/.claude/jobs/6b8f36a7/tmp/`, and the ORCHESTRATOR (not
  the subagent) arms a background watchdog on the completion condition.
- Evidence log already recorded: the fresh FULL-lane baseline run
  `/home/sam/.claude/jobs/6b8f36a7/tmp/full-lane-p6.log` (started
  2026-07-27 ~18:0x; read its final summary before T2/T3 work).

---

### Task 1 (S0): Convert carried not-ported tests to strict xfail

**Files:**
- Modify: `src/finmag/sim/sim_test.py` (34 carried tests below the
  `NOT PORTED` banner)
- Modify: `src/finmag/util/helpers_test.py` (8 carried tests)
- Modify: `src/finmag/tests/comparison/exchange/test_exchange_field.py`
  (1 carried test, `test_against_oommf`)
- Modify: `src/finmag/energies/zeeman_test.py` (1 carried test — it already
  bears master's own `@pytest.mark.xfail(reason='dolfin 1.5')`; master's
  marker STAYS and is sufficient; add only the reason-bearing label if
  absent)
- Modify: `pixi.toml` — remove `-m "not not_ported"` from the gates that
  carry it (grep for `not_ported` in pixi.toml; expected: simulation,
  import, timezeeman, comparison gate lines)
- Modify: `pytest.ini` — update the `not_ported` marker description to
  "selection label for not-yet-ported carried master tests (fail via strict
  xfail)"
- Modify: `docs/superpowers/acceptance-register.md` D30 row +
  `docs/superpowers/HANDOVER.md` mechanism paragraphs (the
  `-m` filter description → xfail description)

**Interfaces:**
- Produces: every carried test bears BOTH `@pytest.mark.not_ported` (label)
  and `@pytest.mark.xfail(reason="not ported: <short feature> (register
  <row>)", strict=True)`; gates run unfiltered.

- [ ] **Step 1: Measure.** Run `pixi run -e dolfinx
  dolfinx-src-simulation-pytest` and `dolfinx-src-import-pytest`; record
  exact tallies (last known: 78 passed/34 deselected/1 xfailed and
  46/9s/8d/1x).
- [ ] **Step 2: Probe the setup-error interaction empirically FIRST.**
  Create a throwaway file under `/home/sam/.claude/jobs/6b8f36a7/tmp/`:

```python
import pytest

class TestSetupErr:
    @classmethod
    def setup_class(cls):
        raise RuntimeError("boom")

    @pytest.mark.xfail(reason="not ported: probe", strict=True)
    def test_a(self):
        assert False
```

  Run `pixi run -e dolfinx python -m pytest -q <file> -rX`. Record whether
  the result is `xfailed` or `error`. If `error`: the 5 `TestSimulation`
  carried methods (whose `setup_class` raises under dolfinx) CANNOT be
  xfailed by standard means — for exactly those 5, keep a narrowly-scoped
  deselection in the simulation gate (`--deselect
  src/finmag/sim/sim_test.py::TestSimulation::<name>` per test, NOT an
  `-m` filter), with an inline pixi.toml comment citing this probe. All
  other carried tests proceed to xfail.
- [ ] **Step 3: Apply markers.** For each carried test add the strict-xfail
  decorator ABOVE the existing `not_ported` marker, reason naming the
  feature and register row: sim_test.py save-surface trio → `(register
  D31)`; shutdown/instance block → `(register D24)`; normal-modes/eigmode
  block → `(deferred: normal modes)`; helpers_test.py 8 → `(deferred:
  finmag.util.helpers dolfin-free port)`; `test_against_oommf` → `(manifest
  N62/C20: finmag.util.oommf not ported)`. Do NOT touch the carried
  function bodies. Where master's own `xfail`/`skipif` already governs the
  outcome (zeeman dipolar test; `test_pbc2d_m_init` skipif), leave the
  outcome markers as master wrote them.
- [ ] **Step 4: Unfilter the gates.** Remove each `-m "not not_ported"`
  (keep any Step-2 per-test `--deselect`s); update the adjacent comments.
- [ ] **Step 5: Verify.** Re-run the gates from Step 1 plus
  `dolfinx-src-timezeeman-pytest` and `dolfinx-src-comparison-pytest`.
  Expected: identical pass counts; former `deselected` counts now appear as
  `xfailed` (setup-error carve-outs excepted per Step 2). Record exact
  before/after tallies. Confirm `pixi run -e dolfinx python -m pytest
  --collect-only -q -m not_ported src/finmag | tail -3` still lists all
  carried tests (label works).
- [ ] **Step 6: Docs.** Update D30 + HANDOVER wording (filter → strict
  xfail; the "reported as failures" sentence becomes "reported as xfailed
  with register-row reasons; strict, so porting the feature forces marker
  removal").
- [ ] **Step 7: Commit** everything as one commit:
  `Standardise deferred-test handling on strict xfail (SR1 S0)`.

### Task 2 (S1): FULL-lane defect fixes

**Files:**
- Read first: final summary of
  `/home/sam/.claude/jobs/6b8f36a7/tmp/full-lane-p6.log` — enumerate every
  FAILED entry and classify real-defect vs timeout. Expected real defects:
  `cubic_anisotropy` and `magnetic_grain`; timeouts are T3's business, NOT
  yours. If the log shows a real defect beyond these two, report
  NEEDS_CONTEXT with the evidence instead of improvising.
- Modify (defect A): `src/finmag/sim/sim.py` / `src/finmag/sim/scheduler`
  area — `sim.schedule('save_m', every=10*ps)`
  (`examples/cubic_anisotropy/sim.py:63`) currently fails (the P0.2 log
  recorded a missing scheduler `save_m` keyword). Reproduce first; the fix
  is to register/route the schedulable `save_m` action onto the EXISTING
  `sim_savers` save_m surface (D31), matching legacy's
  `b5015c5a:src/finmag/sim/sim.py` scheduling contract for `save_m`
  (inspect `git show b5015c5a:src/finmag/sim/sim.py | grep -n save_m` and
  mirror the registration mechanism the port uses for `save_averages`).
- Modify (defect B): `examples/magnetic_grain/suess_2001.py:22` imports
  `from finmag.util.helpers import spherical_to_cartesian`, which imports
  legacy `dolfin`. Fix pattern (P4-viz precedent): move
  `spherical_to_cartesian` (and its inverse `cartesian_to_spherical` if
  present beside it) verbatim from `helpers.py` into a new stdlib+numpy
  module `src/finmag/util/array_helpers.py`; make `helpers.py` re-import
  them from there (legacy env keeps `from finmag.util.helpers import …`
  working); change the example to import from
  `finmag.util.array_helpers`.
- Test: RED evidence per defect = the failing example run (subprocess, as
  the wrapper does) BEFORE the fix; plus a focused unit test per fix:
  scheduler-save_m test in `src/finmag/tests/test_restart_output.py` (its
  scheduler section) asserting `schedule('save_m', every=…)` produces the
  `.npy` files; an import test
  `src/finmag/tests/test_import_boundary.py`-style check that
  `finmag.util.array_helpers` imports dolfin-free and
  `spherical_to_cartesian([1, 0, 0])`-class values match the master
  formula.

**Interfaces:**
- Consumes: Task 1's strict xfails — NOTE: if defect A's fix makes any of
  the three carried D31 save-surface tests PASS, their strict xfail will
  fail the gate loudly. That is the mechanism working: verify each such
  test against master semantics, remove its xfail+not_ported markers,
  promote it to a live test, and record the D31 row as partially/fully
  discharged in the commit body.
- Produces: `finmag.util.array_helpers.spherical_to_cartesian` (public,
  dolfin-free); schedulable `'save_m'` action.

- [ ] **Step 1:** RED for A: `cd examples/cubic_anisotropy &&
  pixi run -e dolfinx python sim.py` (or the wrapper's reduced-mode
  invocation — read the script header) — capture the exact error.
- [ ] **Step 2:** Fix A minimally; add the scheduler unit test; run
  `pixi run -e dolfinx dolfinx-src-restart-output-pytest` and
  `dolfinx-src-simulation-pytest` (watch for D31 xfail flips per
  Interfaces); re-run the example → passes in reduced mode.
- [ ] **Step 3:** Commit A: `Register schedulable save_m action (SR1 S1a,
  D31 exercise)`.
- [ ] **Step 4:** RED for B: `pixi run -e dolfinx python
  examples/magnetic_grain/suess_2001.py` fails at the helpers import
  (reduced mode; confirm from the script header how reduced/full split
  works — run the reduced form).
- [ ] **Step 5:** Fix B (module move + re-export + example import); RED
  unit test first for the new module; run `dolfinx-src-import-pytest` and
  the example reduced run.
- [ ] **Step 6:** Verify helpers re-export in the LEGACY env only if
  materialised (`.pixi/envs/default` exists): `pixi run python -c "from
  finmag.util.helpers import spherical_to_cartesian; print('ok')"`;
  otherwise record not-run.
- [ ] **Step 7:** Commit B: `Dolfin-free spherical_to_cartesian via
  array_helpers (SR1 S1b)`.

### Task 3 (S2): Timeout scaling + the recorded 17/17 FULL run

**Files:**
- Modify: `examples/test_examples.py` — the `FAST_EXAMPLES`/`SLOW_EXAMPLES`
  `(relpath, timeout)` tuples.
- Modify after the run: `docs/superpowers/capability-status.md` C15 row;
  `docs/superpowers/master-pixi-parity-manifest.md` FULL-lane section.

- [ ] **Step 1:** From the pre-fix baseline log
  (`full-lane-p6.log`) and the P0.2 records, tabulate measured wall-times
  per entry. For entries that TIMED OUT, take the timeout as a lower bound
  and consult the P0.2 baseline (`master-pixi-parity-manifest.md`) for its
  recorded durations.
- [ ] **Step 2:** Set each slow/timed-out entry's timeout to
  `ceil(measured × 2)` rounded up to the nearest 300 s, with a per-entry
  comment `# measured <X>s on 2026-07-27 (full-lane-p6.log)`. Entries that
  passed comfortably keep their timeouts.
- [ ] **Step 3:** Commit the scaling:
  `Scale FULL-lane wrapper timeouts to measured reality (SR1 S2)`.
- [ ] **Step 4:** Clean-tree check, then launch the acceptance run
  detached: `setsid nohup env FINMAG_EXAMPLE_FULL=1 pixi run -e dolfinx
  dolfinx-src-examples-pytest >
  /home/sam/.claude/jobs/6b8f36a7/tmp/full-lane-final.log 2>&1 &`
  (orchestrator arms the watchdog). Expected: **17 passed** (fast 14 which
  includes 3 skips only if those skips are environment-genuine — read the
  wrapper: the 3 fast-lane skips are the FULL-only entries, so in FULL mode
  expect 17 executed passes, 0 skipped).
- [ ] **Step 5:** After the run: `git status --short` — for
  `examples/magnetic_grain/mz.png` and `examples/std_prob_3/doc_table.rst`
  decide per file: if the diff is a regeneration of identical content
  (check `git diff --stat` + visual/py diff), restore with `git checkout
  --`; if content genuinely changed, STOP and report the diff (do not
  commit a changed scientific artifact without review).
- [ ] **Step 6:** Record in C15 + manifest: 17/17, log path, per-entry
  durations table. Commit:
  `Record SR1 FULL-lane acceptance: 17/17 (SR1 S2)`.

### Task 4 (S3): std_prob_4 quantitative anchor

**Files:**
- Modify: `examples/std_prob_4/test_std_prob_4.py` (already computes the
  first `<m_x>=0` crossing and saves `m_at_crossing.npy`; docstring cites
  published reference 0.13949 ns and the checked-in
  `m_averages_ref_martinez.txt`).
- Test: this file IS the test; it runs inside the examples gate as a
  subprocess witness.

- [ ] **Step 1:** Read the current script fully; extract what it already
  asserts (window check) and what data `m_averages_ref_martinez.txt`
  contains (columns, units — inspect the header lines).
- [ ] **Step 2:** Derive the tolerance BEFORE running anything. Written
  derivation in the docstring: crossing-time sensitivity ≈ (mesh
  discretisation error in ⟨m_x⟩ near crossing) / (d⟨m_x⟩/dt at crossing,
  read from the Martinez reference trajectory). Compute the slope from the
  reference file numerically; combine with the known Gmsh-vs-Netgen mesh
  difference class (discretisation-level). State the resulting tolerance in
  ns. Sanity floor: the tolerance must be materially tighter than the
  current 0.10–0.18 ns window, else the anchor adds nothing — if the
  derivation cannot beat the window, STOP and report why rather than
  asserting a hollow bound.
- [ ] **Step 3:** Add the FULL-mode assertions: (a) first-crossing time
  within derived tolerance of 0.13949 ns; (b) `⟨m_y⟩` at the crossing
  within the analogous derived tolerance of the reference value read from
  `m_averages_ref_martinez.txt` at its own crossing; (c) keep the existing
  window check as the reduced-mode assertion. RED-first: assert with the
  derived tolerance against the CURRENT saved trajectory if one exists
  (`dynamics.ndt` is checked in — use it to pre-flight the assertion
  logic without a 30-min run).
- [ ] **Step 4:** Full-resolution verification run (detached, watchdog):
  the example's own FULL invocation per its header. If outside tolerance:
  STOP, report the measured value — that is a physics finding, not a
  tolerance problem.
- [ ] **Step 5:** Commit: `std_prob_4: quantitative muMAG anchor (SR1 S3)`.

### Task 5 (S4): D29 physics investigation

**Files:**
- Read: `src/finmag/tests/comparison/anisotropy/test_anis_magpar.py`,
  its `magpar_result/` data (`.inp`/`.femsh` via `finmag.util.magpar_io`),
  `src/finmag/energies/anisotropy.py`, register row D29.
- Create: `docs/superpowers/specs/2026-07-27-d29-verdict.md` (the written
  verdict) plus a probe script under `dev/dolfinx/` (committed, so the
  investigation is reproducible).
- Possibly create (only if verdict = defect): nothing here — a defect
  becomes its own follow-up slice; this task never edits
  `src/finmag/energies/`.

**Steps:**
- [ ] **Step 1:** Reproduce: run the comparison, record the residual field
  map (which nodes, which magnitude, which m-configurations).
- [ ] **Step 2:** Analytic cross-check: for the test's uniaxial setup,
  H_anis(m) = (2 K1 / (mu0 Ms)) (m·a) a is exact per node. Evaluate the
  ported field on the SAME nodes and compare to the analytic values — this
  isolates "our field wrong" vs "comparison wrong" immediately.
- [ ] **Step 3:** Magpar-side check: parse the checked-in Magpar result at
  its own nodes (magpar_io), compare Magpar's values to the analytic
  formula on Magpar's nodes.
- [ ] **Step 4:** Locate the 8%: (a) if ported-vs-analytic ≈ machine
  precision and Magpar-vs-analytic shows the 8%: mesh/data provenance
  issue on the Magpar side → drift verdict with numbers; (b) if
  ported-vs-analytic shows the 8%: DEFECT — write the verdict, stop, and
  report (blocks SR1). (c) mixed → bisect node pairing (coordinate-match
  tolerance, unit scaling `.femsh` vs mesh units).
- [ ] **Step 5:** Write `2026-07-27-d29-verdict.md` (setup, numbers,
  verdict, recommended D29 disposition text) and commit with the probe
  script: `D29 verdict: <drift|defect> (SR1 S4)`.

### Task 6 (S5a): Batch-ratification decision sheet

**Files:**
- Create: `docs/superpowers/specs/2026-07-27-sr1-ratification-sheet.md`

- [ ] **Step 1:** Enumerate from `acceptance-register.md` every row whose
  disposition is not final: pendings D1, D5, D6a, D6b, D7, D9, D13, D16a,
  D16b, D20, D21, D31; documented-deferred-with-pending-permanence D22,
  D23, D24, D25, D26, D27, D28, D29 (insert T5's verdict); D17 and D19
  permanence; M4a/M4b permanence; M5 (restore-vs-ratify — sheet
  recommendation: RESTORE `src/finmag/tests/zhangli/stt_nonlocal_test.py`
  and `src/finmag/tests/zhangli/zhang_li_test.py` verbatim from
  `b5015c5a`, per acceptance criterion 4). Cross-check the register's own
  header list for any row this enumeration misses; the register is
  authoritative.
- [ ] **Step 2:** For each row: 3-line entry — current state, recommended
  disposition (copy the register's own Recommendation where sound; write
  one where the register asks the owner an open question), evidence links
  (test/file/commit). End with a summary table: N rows, recommended
  accept/defer-permanently/fix-now counts.
- [ ] **Step 3:** Commit the sheet:
  `SR1 ratification decision sheet (SR1 S5a)`. **STOP — the orchestrator
  presents the sheet to the owner. No further task until owner line-item
  decisions return.**

### Task 7 (S5b): Apply owner ratifications

**Files:**
- Modify: `docs/superpowers/acceptance-register.md` (every ratified row's
  disposition cell: `ratified 2026-07-2X (owner): <decision>`)
- Possibly restore: the two zhangli files (`git checkout b5015c5a -- …`)
  if the owner accepts the M5 restore recommendation — they land as
  never-ported master files (collection-error class in the inventory lane;
  no markers added, matching every other never-ported master file).
- Modify: `docs/superpowers/HANDOVER.md` open-items section.

- [ ] **Step 1:** Apply exactly the owner's line-item decisions (provided
  in the dispatch); no editorialising. Any row the owner marks "fix now"
  is NOT applied here — it is reported back as a new blocking slice.
- [ ] **Step 2:** If zhangli files restored: verify they appear as
  collection errors in a quick targeted run
  (`pixi run -e dolfinx python -m pytest -q --collect-only
  src/finmag/tests/zhangli --continue-on-collection-errors`), and update
  the D30/M5 rows accordingly.
- [ ] **Step 3:** Grep check: `grep -c "pending owner decision"
  docs/superpowers/acceptance-register.md` → 0 (or exactly the rows the
  owner explicitly chose to keep open, each annotated with why SR1 does
  not need it — only allowed if the owner said so).
- [ ] **Step 4:** Commit: `Apply SR1 batch ratifications (SR1 S5b)`.

### Task 8 (S6): Handoff, declaration, tag

**Files:**
- Create: `docs/SUPPORTED.md`
- Modify: `README.md` (install/quickstart), `docs/superpowers/HANDOVER.md`
  (SR1 declaration section), `docs/superpowers/capability-status.md`
  (release-target row)
- Tag: `sr1`

- [ ] **Step 1: Fresh-clone validation.** `git clone
  /home/sam/repos/finmag /home/sam/.claude/jobs/6b8f36a7/tmp/sr1-clone &&
  cd sr1-clone && git checkout dolfinx-parity`; then `pixi run -e dolfinx
  dolfinx-install-editable`, `pixi run -e dolfinx dolfinx-native-build`,
  `dev/bin/verify-dolfinx-m5` (33/33; detached + watchdog), and a minimal
  from-scratch physics script (barmini relax to `t=1e-10`, assert
  `|m|` unit-norm deviation < 1e-5 and demag+exchange energies finite;
  plus the demag-sphere analytic check via
  `tests/comparison/demag/test_demag_sphere_analytic.py` run directly).
  Save the full transcript to
  `/home/sam/.claude/jobs/6b8f36a7/tmp/sr1-clean-install.log`. Any failure
  = STOP, report (clean-install defect blocks declaration).
- [ ] **Step 2: `docs/SUPPORTED.md`.** Distil `capability-status.md` for
  users: (a) supported API table (Simulation/sim_with signatures, Field
  ops, energies, drivers, scheduling/output, examples); (b) validation
  classes and limits (oracle/analytic/cross-method per family, tolerances,
  the FULL-lane 17/17 evidence, std_prob_4 anchor value); (c) "Waiting to
  be ported" (Later) list; (d) Not-now list; (e) how deferred surfaces
  fail (by-name NotImplementedError; strict-xfail carried tests; the
  inventory lane command and current tally). Link rows to the register.
- [ ] **Step 3: README** install/quickstart: pixi env, editable install,
  verifier one-liner, barmini 5-line example, pointer to SUPPORTED.md.
- [ ] **Step 4: Final evidence.** Re-run `dev/bin/verify-dolfinx-m5`
  (33/33) and `dev/bin/inventory-dolfinx-suite` on the declaration
  candidate tree; record both tallies in HANDOVER's new "SR1 declared"
  section (with the criterion-4 sentence: the inventory failure population
  is the kept future worklist).
- [ ] **Step 5:** Commit content:
  `Declare SR1: SUPPORTED.md, README, handoff evidence (SR1 S6)`.
- [ ] **Step 6:** Tag: `git tag -a sr1 -m "SR1: serial deterministic
  DOLFINx finmag — declared 2026-07-2X"` (annotated tag on the declaration
  commit; do NOT push — the owner pushes).

---

## Self-review notes

- Spec coverage: S0→T1, S1→T2, S2→T3, S3→T4, S4→T5, S5→T6+T7 (owner gate
  between), S6→T8. All four owner acceptance criteria have concrete
  landing places (1: T3/T8 evidence; 2: T5/T7 register-clean; 3: T8
  SUPPORTED.md; 4: T1 + T7 zhangli restore + T8 inventory record).
- The T2/T1 interaction (strict xfail flipping when save_m starts working)
  is stated in T2's Interfaces block.
- No placeholders; exact file paths verified against the tree at
  `aff6a6a3` (sim.py:63 schedule call, suess_2001.py:22 import,
  m_averages_ref_martinez.txt, FAST/SLOW_EXAMPLES tuples).
