# SR1 Completion (Priority 6) — Design

**Date:** 2026-07-27. **Owner decisions relayed this date.** Branch `dolfinx-parity`
(current tip `1839d7c1`, pushed). Supersedes nothing; executes the SR1
prioritised plan's Priority 6 with owner-chosen acceptance criteria.

## Goal

Complete and declare SR1: the serial deterministic DOLFINx micromagnetic
simulator, with every owner-selected ("Now") capability working and tested,
physics and public interface validated as unchanged (or minimally,
explicitly changed), and a clear published boundary between what works and
what is deferred.

## Owner acceptance criteria (verbatim intent, 2026-07-27)

1. **Everything selected for SR1 works, with tests.** No selected-Now row may
   be declared on trust; each has a passing witness in a gated lane.
2. **Physics and interface validated unchanged / minimum changes.** Every
   deviation from master `b5015c5a` behaviour or API is either fixed or an
   explicitly ratified register row — zero rows left "pending owner decision"
   at declaration (batch-ratification, decision 1 below).
3. **Clear documentation of working vs waiting-to-port.** A user-facing
   supported-surface document, not only internal ledgers.
4. **Delayed functionality fails correctly in kept tests — via standard
   pytest mechanisms, not a custom system (owner 2026-07-27).** Carried
   not-yet-ported tests are marked
   `@pytest.mark.xfail(reason="not ported: <feature> (register row)",
   strict=True)`: gates run them unfiltered (reported as xfailed;
   `strict=True` forces marker removal the moment the feature is ported),
   the `-m "not not_ported"` gate deselection filters are removed, and the
   registered `not_ported` marker remains only as a standard selection
   label for listing the backlog. Never-ported whole master files keep
   failing at collection in the inventory lane (plain pytest behaviour).
   Tests for unported functionality are KEPT — they are the port's future
   worklist. No deletion, no silent skip.

## Owner scope decisions (2026-07-27)

1. **SR1 bar:** batch-ratify all open register rows, then declare (zero
   pendings at declaration).
2. **FULL example lane:** fix the two real defects; scale wrapper timeouts to
   measured reality in a reviewed slice; record ONE clean 17/17 FULL run as
   SR1 evidence. CI keeps the fast lane.
3. **std_prob_4:** quantitative anchor against published muMAG field-1
   reference values with a mesh-resolution-justified tolerance.
4. **Handoff:** fresh-clone install validation on this machine; publish
   `docs/SUPPORTED.md`; update README; tag `sr1`.
5. **D29:** physics investigation now, before ratification.

## Architecture: seven slices

Dependency shape: S0 → S1 → S2 → (S3 ∥ S4) → S5 → S6. S3 and S4 are
independent of each other; S5 (ratification) consumes S4's verdict; S6
declares only when S0–S5 are landed and the register is pending-free.

### S0 — standardise deferred-test failure handling (criterion 4)

- Convert every carried not-yet-ported test (34 in `sim/sim_test.py`, 8 in
  `util/helpers_test.py`, 1 in `comparison/exchange/test_exchange_field.py`,
  1 in `energies/zeeman_test.py` — the last already carries master's own
  xfail, which stays verbatim and wins) to
  `@pytest.mark.xfail(reason="not ported: <feature> (register row)",
  strict=True)` alongside the `not_ported` label marker.
- Remove the `-m "not not_ported"` filters from the four gates
  (simulation, import, timezeeman, comparison); gate pass counts must be
  unchanged with the carried tests now reported as xfailed (or errors where
  setup fails — verify empirically how xfail interacts with setup-phase
  errors and record the observed behaviour honestly; if a setup-erroring
  test cannot be made to report xfailed by standard means, it stays an
  error in the inventory lane and the gate keeps a narrowly-scoped
  deselection for THAT test only, documented inline).
- Update D30's register row text and HANDOVER's mechanism description.

### S1 — FULL-lane defect fixes

- Fix (a) the missing scheduler `save_m` keyword failure
  (`examples/cubic_anisotropy/` entry) and (b) the raw legacy `dolfin`
  import through `finmag.util.helpers` in
  `examples/magnetic_grain/suess_2001.py` (re-source the needed helpers
  dolfin-free, following the P4-viz narrow-import pattern; do NOT port all
  of `helpers.py`).
- Evidence base: the fresh FULL run (in progress at design time) supersedes
  the frozen P0.2 baseline; the slice fixes whatever REAL defects that run
  shows, expected to be these two. Wrapper timeouts are NOT touched here.
- RED-first per defect; per-example witness; no physics changes; fast lane
  and 33-gate verifier stay green.

### S2 — timeout scaling + recorded 17/17 FULL run

- Using measured per-example durations from the fresh run, set each slow
  entry's wrapper timeout to measured × 2 (rounded up to a clean figure),
  documented per entry in the wrapper. This is the separate reviewed slice
  the P0.2 protocol required before any FULL acceptance claim.
- Then execute one clean-tree FULL run; requirement: 17/17 pass (nothing
  skipped except environment-genuine skips already catalogued). Record the
  log path, durations and result in `capability-status.md` (C15 → PASS) and
  the manifest. Restore any tracked artifacts the run rewrites
  (`examples/magnetic_grain/mz.png`, `examples/std_prob_3/doc_table.rst`)
  or commit them deliberately with the diff explained.

### S3 — std_prob_4 quantitative anchor

- Extend the std_prob_4 example/test: extract first `m_x = 0` crossing time
  and `m_y` at that crossing from the `.ndt` trajectory; compare against
  published muMAG standard problem 4 (field 1) reference values with a
  tolerance derived from mesh resolution and integrator settings (derivation
  written in the test docstring; no tolerance-to-fit).
- Full-resolution assertion lives in the FULL lane; the fast lane keeps a
  reduced-resolution sanity variant. If the full-resolution result lies
  outside the justified tolerance, that is a physics finding — investigate
  before declaration, do not widen the tolerance.

### S4 — D29 physics investigation (parallel with S2/S3)

- Question: is the ~8% anisotropy-vs-Magpar residual mesh-regeneration
  drift or a masked anisotropy-field defect?
- Method: (i) evaluate the ported `UniaxialAnisotropy` field against the
  analytic expression on the exact node set the comparison uses; (ii)
  compare on the checked-in Magpar mesh nodes (`.inp`/`.femsh`) rather than
  a regenerated Netgen mesh; (iii) bisect where the 8% enters
  (field values vs node pairing vs mesh geometry).
- Output: a written verdict with numbers. Either "drift, quantified →
  ratify D29" or "defect → new fix slice, SR1 blocks until fixed".
  Physics-heavy: Opus implementation, Fable review.

### S5 — batch-ratification decision sheet

- One document (`docs/superpowers/specs/2026-07-27-sr1-ratification-sheet.md`)
  listing EVERY open register row — D1, D5, D6a/b, D7, D9, D13, D16a/b,
  D20, D21 (pending), D22–D28, D31 (documented-deferred, permanent
  disposition pending), D29 (with S4's verdict), D17/D19 permanence,
  M4a/M4b permanence, M5 (restore-vs-ratify; sheet recommendation: RESTORE
  `tests/zhangli/stt_nonlocal_test.py` and `zhang_li_test.py` from
  `b5015c5a` as retained never-ported backlog, consistent with criterion 4)
  — each with recommended disposition and evidence links.
- Owner approves/adjusts line-items in one pass. One commit updates the
  register; any row the owner redirects to "fix now" becomes its own slice
  and blocks declaration until landed.

### S6 — handoff and declaration

- Fresh-clone validation: clone the pushed branch to a scratch directory,
  `pixi run -e dolfinx dolfinx-install-editable` + `dolfinx-native-build`,
  run `dev/bin/verify-dolfinx-m5` (33/33) and a minimal demag/dynamics
  script (barmini relax + demag-sphere check) from scratch; transcript
  recorded.
- Publish `docs/SUPPORTED.md`: supported API surface, validation limits and
  evidence classes, Later list, Not-now list — distilled from
  `capability-status.md` for users, with pointers into the register.
- Update README install/quickstart for the pixi/DOLFINx environment.
- Final evidence: 33-gate verifier + inventory lane run recorded (the
  inventory failure population IS the criterion-4 artifact: kept tests for
  deferred functionality, failing correctly).
- Tag `sr1` on the declaration commit; HANDOVER gains the SR1 declaration
  section (what is declared, evidence links, open Later/Not-now backlog).

## Error handling

- Any slice discovering a real defect (S2 run, S3 outside tolerance, S4
  defect verdict, S6 clean-install failure) spawns a fix slice; declaration
  blocks until it lands. No tolerance widening, no quiet re-scoping.
- Long runs follow the controller-watchdog discipline (orchestrator-owned
  background watchdogs; detached `setsid nohup` execution).

## Testing

- Every behavioural change is RED-first with a focused gate; the 33-gate
  verifier must stay green after every slice; the inventory lane is re-run
  at S6 and its tally recorded.
- Deferred-functionality tests are never deleted or muted (criterion 4):
  by-name deferral witnesses stay in gates; carried/never-ported master
  tests stay failing in the inventory lane.

## Out of scope for SR1 (unchanged)

Normal modes/eigensolvers, thermal SLLG/LLB, MPI stepping, function-space
PBC (D19), legacy NEB, nonlocal STT (M5 row governs its witness), live
external harnesses, Demag2D/GCR/Treecode-factory exposure, movie tooling —
per the SR1 prioritised plan's deferred list.
