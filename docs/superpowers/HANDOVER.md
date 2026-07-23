# Finmag DOLFINx Port — Handover

**Reconciled:** 2026-07-23

**Active branch:** `dolfinx-parity`

**Source baseline audited:** `f1a1344c423e74687ddf820c1fafc056a6271fe1`

**Maintainer contact:** Sam Holt

This is the short entry point for a human or agent resuming the port. Read, in
order:

1. [`capability-status.md`](capability-status.md) — canonical current scope,
   evidence, known failures, and the interim release target;
2. [`owner-porting-checklist.md`](owner-porting-checklist.md) — completed owner
   triage record and printable functionality checklist;
3. [`acceptance-register.md`](acceptance-register.md) — approved and pending
   behavior/deferment decisions;
4. [`master-pixi-parity-manifest.md`](master-pixi-parity-manifest.md) — atomic
   owner-Now evidence map, the pytest-discoverable inventory, and the complete
   original example inventory (retired non-discoverable tests remain decisions);
5. [`plans/2026-07-23-sr1-prioritised-plan.md`](plans/2026-07-23-sr1-prioritised-plan.md)
   — owner-informed SR1 execution order and gates;
6. [`plans/2026-07-21-dolfinx-full-parity.md`](plans/2026-07-21-dolfinx-full-parity.md)
   — historical execution detail and remaining full-parity work;
7. [`../../transition-notes.org`](../../transition-notes.org) and
   [`../../dev/dolfinx/porting_map.md`](../../dev/dolfinx/porting_map.md) —
   chronological engineering evidence; later entries supersede earlier ones.

Do not depend on `.superpowers/sdd/progress.md`: it is useful local session
memory but is git-ignored and therefore unavailable in a clean clone.

## Goal and present milestone

The final contract is functionality and public-interface parity with original
`master` (`b5015c5a`). A deviation is not accepted until the repository owner
records a disposition in the acceptance register.

The next interim goal is **SR1: a serial deterministic simulator release
candidate**. It must be installable, scientifically testable, useful through
the familiar `Simulation`/`sim_with` API, and explicit about every unavailable
surface. Thermal solvers, normal modes, MPI stepping, legacy NEB, live external
comparison harnesses, and specialist I/O outside the selected owner-Now rows
can follow SR1 without being dropped from final parity. Checked-in external
reference data remains part of SR1 validation. No public GNEB API/workflow was found on master, so it is not
an interface-parity condition; possible overlap with legacy spherical/modified
NEB algorithms must be inventoried with the NEB slice.

## What works now

The port already has a broad serial deterministic core: common energy terms,
FK and direct treecode/MacroGeometry demag, spatially varying parameters,
regions, local Slonczewski and Zhang-Li STT, SciPy and native CVODE trajectory
integration, scheduling, restart v2, NDT and field output, point/topology
utilities, common Gmsh mesh generation, editable packaging, and converted
examples. Validation is a mixture of frozen-oracle, analytic, cross-method,
regression, MPI ownership probes, and end-to-end tests; it is not all “oracle
validated”. The exact boundaries are in `capability-status.md`.

Important current limitations:

- Sundials advance works, but its `Simulation.reset_time()` and restart path
  fail because the implementation assumes a SciPy-only `.ode` attribute.
- several compatibility exports still fail through raw legacy-`dolfin`
  imports;
- `sim_with` does not yet wire the already-ported dense-FK MacroGeometry path;
  that wiring is required now, while a Treecode factory selector is deferred;
- the clean FULL baseline is recorded (12 passed, 5 failed): three unchanged
  wrapper timeouts, a missing scheduler `save_m` keyword, and a raw legacy
  `dolfin` import through `finmag.util.helpers`. It is not an all-green
  acceptance run, and std_prob4's broad switching window is only qualitative;
- normal modes, thermal SLLG/LLB, MPI stepping, function-space PBC, legacy NEB,
  external harnesses, and long-tail I/O/visualisation remain unported.

## Install and verify the supported environment

```sh
pixi run -e dolfinx dolfinx-install-editable
pixi run -e dolfinx dolfinx-native-build
```

For a complete developer verification, run one command:

```sh
dev/bin/verify-dolfinx-m5
```

The aggregate verifier already performs editable install, native build and
provenance before its focused gates; do not run all three commands redundantly
unless diagnosing an install/build failure.

At the reconciled commit, the dirty-doc aggregate baseline passed all 32 steps
on Python 3.12.13 and DOLFINx 0.10.0; its fast lane reported 14 passed and 3
skipped. The clean detached FULL baseline then reported 12 passed and 5 failed
in 42:37. Its three timeouts are harness evidence, not physics failures; the
two immediate defects are the scheduler `save_m` keyword and raw `dolfin`
import above. See the manifest for exact logs and environment versions.

## Safe execution protocol

- Use the frozen Python-3/FEniCS-2019 oracle at
  `ba9280934e188d7f3800e7b9865e70a9422f7687` through
  `dev/bin/run-legacy-oracle`. Prefer analytic physics where it is stronger,
  and label cross-method checks honestly.
- Probe uncertain DOLFINx mechanics under `dev/dolfinx`; implement accepted
  behavior once, minimally, in the existing `src/finmag` module.
- For every slice: state the API and scientific invariant, obtain RED evidence,
  make the smallest source change, run focused and aggregate gates, update the
  capability/decision documents, and obtain an independent review.
- Preserve public names and defaults unless forced or owner-approved. Every
  unavailable current API must fail by feature name, not through an incidental
  import or unrelated attribute error.
- Do not combine unrelated cleanup with a parity fix. Do not dispatch new
  implementation work until the user chooses the next slice.
- Before running the legacy oracle, inspect `git worktree list`. At this
  reconciliation a disposable partial oracle worktree remained registered at
  `/tmp/finmag-legacy-oracle.omGAF8/checkout`; it is not project state. Remove
  it only after confirming it contains no needed probe output.

## Recommended next decision

The canonical bounded slices and gates are in
[`capability-status.md`](capability-status.md#bounded-sr1-work-slices). The
recommended next slice is SR1-L1 (Sundials reset/restart), followed by SR1-A1
(top-level error boundary) and SR1-A2 (`sim_with` MacroGeometry wiring). SR1-V1
now has its baseline failure inventory; rerun it only after separately reviewed
fixes, including the scheduler and raw-import defects, have landed.
SR1-O1 is now decided: fix the stale energy and hysteresis defects, and restore
Sundials as the default after lifecycle validation. The detailed, superseding
sequence is in `plans/2026-07-23-sr1-prioritised-plan.md`.

[Codex GPT-5]
