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
integration, backend-neutral reset/reinitialisation, truthful v2 restart on
both backends, scheduling, NDT and field output, point/topology
utilities, common Gmsh mesh generation, editable packaging, and converted
examples. Since `626dfcc8`, `finmag.example` (`bar`, `barmini`, `nanowire`) and
`finmag.set_logging_level` also work in the DOLFINx environment with no legacy
`dolfin` installed. Validation is a mixture of frozen-oracle, analytic, cross-method,
regression, MPI ownership probes, and end-to-end tests; it is not all “oracle
validated”. The exact boundaries are in `capability-status.md`.

Important current limitations:

- Native Sundials is the public `Simulation`/`sim_with` default, as in legacy;
  its native construct/advance is witnessed in the core smoke. SciPy remains a
  fully supported explicit opt-in, and both backends share reset/reinitialise
  and restart lifecycle support.
- the top-level attribute boundary is repaired: the unported
  `NormalModeSimulation`, `normal_mode_simulation`,
  `example.sphere_inside_airbox` and `example.normal_modes` raise a curated
  `NotImplementedError` naming the feature. Submodule spellings are not covered:
  `from finmag.example.normal_modes import disk` still raises raw
  `ModuleNotFoundError`, and `finmag.util.helpers` as a module still imports
  legacy `dolfin` and remains unimportable in the DOLFINx environment (only its
  `set_logging_level` re-export was moved into
  `finmag.util.logging_helpers`);
- the `bar`/`barmini`/`nanowire` meshes match legacy `df.BoxMesh` on vertex
  count and on 6 tetrahedra per cuboid cell, but the intra-cuboid diagonal
  orientation convention is unverified without legacy dolfin installed;
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

At the frozen P0.2 commit, the dirty-doc aggregate baseline passed all 32 steps
on Python 3.12.13 and DOLFINx 0.10.0; its fast lane reported 14 passed and 3
skipped. The clean detached FULL baseline then reported 12 passed and 5 failed
in 42:37. P1.3's final clean main-worktree aggregate on `81fab481` exited 0
with all 32 steps green; its fast lane was 14 passed/3 skipped in 246.36s
(`/tmp/finmag-p1-default-m5.log`), and its core smoke witnesses the native
Sundials default at `t=1e-12`. It does not rerun or make green the frozen FULL
lane: its three timeouts are harness evidence, and the two immediate defects
are the scheduler `save_m` keyword and raw `dolfin` import above. See the
manifest for exact logs and environment versions.

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
P1.3 (restore Sundials as the public default) and P2.1 / SR1-A1 (core
import boundary, `626dfcc8`) are complete. The recommended next slice is
**P2.2 `sim_with` MacroGeometry wiring** (SR1-A2): wire the already-ported
dense-FK MacroGeometry path through the legacy `nx`/`ny`/`spacing` arguments,
with no Treecode selector, GCR or Demag2D work.
SR1-V1 now has its baseline failure inventory; rerun it only after separately
reviewed fixes, including the scheduler and raw-import defects, have landed.
SR1-O1 is now decided: fix the stale energy and hysteresis defects, and restore
Sundials as the default after lifecycle validation. The detailed, superseding
sequence is in `plans/2026-07-23-sr1-prioritised-plan.md`.

[Codex GPT-5]

[P2.1 updates: Claude Opus 4.8]
