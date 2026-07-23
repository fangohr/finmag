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
- `sim_with` now wires the dense-FK MacroGeometry path through the legacy
  `nx`/`ny`/`spacing_x`/`spacing_y` arguments (P2.2, pitch semantics in mesh
  units, no `unit_length` scaling); it refuses the numerically-broken
  touching/overlapping pitch (`pitch <= extent`) by name (divergence D17). A
  Treecode factory selector, GCR and Demag2D remain deferred;
- callable pin masks work again through `Simulation.pins` (P2.3), the
  discrete-time Zeeman energy tracks the current field (P2.4, D3), and
  configuring both local STT modes raises `ValueError` (P2.6, D11); the D4
  hysteresis re-relax "defect" was found not reproducible and closed as a
  diagnosis correction (P2.5);
- the clean FULL baseline remains the recorded P0.2 evidence (12 passed,
  5 failed) — three unchanged wrapper timeouts, a missing scheduler `save_m`
  keyword, and a raw legacy `dolfin` import through `finmag.util.helpers`. No P2
  slice reran or reclassified it; SR1-V1 (FULL-lane diagnosis) is still open, and
  std_prob4's broad switching window is only qualitative;
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
[`capability-status.md`](capability-status.md#bounded-sr1-work-slices). **All of
Priority 1 and Priority 2 is now complete**, each slice landed as a reviewed
source/test commit plus a separate documentation commit, and integrated on
`dolfinx-parity`:

- P2.1 core import boundary (`626dfcc8`); P2.2/P2.2a `sim_with` MacroGeometry
  wiring + guard (`2cf4647f`/`571df59f`/`899d3906`, divergence D17); P2.3
  callable pin masks (`df5fc1a1`); P2.4 discrete-time Zeeman energy fix
  (`ff906f11`, register D3); P2.6 conflicting-STT-mode guard (`de518888`,
  register D11).
- P2.5 hysteresis (register D4) was resolved as an owner-approved **diagnosis
  correction, not a source fix** (`0ad5a0e3`): the re-relax defect was found not
  reproducible — `hysteresis()` already re-relaxes every stage — and the on-axis
  oracle stall is a genuine degenerate Stoner-Wohlfarth saddle, reclassified as a
  degeneracy pin with new non-degenerate switching witnesses.
- The final clean-tree `dev/bin/verify-dolfinx-m5` over the complete Priority-2
  stack exited 0 with all 32 steps green (STT 21, simulation 49, treecode
  22 passed/1 xfailed, timezeeman+hysteresis 34, import 19 passed/3 skipped,
  fast examples 14 passed/3 skipped in 222.03s, core smoke
  `integrator_backend: sundials` at `t=1e-12`; log `/tmp/finmag-p2-final-m5.log`).

Two owner decisions were newly recorded in
[`acceptance-register.md`](acceptance-register.md). **D17** (the touching/
overlapping macro-geometry pitch refusal) is now **approved 2026-07-23**: the
by-name refusal is the accepted SR1 contract, and the coincident-node BEM kernel
fix is deferred to a separate later slice. **D18** (the `toggle_stt` back-door
that bypasses the D11 guard, plus its `toggle_stt(False)` flip-not-force-off
quirk) is being taken up as a hardening slice.

The recommended next work is **Priority 3** (serial function-space PBC probe,
`Field.cross`/`dot`/coercion, `from_generic_vector`, varying cubic axes) and the
**SR1-V1 FULL-lane** diagnosis, which still shows its recorded baseline
(12 passed, 5 failed) and has not been rerun or reclassified by any P2 slice.
The detailed, superseding sequence is in
`plans/2026-07-23-sr1-prioritised-plan.md`.

[Codex GPT-5]

[P2.1 updates: Claude Opus 4.8]

[P2.2–P2.6 completion update: Claude Opus 4.8]
