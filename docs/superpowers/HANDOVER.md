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
- `Field.cross`/`Field.dot`/scalar coercion (P3.3, `c59f3438`) and
  `Field.from_generic_vector` (P3.4, `ef92eb7d`) are ported, operating on
  owned nodal rows and respecting the Task-31 public ordering; `Field.__add__`
  stays deferred. Spatially varying cubic-anisotropy axes u1/u2 (P3.5,
  `595f8335`) are ported and greenfield (no legacy oracle exists, since legacy
  crashes on any varying axis at construction); K2 physics is unchanged. The
  by-name serial function-space PBC deferral (P3.1/P3.2, register D19) is a
  deliberate owner decision, not a gap;
- the clean FULL baseline remains the recorded P0.2 evidence (12 passed,
  5 failed) — three unchanged wrapper timeouts, a missing scheduler `save_m`
  keyword, and a raw legacy `dolfin` import through `finmag.util.helpers`. No P2
  slice reran or reclassified it; SR1-V1 (FULL-lane diagnosis) is still open, and
  std_prob4's broad switching window is only qualitative;
- normal modes, thermal SLLG/LLB, MPI stepping, function-space PBC (deferred,
  D19), legacy NEB, external harnesses, HDF5/plotting/VTK-XDMF readback, and
  the rest of the long-tail I/O/convenience surface remain unported;
- **`Simulation.save_field`/`save_m` (the `.npy` snapshot surface, implemented
  in `src/finmag/sim/sim_savers.py`) has no test coverage under DOLFINx**
  (register **D31**): the implementation exists and is reachable, but master's
  three tests are carried as `not_ported` in `sim/sim_test.py`. The
  `save_field_to_vtk` (XDMF/VTK) path is a different method and *is* covered;
  do not mistake one for the other;
- **two GitHub workflows must be reconciled before any merge to `master`.**
  `.github/workflows/python3-m3.yml` invokes `dev/bin/verify-python3-m3`, which
  was deleted with the legacy lane (`62aae519`), and
  `.github/workflows/python3-core-suite.yml` inlines a stale explicit list of
  master test paths, several of which now hold DOLFINx ports (D30). Both jobs
  carry `if: github.ref_name != 'dolfinx-parity' && github.head_ref !=
  'dolfinx-parity'`, so both are **inert on this branch** and no CI here is
  red — but both are genuine dangling references on any line where they are
  live. Reconcile them (delete, or repoint at the oracle/inventory lanes) as
  part of the merge, not as a parity slice.

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

## Canonical test paths, the legacy oracle lane, and the inventory lane

Owner decision **2026-07-27** (register **D30**), implemented in the nine
commits `e10c5893` (plan) .. `62aae519`: `98e381ae`+`7d5c4177` (inventory lane
and the `not_ported` marker), `608bf67b`+`0f9c2298` (comparison/nmag),
`9dd89514` (energies/demag), `e7041dc9`+`b97f89c8` (core simulation/driver/
util), `8b96563a` (bucket-C suffix drop), `62aae519` (legacy-lane retirement),
plus this documentation commit.

**Every ported DOLFINx test now lives at its original `master` (`b5015c5a`)
path.** The `*_dolfinx.py` sibling-file convention adopted on 2026-07-25 is
retired: a port sitting *beside* its ancestor cannot be diffed against it, and
the minimal-diff phase's whole point was reviewability. Bucket-C files (the 7
genuinely-new-under-DOLFINx tests with no master ancestor) simply dropped the
suffix. Paths changed; test content did not, apart from carried master
functions, `__file__`-relative fixture-path fixups and import renames.

**How to review a port.** For any test file, the port-vs-legacy diff is:

```sh
git diff b5015c5a..HEAD -- src/finmag/sim/sim_test.py
```

That is the whole review artefact — no ledger lookup, no sibling hunting.

**How to run the legacy suite.** There is no in-tree legacy lane any more (the
`barmini-suite` pixi task and `dev/bin/verify-python3-{m3,barmini-suite,`
`core-suite,minimal-suite}` are deleted). Run it at the frozen oracle commit
`ba928093`, which carries its own task and its own copies of every file:

```sh
dev/bin/run-legacy-oracle -- pixi run --locked barmini-suite
```

**How to see the parity backlog.** `dev/bin/inventory-dolfinx-suite` is a
**non-gating** lane that collects and runs the entire `src/finmag` tree and
prints one summary line. It is not a verdict — `dev/bin/verify-dolfinx-m5`'s 33
focused gates are the verdict. Run (~27 min):

```sh
dev/bin/inventory-dolfinx-suite     # or: pixi run -e dolfinx dolfinx-src-suite-inventory
```

Result on 2026-07-27 at `62aae519`:

```
INVENTORY: passed=752 failed=34 errors=59 skipped=25 xfailed=12
```

Every failure and error in that population falls into one of six classes, all
expected, none of them move breakage (classified file-by-file in the Task-7
verification report):

1. **never-ported master files** — module-scope `import dolfin` still fails
   collection (e.g. `energies/anisotropy_test.py`, `util/meshes_test.py`,
   `energies/dmi_test.py`, `util/fileio_test.py`);
2. **retained bucket-B ancestors** (10+ files) — master files kept on purpose
   because at least one of their test functions has no named covering port
   function; they are meant to fail here until ported. The itemised list with a
   per-file reason is in the Task-3/Task-4 reports under
   `.superpowers/sdd/2026-07-27-canonical-test-paths/` (`util/meshes_test.py`,
   `drivers/tests/test_integrators.py`, `tests/bugs/test_bug_ndt_file_writing.py`,
   `tests/test_restart_simulation.py`, `tests/test_writing_data.py`,
   `scheduler/scheduler_test.py`, `tests/test_skyrmions.py`,
   `drivers/tests/sundials_nsteps_test.py`,
   `util/ode/tests/test_sundials_stiff_ode.py`,
   `tests/slonczewski/oscillator/test_oscillator.py`, plus the energies-group
   retentions);
3. **carried `not_ported` tests** — master functions transcribed verbatim under
   `NOT PORTED` banners so nothing vanished when the port took master's path:
   34 in `sim/sim_test.py`, 8 in `util/helpers_test.py`, `test_against_oommf`
   in `tests/comparison/exchange/test_exchange_field.py`, and one dipolar
   stray-field xfail in `energies/zeeman_test.py`. They are deselected from
   every gate by `-m "not not_ported"` and fail loudly here by design;
4. **one previously documented functionality gap** —
   `tests/bugs/test_bug_ndt_file_writing.py::test_ndt_writing_pretest`
   (register **D26**, the `get_field_as_dolfin_function` UFL-bool crash);
5. **one order-dependent, full-suite-only artefact** —
   `tests/test_llg.py::test_ported_llg_does_not_load_legacy_dolfin_or_native`
   fails only when an earlier test in the same process has already imported
   `finmag.native.*`; it passes in its own gate (`dolfinx-src-llg-pytest`,
   24/24);
6. **collection-error arithmetic**: the pre-move baseline was only a
   collect-only proxy (789 collected / 85 collection errors — the real
   pre-move pass/fail tally was lost with a crashed session and is recorded as
   lost, not reconstructed). Errors are down to 59 post-move.

Post-move `dev/bin/verify-dolfinx-m5` is **33/33 green with every per-gate
pass/skip/xfail/deselect count identical to the pre-move measured baselines** —
the moves regressed nothing.

**Follow-up backlog** (none of it blocking, all of it visible):

- burn down the retained ancestors (class 2) and the carried `not_ported`
  tests (class 3) — that *is* the remaining P5.3 legacy-test backlog;
- register **D31**: `Simulation.save_field`/`save_m` (`.npy`) has no witness;
- a **deferred cosmetic sweep**: ~32 in-tree files (4 of them implementation
  modules — `sim/sim.py`, `energies/cubic_anisotropy.py`,
  `energies/demag/fk_demag.py`, `energies/demag/treecode_bem.py`) still carry
  comments and docstrings naming the old `*_dolfinx.py` sibling filenames.
  These are stale prose, not stale imports (all live imports were repointed in
  `8b96563a`), so they were deliberately left alone rather than mixed into a
  move commit. `grep -rn "_dolfinx\.py" src examples` finds them all.

## Safe execution protocol

- Use the frozen Python-3/FEniCS-2019 oracle at
  `ba9280934e188d7f3800e7b9865e70a9422f7687` through
  `dev/bin/run-legacy-oracle`. Prefer analytic physics where it is stronger,
  and label cross-method checks honestly. Since `62aae519` (register **D30**)
  there is **no in-tree legacy test lane**: the `barmini-suite` pixi task and
  the `dev/bin/verify-python3-{m3,barmini-suite,core-suite,minimal-suite}`
  scripts are deleted, and the whole legacy suite runs only at the oracle
  commit, which carries its own task and its own copies of every file it
  lists:

  ```sh
  dev/bin/run-legacy-oracle -- pixi run --locked barmini-suite
  ```
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
quirk) is now **implemented in `a7b0fb87`**: `toggle_stt` honours explicit
`False` and inherits the D11 conflict guard, closing the back-door.

**All of Priority 3 is now also complete** (three ported slices plus the PBC
pair deferred), each landed as a reviewed source/test commit plus this
combined documentation commit, and integrated on `dolfinx-parity`:

- P3.1/P3.2 (serial function-space PBC probe and implementation,
  `3d9c9be6`): the probe found it BLOCKED in the supported environment (no
  `dolfinx_mpc`; DOLFINx 0.10 has no `constrained_domain`; the exchange seam
  must be operator-coupled, not value-copied), so it is **deferred past SR1**
  under owner decision register **D19**, not implemented. The `_deferred('pbc')`
  guard stands; the working demag image-lattice PBC (MacroGeometry, P2.2) is
  unaffected.
- P3.3 `Field.cross`/`Field.dot`/scalar coercion (`c59f3438`): pointwise
  per-node operations on owned nodal rows, guarded against mismatched-space
  scrambling. Gate `dolfinx-src-field-pytest` 29 -> 34 passed. Review:
  APPROVE, nothing blocking.
- P3.4 `Field.from_generic_vector` (`ef92eb7d`): restores the backend-vector
  (`dolfinx.la.Vector`/`PETSc.Vec`) entry point, owned-only read with
  MPI-ownership witnesses. Gate `dolfinx-src-field-pytest` 34 -> 35 passed;
  `dolfinx-src-field-mpi` exit 0 with `from_generic_vector_owned_only: true`.
  Review: APPROVE WITH FOLLOW-UPS, nothing blocking (both follow-ups acted on
  in the final commit).
- P3.5 spatially varying cubic-anisotropy axes (`595f8335`): callable/Field/
  Function `u1`/`u2` now form `u3 = u1 x u2` per node. GREENFIELD, no legacy
  oracle (legacy crashes at construction on any varying axis), validated by
  constant-reduction and per-region composition against the oracle-validated
  constant-axis path; K2 physics untouched. Gate
  `dolfinx-src-cubicanis-pytest` 25 -> 30 passed; `dolfinx-src-varparams-pytest`
  33 passed, unchanged. Review: APPROVE ("ship it").
- The final clean-tree `dev/bin/verify-dolfinx-m5` over the integrated
  P3.3+P3.4+P3.5 tip `595f8335` exited 0 with all 32 steps green (field 35,
  cubic anisotropy 30, varying parameters 33, fast examples 14 passed/3
  skipped in 223.66s, core smoke `integrator_backend: sundials` at
  `t=1e-12`; log `/tmp/finmag-p3-final-m5.log`). See
  `transition-notes.org`'s "Priority 3 Field completeness and varying cubic
  axes" section and `dev/dolfinx/porting_map.md` for full evidence.

**All of Priority 4 (selected I/O and convenience surface) is now also
complete** (2026-07-24), eight reviewed source/test slices plus this combined
documentation commit, integrated on `dolfinx-parity` (`1a9df5eb`):

- P4-probe `Simulation.probe_field*` (`584820a0`); P4-viz backend-neutral
  matplotlib `plot_helpers` importable (`5be9f5e8`, `surface_3d` has a
  pre-existing mpl-3.11 break, importable-not-executable); P4-M correct-physics
  `LLG.M`/`M_average` in A/m (`67e5ebe1`, register **D20**); P4-mesh
  `mesh_info`/`length_scales` diagnostics (`868f664d`); P4-init vortex
  initialiser family (`8de6927b`, skyrmion family unvalidated/call-time-broken);
  P4-helpers logging helpers extracted dolfin-free (`4acd9a81`,
  `shutdown`/instance are Simulation methods, deferred); P4-region
  `save_m_in_region` per-region `.ndt` column (`123df146`, register **D21**);
  P4-hdf5 `Field.save_hdf5`/`from_hdf5` single-`.h5` round-trip (`1a9df5eb`).
- **`h5py>=3.16.0,<4` was added to `[feature.dolfinx.dependencies]`** (owner
  decision) for the HDF5 round-trip — the first new runtime dependency of the
  dolfinx port lane; `pixi.lock` was regenerated and `import finmag` does not
  eagerly import it. Two new register rows are **pending owner decision**:
  **D20** (corrected `M`/`M_average` physics vs legacy's unit bug) and **D21**
  (`save_m_in_region` restores intent with a region-id calling convention, since
  legacy's version was itself non-functional).
- Deferred by evidence (not owner-refused): mayavi `quiver`, paraview/X
  `visualization.py`, the optional PyVista adapter (pyvista absent); and the
  Simulation instance-lifecycle/`shutdown` methods and the skyrmion initialiser
  family (a call-time 2D/3D-coordinate gap) — each a future slice.
- Final clean-tree `dev/bin/verify-dolfinx-m5` over the integrated Priority-4
  tip `1a9df5eb` (with h5py) exited 0 with all 32 steps green (field 42,
  simulation 61, meshes 53, llg 23, import 28, varparams 35, fast examples
  14 passed/3 skipped in 276.40s, core smoke `integrator_backend: sundials` at
  `t=1e-12`; log `/tmp/finmag-p4-final-m5.log`).

**All of Priority 5 (mesh and external-reference validation) is now also
complete** (2026-07-25), integrated on `dolfinx-parity` (tip `dadf35ae`):

- **P5.1 Netgen necessity probe**: no selected test or geometry was found that
  Gmsh plus the `from_geofile` text-subset loader (Task 18/30) cannot
  represent or validate. Netgen's binary backend stays deferred; the owner
  ratified leaving it deferred (register `M4a`/`M4b`). No source change.
- **P5.2 comparison data** (`133a24ff`..`48e611ad`, 7 slices): OOMMF, Nmag and
  Magpar comparisons are restored from checked-in reference data, comparing by
  coordinate probe or analytic invariant rather than raw node ordering, which
  makes them immune to the `M8` mesh-regeneration node-drift class. A new
  dolfin-free `finmag.util.magpar_io` reader replaces the legacy `dolfin`-
  dependent `finmag.util.magpar`. `nsim`/live Magpar/OOMMF execution was
  deliberately NOT resurrected (register `M1`/`M14`/`M15` SR1 deferment
  stands). The comparison gate is folded into `dev/bin/verify-dolfinx-m5`.
- **Minimal-diff test-conversion phase** (`d9226bcb`..`dadf35ae`, 31 commits,
  4 waves): every `*_dolfinx.py` test file was reshaped to read as a minimal
  diff of its `master` (`b5015c5a`) original, so a reviewer can check each
  port against its legacy ancestor line-by-line instead of trusting a
  from-scratch rewrite. **The `*_dolfinx.py` sibling-file convention this
  phase used is now superseded** — see the "Canonical test paths" section
  (register **D30**); the filenames named in this paragraph are historical.
  Bucket A (literal transcription) covers files with a
  1:1 master ancestor; bucket B adds a master->port traceability-mapping
  header where no 1:1 mapping exists (e.g. files that were split, merged, or
  renamed across the port); bucket C adds a "NO MASTER ANCESTOR" header to
  the 7 test files that are genuinely new under DOLFINx (no master file to
  restore). Restored dropped coverage discovered along the way:
  `test_sim_ode` at `1e-9`, method-of-averaging, the hysteresis D4 path,
  `get_interaction_list`, DMI unit-length invariance, three variable-params
  cases, exchange PBC, Slonczewski nmag validation,
  `test_regression_Ms_numpy_type`, `test_dipolar_field_class`. An independent
  Opus review of the whole phase returned a **FAITHFUL** verdict: no
  silently-loosened tolerance, no gutted xfail, no false restoration. The
  aggregate verifier `dev/bin/verify-dolfinx-m5` is **33/33 gates green**
  (up from 32; the comparison gate from P5.2 is the new one).

The transcription phase also surfaced four HIGH-priority parity-debt findings,
each test-pinned and left as a known divergence (fixes deferred to a later
slice, per owner decision); see `acceptance-register.md` for the itemised,
authoritative list:

- **treecode/PBC coincident-node BEM defect** (register `D17`): restoring
  master's own `demag_pbc_test.py` as a strict `xfail` reproduces the
  touching-tile periodic BEM row sums reaching `-2` instead of `-1` (a
  ~158% wrong demag field on the affected geometry) -- this is new evidence
  for the already-registered `D17` disposition (by-name refusal kept for SR1;
  the BEM kernel fix itself stays deferred), not a new row.
  (`energies/demag/demag_pbc_test.py`)
- **Magpar anisotropy comparison at 8% tolerance**: `tests/comparison/anisotropy/test_anis_magpar.py`
  had to loosen `REL_TOLERANCE` from legacy's `5e-7` (identical-mesh
  assumption) to `8e-2` to absorb `M8` mesh drift, but the measured maximum
  disagreement (~5.1e-2) is flagged in the test's own docstring as a genuine
  finmag-vs-Magpar method/discretisation disagreement, not fully explained by
  mesh drift alone -- needs a physics check. Candidate new register row.
- **`set_m`/`LLG.set_m` NaN guard missing**: the ported `Simulation.set_m()`
  does not reproduce legacy's NaN validation -- a NaN-valued `m_init` is
  accepted silently (`sim.m` ends up containing NaN) instead of raising
  `ValueError`. Pinned as `xfail(strict=True)` (`test_set_m`) so it stays
  visible and flips to XPASS the moment the guard is restored. Candidate new
  register row.
- **`get_field_as_dolfin_function` UFL-bool crash**: calling
  `get_field_as_dolfin_function('m')(point)` raises `ValueError: UFL
  conditions cannot be evaluated as bool in a Python context`. This is a
  pre-existing porting gap in `Simulation.get_field_as_dolfin_function`
  (`src/finmag/sim/sim.py`), unrelated to the NDT-writing regression test it
  was found in; fixing it means touching `sim.py`, out of scope for the
  test-only conversion phase. Candidate new register row.
  (found via `tests/bugs/test_bug_ndt_file_writing.py`)

**The canonical-test-paths restructure then followed** (2026-07-27,
`e10c5893`..`62aae519`, register **D30**): every ported test was relocated onto
its master path, the in-tree legacy lane was retired to the frozen oracle, and
the non-gating inventory lane was added. Two of the four candidate rows above
are now recorded (**D23** NaN guard, **D26** UFL-bool crash; the Magpar 8%
residual is **D29**), and one new row was opened (**D31**, no witness for the
`.npy` `save_field`/`save_m` surface). See "Canonical test paths, the legacy
oracle lane, and the inventory lane" for the run commands, the current
INVENTORY tally and the follow-up backlog.

The recommended next work is **Priority 6** (full scientific acceptance and
SR1 handoff) and the **SR1-V1 FULL-lane** diagnosis, which still shows its
recorded baseline (12 passed, 5 failed) and has not been rerun or
reclassified by any Priority 2-5 slice. All of Priority 1-5 is complete. The
detailed, superseding sequence is in
`plans/2026-07-23-sr1-prioritised-plan.md`.

[Codex GPT-5]

[P2.1 updates: Claude Opus 4.8]

[P2.2–P2.6 completion update: Claude Opus 4.8]

[P3.3–P3.5 completion update: Claude Sonnet 5]

[P5.1–P5.2 and minimal-diff test-conversion phase completion update: Claude Sonnet 5]

[Canonical test paths (D30), legacy-lane retirement and inventory lane: Claude Opus 4.8]
