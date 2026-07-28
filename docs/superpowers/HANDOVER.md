# Finmag DOLFINx Port — Handover

**Reconciled:** 2026-07-28 (SR1 declaration)

**Active branch:** `dolfinx-parity`

**Source baseline audited:** `f1a1344c423e74687ddf820c1fafc056a6271fe1`

**Maintainer contact:** Sam Holt

> **SR1 is declared** (tag `sr1`, 2026-07-28). Jump to
> ["SR1 declared"](#sr1-declared-2026-07-28) for what that means, the evidence,
> and what comes next. Users should read
> [`../SUPPORTED.md`](../SUPPORTED.md), not this file.

This is the short entry point for a human or agent resuming the port. Read, in
order:

0. [`../SUPPORTED.md`](../SUPPORTED.md) — the user-facing SR1 contract:
   supported API, validation classes and tolerances, Later list, Not-now list,
   and how deferred surfaces fail;
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
- the FULL example lane stands at **15 of 17 green at full workload**, with
  `std_prob_4` and `magnetic_grain` deferred by owner decision to a post-
  performance re-run. The 2026-07-23 P0.2 baseline (12 passed, 5 failed) is
  **superseded** — see "SR1 declared" below and the manifest's "FULL-lane
  closure at SR1 declaration";
- normal modes, thermal SLLG/LLB, MPI stepping, function-space PBC (deferred,
  D19), legacy NEB, external harnesses, HDF5/plotting/VTK-XDMF readback, and
  the rest of the long-tail I/O/convenience surface remain unported;
- **`Simulation.save_field`/`save_m` (the `.npy` snapshot surface, implemented
  at `src/finmag/sim/sim.py:639-665`; `sim_savers.py` is dead legacy code) has
  only PARTIAL test coverage under DOLFINx** (register **D31**): `f7517688`
  added a live witness for the *scheduled* `save_m` path, but direct
  (non-scheduled) `save_field`/`save_m` calls remain uncovered and master's
  three tests are still carried as `not_ported` in `sim/sim_test.py`. The
  `save_field_to_vtk` (XDMF/VTK) path is a different method and *is* covered;
  do not mistake one for the other;
- **merge blocker cleared 2026-07-28 (CI T6): legacy workflows removed.**
  `.github/workflows/python3-m1.yml`, `python3-m2.yml`, `python3-m3.yml`
  (which invoked `dev/bin/verify-python3-m3`, deleted with the legacy lane in
  `62aae519`), and `python3-core-suite.yml` (which inlined a stale explicit
  list of master test paths, several of which now held DOLFINx ports, D30)
  are all deleted. CI on this branch is now `.github/workflows/dolfinx-m5.yml`
  (push/PR fast gate), `.github/workflows/test-python.yml` (weekly + on-demand
  full-suite inventory sweep, gated on `failed=0 errors=0`), and
  `.github/workflows/test-slow.yml` (on-demand-only heavy FULL-workload
  examples lane).

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
the minimal-diff phase's whole point was reviewability. Bucket-C files (the 8
genuinely-new-under-DOLFINx tests with no master ancestor -- 7 under
`src/finmag/` plus `examples/test_examples.py`) simply dropped the suffix.
Paths changed; test content did not, apart from carried master functions,
`__file__`-relative fixture-path fixups and import renames.

A master ancestor was DELETED when every one of its test functions had a
named covering port function -- with two disclosed exceptions in `e7041dc9`:
`tests/zhangli/zhang_li_test.py::test_zhangli_sllg` and
`tests/zhangli/stt_nonlocal_test.py::test_zhangli` were deleted on the
strength of a deferral-guard witness (`test_nonstandard_kernels_are_deferred
[sllg]` / `[llg_stt]` in `test_stt.py`) that asserts `NotImplementedError` by
name rather than a covering port function reproducing the physics. Both
exceptions are disclosed in `test_stt.py`'s own mapping-header; the second one
also removed the sole master witness for the nonlocal-`LLG_STT` gap tracked as
register **M5**, so that gap no longer surfaces as a failing test in the
inventory lane (see register D30/M5 for the full disclosure and the pending
owner decision on restoration vs. retroactive ratification).

**2026-07-28 update (SR1 S5b, owner batch ratification, M5 RESTORE):** both
files were restored verbatim from `b5015c5a`; they fail at collection
(Python-2 source; also module-scope `import dolfin`), so
`dev/bin/inventory-dolfinx-suite` again surfaces the nonlocal-`LLG_STT` gap in
the non-gating lane (+2 collection errors, counted in this document's own
inventory delta below). Restore-vs-ratify is no longer a pending decision —
see register D30/M5 for the full disposition.

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
verification report).

Classes 1 and 2 describe two different things -- class 1 is the *mechanism*
(why the file fails to collect/run), class 2 is the *retention reason* (why
the file is still in the tree) -- and they can overlap: a file can have a
module-scope `import dolfin` (class-1 mechanism) AND be kept on purpose
because of partial port coverage (class-2 reason). `energies/anisotropy_test.py`,
`energies/dmi_test.py` and `util/meshes_test.py` are exactly this overlap --
they fail collection on `import dolfin` like any never-ported file, but they
are in the tree as class-2 retained bucket-B ancestors (see the itemised list
below), not because nobody ever attempted to port them:

1. **never-ported master files** — module-scope `import dolfin` still fails
   collection, with no port ever attempted and no retention story beyond
   "nobody ported it" (e.g. `field_setters_test.py`, `sim/sim_helpers_test.py`,
   `util/fileio_test.py` — a function-scope `import dolfin` that fails at call
   time rather than at collection, same never-ported outcome);
2. **retained bucket-B ancestors** (10+ files) — master files kept on purpose
   because at least one of their test functions has no named covering port
   function; they are meant to fail here until ported. The itemised list with a
   per-file reason was in the Task-3/Task-4 reports under
   `.superpowers/sdd/2026-07-27-canonical-test-paths/`; that session record no
   longer exists — the authoritative in-repo record is this document's six
   inventory classes (files named inline) plus git history (`util/meshes_test.py`,
   `drivers/tests/test_integrators.py`, `tests/bugs/test_bug_ndt_file_writing.py`,
   `tests/test_restart_simulation.py`, `tests/test_writing_data.py`,
   `scheduler/scheduler_test.py`, `tests/test_skyrmions.py`,
   `drivers/tests/sundials_nsteps_test.py`,
   `util/ode/tests/test_sundials_stiff_ode.py`,
   `tests/slonczewski/oscillator/test_oscillator.py`, plus the energies-group
   retentions -- `energies/anisotropy_test.py`, `energies/dmi_test.py`,
   `energies/magnetostatic_field_test.py`, `energies/test_energies_in_regions.py`,
   `tests/test_dmi_terms.py`, `energies/demag/fk_demag_2d_test.py`);
3. **carried `not_ported` tests** — master functions transcribed verbatim under
   `NOT PORTED` banners so nothing vanished when the port took master's path:
   34 in `sim/sim_test.py`, 8 in `util/helpers_test.py`, `test_against_oommf`
   in `tests/comparison/exchange/test_exchange_field.py`, and one dipolar
   stray-field xfail in `energies/zeeman_test.py`. SR1 S0 standardised their
   handling on **strict `xfail`** rather than gate filtering: every carried
   test bears the `not_ported` selection label, and 35 of the 44 additionally
   gained their own `@pytest.mark.xfail(reason="not ported: <feature>
   (register <row>)", strict=True)`. Master's own pre-existing `xfail`/
   `skipif` markers, where present, are left as master wrote them and govern
   the outcome instead rather than gaining a second marker — 5 are
   skipif-governed (`test_pbc2d_m_init` and four unconditional
   `skipif("True")` module-level tests), and 4 keep master's own **non-strict**
   `xfail` (`test_mark_regions`, `test_setting_different_material_parameters_
   in_different_regions`, `test_profile`, the zeeman dipolar test), so the
   strict-flip guarantee (an unexpected pass fails the gate) covers those 35
   newly-marked tests, not these 9. The four gates that used to carry
   `-m "not not_ported"` (`dolfinx-src-simulation-pytest`,
   `dolfinx-src-import-pytest`, `dolfinx-src-timezeeman-pytest`,
   `dolfinx-src-comparison-pytest`) now run unfiltered; the carried tests are
   reported as xfailed (or skipped, where master's skipif governs) with
   register-row reasons;
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

**2026-07-28 update (CI T5, register D33): classes 1 and 2 no longer surface
as `errors=`.** The CI-nail-down plan's Task 5 converted the remaining
never-ported master files (class 1, plus the class-2 retained-ancestor files
that overlapped it — `energies/anisotropy_test.py`, `energies/dmi_test.py`,
`util/meshes_test.py` and the rest of the itemised list above) from raw
collection errors to the same guarded-import + `not_ported`/strict-`xfail`
mechanism class 3 already used (see register **D33** for the full mechanism
and commit list `d2dcb332`..`732bc72a`, plus straggler commit `828f45d6`
for two files whose `import dolfin` only fires inside `setup_module()` at
test-setup time and so were invisible to a `--collect-only`-derived file
list). The six-class taxonomy above still
describes *why* each file is in the tree (retention reason), but the
*mechanism* column now collapses everywhere to the two described in
[`SUPPORTED.md` §7](../SUPPORTED.md): by-name `NotImplementedError`, and
strict `xfail`. `dev/bin/inventory-dolfinx-suite` is expected `errors=0` from
here on; the CI T6 weekly/dispatch workflow greps the `INVENTORY:` line for
that shape so a regression fails the job. Nine tests across the converted
files turned out not to need the guarded import at all and were left
unmarked as recovered live coverage (listed in register D33), and two
`sim/sim_helpers_test.py` tests surfaced a genuine functional divergence
unrelated to the import guard (restart-data round-trip, register D16a/D16b)
that is now visible as a named `xfail` instead of being masked by the
collection error.

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
  legacy's version was itself non-functional) (both since ratified ACCEPT,
  2026-07-28 batch).
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
slice, per owner decision, at the time this phase ran); see
`acceptance-register.md` for the itemised, authoritative list. **2026-07-28
update (CI T3/T4):** two of the four -- the teardown surface (D24) and the
`get_field_as_dolfin_function` UFL-bool crash (D26) -- are now FIXED, not
deferred; see their register rows for the fix commits. The other two (the
Magpar 8% residual, D29; the `set_m` NaN guard, D23) remain as described
below:

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

**Superseded.** Priority 6 / SR1-V1 was executed as the SR1-completion
pipeline (S0–S6, 2026-07-27..28) and **SR1 is now declared** — see
["SR1 declared"](#sr1-declared-2026-07-28) below for the current standing,
evidence and next work. The paragraph above is retained as the state of play
at the end of Priority 5.

## Open items (2026-07-28 update)

**Batch ratification landed 2026-07-28 (SR1 S5b).** The repository owner
agreed to every recommendation in the
[SR1 batch-ratification decision sheet](specs/2026-07-27-sr1-ratification-sheet.md)
(13 ACCEPT, 16 DEFER, 14 DROP, 1 RESTORE, 0 FIX NOW across 44 rows: the 42
open `acceptance-register.md` rows plus two new rows, **D32** and **P1**).
`acceptance-register.md` is now **pending-free**: every previously-open row
in the range D1-D32, M1-M16, P1 bears the 2026-07-28 batch stamp (rows with
earlier final ratifications -- e.g. D2-D4, D8, D10-D12, D14-D15, D18, D30 --
are unchanged) -- 44 rows carry a finalised `ratified 2026-07-28 (owner,
batch): ...` disposition; see the sheet for full per-row rationale and
evidence. The M5 witnesses (`src/finmag/tests/zhangli/stt_nonlocal_test.py`,
`zhang_li_test.py`) were restored verbatim from `b5015c5a` per the sheet's
RESTORE recommendation -- they fail at collection (Python-2 source; also
module-scope `import dolfin`): pytest's AST-rewrite import raises
`SyntaxError: Missing parentheses in call to 'print'` on the Python-2 `print`
statements before either file's `import dolfin` is ever reached, so a future
porter should read this as "needs a real port", not just a missing legacy
`dolfin` install. No pixi gate references them.

**S6 documentation obligations this ratification creates** (for whoever picks
up S6/T8):

- **D6a/D6b** (`UniaxialAnisotropy` axis normalisation: constant axis
  normalised, varying axis used as supplied) -- both ACCEPTed; document both
  contracts in `docs/SUPPORTED.md`.
- **D16a/D16b** (restart does not reapply/validate material/interaction
  metadata; varying-material restart metadata is a lossy scalar summary) --
  both ACCEPTed (D16a as the SR1 restart contract, D16b relabelled
  *informational*); document in `docs/SUPPORTED.md`.
- **D32** (legacy `method=` names raise by name; default changed to
  `box-assemble`) -- ACCEPTed option (a), keep the raise-by-name; document the
  interface change and the changed default in `docs/SUPPORTED.md`.
- **M12a/M12b/M12c** (legacy matrix/project/direct energy-assembly
  implementations) -- DROPped permanently, which **implies a wording fix, not
  a test deletion**: the `NotImplementedError` runtime message in
  `src/finmag/energies/energy_base.py` currently says the methods are "not yet
  ported to DOLFINx", which contradicts a permanent drop, and the two gated
  deferral tests (`src/finmag/tests/test_energies.py:291`/`:298`,
  `src/finmag/tests/test_dmi.py:167`/`:172`) must be reworded from "deferred"
  to "removed".
- **capability-status.md disposition sweep** -- `capability-status.md`
  currently still cites three now-superseded dispositions: line ~92 (`D20`
  "pending owner decision"), line ~96 (`D21` "pending owner decision"), and
  line ~129 (`D17` "disposition pending owner decision"). All three rows are
  now finalised in `acceptance-register.md` (D20 ACCEPT, D21 ACCEPT, D17
  DEFER past SR1) as of this 2026-07-28 batch ratification; T8 must sweep
  `capability-status.md` for every disposition reference and update it to
  match, so the two documents do not contradict each other.

None of these five items are SR1-blocking (all ratified ACCEPT/DROP, not FIX
NOW), but they are unclosed documentation/wording debt this ratification
created and should not be silently dropped.

## SR1 declared (2026-07-28)

**Tag:** `sr1` (annotated) on the declaration commit, branch `dolfinx-parity`.
Not pushed — the owner pushes.

**User-facing contract:** [`../SUPPORTED.md`](../SUPPORTED.md). That file, not
this one, is what a user reads: supported API, validation classes and
tolerances, the muMAG anchor, the "waiting to be ported" list, the
permanently-dropped list, and the three mechanisms by which an unavailable
surface fails.

### What is declared

A **serial, deterministic DOLFINx micromagnetic simulator** with a substantial
legacy-compatible `Simulation`/`sim_with` interface and an explicit,
register-traceable statement of everything that is unavailable. Concretely:

- installs from a clean clone via pixi + editable install + native build, and
  `finmag.example.barmini()` runs;
- Exchange, DMI, uniaxial and cubic anisotropy (incl. spatially varying axes),
  the Zeeman family, FK demag, treecode/MacroGeometry demag, ThinFilmDemag,
  spatially varying parameters, regions and each **local** STT mode compose in
  serial simulations;
- `run_until`, `relax`, `hysteresis`, scheduling, `.ndt`, `.npy` snapshots,
  VTK/XDMF write, HDF5 field round-trip and coordinate-aware v2 restart work
  through the documented lifecycle;
- native Sundials is the public default and SciPy a fully supported opt-in;
  both construct, advance, reset/reinitialise, schedule, save and restart;
- physics is covered by oracle, analytic, cross-method, external-reference-data,
  regression and end-to-end tests, per family, at stated tolerances;
- every unavailable public surface fails by **feature name**, never through an
  incidental raw `dolfin` import (two recorded holes remain, both in C02).

**SR1 is explicitly not full parity.** Thermal SLLG/LLB, normal modes, legacy
NEB, general MPI stepping, function-space PBC, nonlocal `LLG_STT`, live external
harnesses and the long I/O tail remain in the parity backlog.

### Evidence

**1. Clean-install validation (fresh clone, 2026-07-28).** `git clone` of this
repository into `/home/sam/.claude/jobs/6b8f36a7/tmp/sr1-clone`, `git checkout
dolfinx-parity` at `5e572745`, then `pixi run -e dolfinx
dolfinx-install-editable`, `dolfinx-native-build`, `dolfinx-provenance-check`,
`dev/bin/verify-dolfinx-m5`, and a from-scratch physics script. Full transcript:
`/home/sam/.claude/jobs/6b8f36a7/tmp/sr1-clean-install.log`.

- provenance resolved to the clone's own `src/finmag/__init__.py`;
- `dev/bin/verify-dolfinx-m5` **exit 0, all 33 steps green**;
- barmini relax to `t = 1e-10`: max `|m|` unit-norm deviation
  **9.969e-08** (< 1e-5), `E_demag = 4.641e-21 J`, `E_exch = 5.942e-24 J`,
  both finite;
- `src/finmag/tests/comparison/demag/test_demag_sphere_analytic.py` run
  directly: **1 passed**.

There is no clean-install defect.

**2. Declaration-tree verifier and inventory.** See the two tallies recorded
below under "Tallies at declaration".

**3. FULL example lane — 15 of 17, two deferred by owner decision.** This is
the one acceptance criterion that is **declaration-qualified**, and it is
recorded as such rather than rounded up.

- **15 of the 17 entries are witnessed green at full workload.** Fourteen —
  every `test_example_runs` entry, including the two that had failed every
  earlier FULL attempt, `cubic_anisotropy/hysteresis.py` and `std_prob_3/run.py`
  (the complete 10-simulation bisection) — passed in the attempt-3 acceptance
  run (9.2 h, log
  `/home/sam/.claude/jobs/6b8f36a7/tmp/full-lane-final.log`). The fifteenth,
  `cubic_anisotropy/sim.py`, was separately measured green at 3551 s
  (~59.2 min) on 2026-07-27 (`/tmp/cubic_anisotropy_sim_run2.log`, SR1 T2),
  after `f7517688` registered `save_m` as a scheduler shortcut.
- **The remaining two — `std_prob_4/test_std_prob_4.py` (full 2 ns trace) and
  `magnetic_grain/suess_2001.py` (full three-field physics) — are DEFERRED BY
  OWNER DECISION (2026-07-28)** to a re-run scheduled after
  [`plans/2026-07-28-post-sr1-performance.md`](plans/2026-07-28-post-sr1-performance.md),
  because those two entries are the primary beneficiaries of the planned
  speedups (their measured-rate budgets are 38400 s and 21600 s). This is an
  owner-decided **sequencing** choice, not a silent evidence downgrade:
  `capability-status.md` C15 stays **partial (declaration-qualified)**, never
  PASS.
- **`std_prob_4` additionally carries the T4 quantitative muMAG anchor**
  (commit `5e572745`): tolerance **8.089 ps**, derived a priori from the mesh
  and the reference trajectory before any output was compared
  (`eps = (1/8)(h/l_ex)^2` with `l_ex = 5.6858 nm`, `h = 5.830 nm`, divided by
  the reference slope `|d<m_x>/dt| = 16.246 /ns`), 4.9x tighter than the coarse
  switching window. Pre-flight against the checked-in 10 ps-sampled partial
  trajectory: **PASS, `|dt| = 0.75 ps`, 11x inside tolerance**. It is therefore
  **pre-flighted, not full-resolution confirmed** — confirming it end-to-end is
  part of the same deferred re-run.
- This also closes **SR1-T3 phase 2**. The full FULL-lane record, including the
  timeout-scaling evidence chain (`20a0a22c`, `71dfc064`, `c02809bf` — every
  budget derived from a measured rate, never raised to make a run pass) and the
  `doc_table.rst` artifact-policy outcome, is in
  [`master-pixi-parity-manifest.md`](master-pixi-parity-manifest.md),
  "FULL-lane closure at SR1 declaration".

**4. Decision ledger.** `acceptance-register.md` is pending-free: all 44 open
rows (D1–D32, M1–M16, P1) carry a finalised owner disposition as of the
2026-07-28 batch ratification. The five S6 documentation obligations that
ratification created (D6a/D6b axis contracts, D16a/D16b restart contract, D32
`method=` rejection, the M12a–c "not yet ported" → "removed" rewording, and the
`capability-status.md` disposition sweep) are all discharged in the declaration
commit.

### Criterion 4 — the inventory lane is the kept worklist

**The `dev/bin/inventory-dolfinx-suite` failure population is not a defect
list and not a CI verdict: it is the deliberately kept future worklist, and it
is expected to be non-empty at SR1.** Master files that were never ported stay
in the tree and fail visibly there rather than being silently deleted; the
verdict is `dev/bin/verify-dolfinx-m5`'s 33 focused gates, which are green.
Every failure and error falls into one of the six classes itemised in "Canonical
test paths, the legacy oracle lane, and the inventory lane" above.

### Tallies at declaration

Measured 2026-07-28 on the declaration-candidate tree (logs
`/home/sam/.claude/jobs/6b8f36a7/tmp/sr1-final-verify.log` and
`sr1-final-inventory.log`):

```
dev/bin/verify-dolfinx-m5         exit 0 — all 33 steps green
dev/bin/inventory-dolfinx-suite   INVENTORY: passed=754 failed=5 errors=55 skipped=25 xfailed=47
```

**Inventory delta vs the 2026-07-27 baseline** (`INVENTORY: passed=752
failed=34 errors=59 skipped=25 xfailed=12` at `62aae519`). Every number moved,
and every move is accounted for — the population did not shrink because
anything was hidden:

- **errors 59 → 55 = −6 + 2.** The **+2** are the restored M5 witnesses
  `src/finmag/tests/zhangli/stt_nonlocal_test.py` and
  `zhang_li_test.py`, brought back verbatim from `b5015c5a` by the SR1 S5b
  batch ratification's **RESTORE** decision as retained never-ported backlog.
  Both are Python-2 source, so pytest's AST-rewrite import raises
  `SyntaxError: Missing parentheses in call to 'print'` before either file's
  module-scope `import dolfin` is even reached — two collection errors, zero
  collected. That is exactly what criterion 4 prescribes: the
  nonlocal-`LLG_STT` gap (register **M5**) surfaces in the non-gating lane
  instead of vanishing. No pixi gate references `zhangli`, so the 33-gate
  verdict is unaffected. The **−6** are six `sim/sim_test.py::TestSimulation`
  entries (`test_length_scales`, `test_save_field`, `test_save_field_scheduled`,
  `test_save_m`, `test_sim_sllg`, `test_sim_sllg_time`) whose carried-master
  `setup_class` fails on `df.BoxMesh`: SR1 S0 gave them strict `xfail` markers,
  so pytest now reports them as **xfailed** rather than as setup errors. They
  are the same six items, relabelled — none was fixed or removed.
- **failed 34 → 5 and xfailed 12 → 47 (−29 / +35).** The same SR1 S0 change:
  35 carried `not_ported` master tests gained
  `@pytest.mark.xfail(reason="not ported: … (register <row>)", strict=True)`,
  so 29 of them moved failed → xfailed and the 6 above moved error → xfailed.
  Nothing was deleted and nothing was made non-strict; **strict** means each
  flips the gate red the day someone ports the feature.
- The 5 residual failures are all previously-classified:
  `tests/bugs/test_bug_ndt_file_writing.py::test_ndt_writing_pretest`
  (register **D26**), the two
  `tests/test_cyclic_references_in_sim.py` cases (the dropped teardown surface,
  **D24**), `tests/test_llg.py::test_ported_llg_does_not_load_legacy_dolfin_or_native`
  (the order-dependent full-suite-only artefact — 24/24 in its own gate), and
  `util/fileio_test.py::test_Table_writer_and_reader` (never-ported master file).
- **passed 752 → 754** from the SR1 S1a scheduler-`save_m` witness and its
  neighbours.

Note for the next runner: the inventory lane writes two untracked artifacts
(`src/finmag/sim/nanodisk_with_spherical_particle.{h5,xdmf}`) that are **not**
gitignored. They were deleted before the declaration commit; do not let them
into a commit. **2026-07-28 update (CI T7):** these two files are now
gitignored (`.gitignore`), so this note describes a problem that no longer
recurs.

**2026-07-28 update (CI T7): the declaration tally above is SUPERSEDED.**
CI T1-T5 fixed all 5 real inventory-lane failures (D23, D24, D26, and the two
llg/fileio ordering-artifact/never-ported items) and converted the remaining
53 never-ported files from raw collection `errors=` to guarded strict `xfail`
(register **D33**). Re-measured 2026-07-28 on the CI-nail-down-closure tree
(logs `/home/sam/.claude/jobs/6b8f36a7/tmp/t7-verify.log` and
`t7-inventory.log`):

```
dev/bin/verify-dolfinx-m5         exit 0 — all 33 steps green
dev/bin/inventory-dolfinx-suite   INVENTORY: passed=769 failed=0 errors=0 skipped=46 xfailed=271
```

`errors=0`/`failed=0` is now the expected, enforced shape: the
`test-python.yml` weekly/on-demand CI job greps the inventory line for this.
CI tiers (unchanged since CI T6, see "merge blocker cleared" above and
`README.md`'s "Continuous integration" section, kept consistent here):
`dolfinx-m5.yml` runs the 33-gate fast witness on every push/PR;
`test-python.yml` runs this full-suite inventory weekly (Mondays) on the
default branch plus on-demand, gated on `failed=0 errors=0`; `test-slow.yml`
runs the heavy `FINMAG_EXAMPLE_FULL=1` example lane on-demand only (no
schedule). Every remaining `skipped`/`xfailed` entry is either a
master-governed skip/xfail, a D33-converted never-ported file, or the D22
outer-face caveat — none is a CI verdict; the verdict stays the 33 focused
gates.

### Next work

> **This is a priority ORDER, not a standing execution order. Nothing below
> starts without an explicit owner instruction.** The performance plan is
> **planned, NOT started**: the repository owner issued a stop-order on
> 2026-07-28, and the untracked `dev/benchmarks/` directory holds the
> **stopped PERF-T1 partial** from before that stop. Do not resume it, do not
> treat item 2's "after (1)" as authorisation to begin item 1, and do not
> commit `dev/benchmarks/`. Work begins only when the owner says so.

1. **[`plans/2026-07-28-post-sr1-performance.md`](plans/2026-07-28-post-sr1-performance.md)
   — register P1.** *(Planned; awaiting an explicit owner start.)* Root-caused
   2026-07-28: the port re-does `fem.form(...)` +
   `assemble_vector(...)` on **every** field evaluation
   (`src/finmag/energies/energy_base.py:183`), where legacy's default
   `box-matrix-petsc` assembled the field operator once at setup and merely
   applied it (`b5015c5a:src/finmag/energies/energy_base.py:214-222`); the
   numpy LLG RHS versus master's compiled `Equation` backend (**M3**) compounds
   it. The plan rebuilds assemble-once as an **internal** optimisation of
   `box-assemble` semantics — it does **not** resurrect the removed legacy
   `method=` names (**D32** stands).
2. **Complete the FULL lane to 17/17.** This is *sequenced* after (1) — it is
   the reason the two entries were deferred — but it inherits (1)'s gate: it
   starts when the owner starts (1) and (1) lands, not automatically. Re-run
   `std_prob_4/test_std_prob_4.py` and `magnetic_grain/suess_2001.py` at full
   workload, confirm the 8.089 ps muMAG anchor end-to-end on a completed 2 ns
   trajectory, recalibrate the wrapper timeouts **downward** to the new
   measured rates, and lift C15 from *partial (declaration-qualified)*.
3. Then the parity backlog proper: the inventory lane's retained ancestors and
   carried `not_ported` tests, and the Later list in
   [`../SUPPORTED.md`](../SUPPORTED.md) §5.
4. **Before any merge to `master`**, reconcile the two dangling GitHub
   workflows described under "Important current limitations" above — they are
   inert on `dolfinx-parity` but genuine dangling references anywhere else.

[Codex GPT-5]

[P2.1 updates: Claude Opus 4.8]

[P2.2–P2.6 completion update: Claude Opus 4.8]

[P3.3–P3.5 completion update: Claude Sonnet 5]

[P5.1–P5.2 and minimal-diff test-conversion phase completion update: Claude Sonnet 5]

[Canonical test paths (D30), legacy-lane retirement and inventory lane: Claude Opus 4.8]

[SR1 S5b batch ratification and M5 restore: Claude Sonnet 5]

[SR1 declaration (S6): Claude Opus 4.8]
