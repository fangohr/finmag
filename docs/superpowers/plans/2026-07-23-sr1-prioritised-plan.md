# Finmag SR1 Prioritised Implementation Plan

**Decision basis:** owner checklist reviewed with Hans Fangohr; behavioral
answers relayed 2026-07-23.

**Target:** the earliest trustworthy, deterministic serial micromagnetic
simulator with the selected everyday interface and explicit deferred backlog.

This plan supersedes historical numeric task ordering for SR1. Full-master
parity remains the eventual target. A checklist **Not now** tick means deferred,
not deleted and not an accepted permanent exception.

## Resolved interpretation of the owner checklist

- Exchange remains essential; only its obsolete matrix/project/direct assembly
  algorithms are not short-term work.
- The working serial LLG remains in SR1. Callable pin masks are Now; MPI is
  Later.
- MacroGeometry through dense FK demag is Now. Treecode factory exposure is
  Later/needs review and is not coupled to MacroGeometry work.
- `finmag.example` and barmini are Now; normal-mode exports remain Later.
- Existing checked-in OOMMF/Nmag/Magpar reference data is sufficient for SR1;
  live legacy external programs are not required.
- Netgen has no demonstrated required workflow yet. Investigate first; do not
  port the binary backend without evidence. The owner’s Now tick means the
  necessity probe is Now, not that the binary backend itself is pre-approved.
- Correct known physics defects rather than reproduce them. Sundials becomes
  the default after complete lifecycle validation. Conflicting STT modes raise.
- Restart supports coordinate-aware v2 only. String expressions and raw-dof
  snapshot compatibility are not required.

## Priority 0 — Freeze evidence before source changes

### [x] P0.1 Master/pixi parity manifest

- Inventory every master/pixi test and example relevant to owner-Now features.
- Classify each as already covered, needs a DOLFINx translation, Later, Not now,
  or obsolete external execution with reusable reference data.
- Record the exact interface, physical invariant, tolerance and validation type.
- **Non-goal:** no source or tolerance changes.
- **Gate:** reviewed file-by-file manifest; `git diff --check`.

### [x] P0.2 Full-lane baseline — evidence complete, acceptance not green

- Run the aggregate verifier and `FINMAG_EXAMPLE_FULL=1` lane unchanged.
- FULL mode runs the fourteen fast entries plus three slow entries; retain the
  wrapper's unchanged 120–3600-second per-example timeouts defined in
  `examples/test_examples_dolfinx.py` rather than introducing a new global
  timeout. Baseline wrapper timeouts are expected evidence to record, not
  physics failures; only after evidence is frozen may timeout scaling/wrapper
  adjustment be proposed as a separate reviewed slice before an all-green FULL
  acceptance claim.
- Use a disposable clean worktree, run the editable-install task there first,
  and restore the editable install to the main worktree afterwards.
- Record failures/timeouts, standard-problem observables, post-run `git status
  --short`, and diffs for tracked `examples/magnetic_grain/mz.png` and
  `examples/std_prob_3/doc_table.rst` before fixing anything.
- **Non-goal:** no opportunistic fixes in the baseline run.
- **Gate:** reproducible commands, logs and clean worktree.
- **Recorded:** aggregate exit 0/all 32 green; clean FULL exit 1 (12 passed,
  5 failed). See `master-pixi-parity-manifest.md` for logs, exact versions and
  the failure split. This records baseline evidence only; it does not accept
  wrapper timeouts as physics failures or make FULL acceptance green.

## Priority 1 — Trustworthy simulator lifecycle

### [x] P1.1 Sundials reset and reinitialisation

- Remove the SciPy-only `.ode` assumption through the smallest backend-neutral
  lifecycle change.
- Verify existing tolerance/scheduling behavior before modifying it.
- **Completed:** `3f4ed4ea` makes reset backend-neutral: `Simulation` passes
  `t0` through the driver factory, preserves SciPy positional factory slots,
  and has a trajectory-continuity regression. Focused result: 76 passed,
  2 skipped. Aggregate: 32 green steps.

### [x] P1.2 Restart integrity on both backends

- Correct saved backend provenance.
- Compare uninterrupted and checkpoint/restart trajectories through the same
  final time on SciPy and Sundials.
- Retain v2-only rejection of legacy raw-dof archives.
- **Completed:** `17f24413` archives truthful `sim.integrator_backend`, keeps
  writable `sim.driver` in sync, retains v2-only restart, and adds immediate
  plus uninterrupted-vs-restarted trajectory checks on SciPy and Sundials.
  Metadata and mesh-mismatch semantics are unchanged. Focused results:
  restart 27, Simulation 33, Sundials 22, SciPy 21 passed/2 skipped, LLG 16;
  aggregate: 32 green steps.

### [x] P1.3 Restore Sundials as public default

- **Completed in `81fab481`:** `Simulation` and `sim_with` now default to
  native Sundials, matching legacy. The default is witnessed constructing and
  advancing the native `SundialsIntegrator`; the core smoke records
  `"integrator_backend": "sundials"` at `t=1e-12`.
- **SciPy remains supported:** `integrator_backend="scipy"` is a fully
  supported explicit opt-in and remains the always-available driver when the
  native extension is absent.
- **Evidence:** Simulation 37 passed; Sundials 22 passed; SciPy 21
  passed/2 skipped; restart/output 27 passed; cubic anisotropy 25 passed;
  time-Zeeman 28 passed; varying parameters 33 passed; fast examples 14
  passed/3 skipped. The final clean main-worktree `dev/bin/verify-dolfinx-m5`
  run on `81fab481` exited 0 with all 32 steps green; its fast lane was 14
  passed/3 skipped in 246.36s
  (log `/tmp/finmag-p1-default-m5.log`). The P0.2 FULL baseline remains
  blocked (12 passed, 5 failed) and was not rerun or reclassified by this slice.

## Priority 2 — Everyday API and approved physics corrections

### [x] P2.1 Core import boundary

- Make `finmag.example`/barmini and selected everyday conveniences import.
- Deferred normal-mode families must fail explicitly by feature name.
- **Non-goal:** no normal-mode implementation.
- **Completed in `626dfcc8`:** `finmag.example` (`bar`, `barmini`, `nanowire`)
  and `finmag.set_logging_level` work in the DOLFINx environment with no legacy
  `dolfin` installed. `bar`/`barmini`/`nanowire` build their box meshes with
  `dolfinx.mesh.create_box`, preserving every signature, dimension,
  discretisation and `sim_with` keyword; `set_logging_level` moves into a new
  stdlib-only `finmag.util.logging_helpers` re-exported from
  `finmag.util.helpers`; and `NormalModeSimulation`,
  `normal_mode_simulation`, `example.sphere_inside_airbox` and
  `example.normal_modes` now raise a curated `NotImplementedError` naming the
  deferred feature instead of leaking `ModuleNotFoundError: No module named
  'dolfin'`. No normal-mode implementation was added.
- **Evidence:** RED was 10 failing tests before the source change (9 in
  `test_example_dolfinx.py`, 1 in `test_import_boundary.py`), independently
  reproduced by the reviewer from a reconstructed pre-change tree; every
  failure bottomed out in a genuine missing-`dolfin` import. Focused gate
  `dolfinx-src-import-pytest` went from 9 passed/2 skipped to 19 passed/3
  skipped in 23.65s; neighbours `dolfinx-src-simulation-pytest` 37 passed and
  `dolfinx-src-deferred-pytest` 1 passed/4 skipped. In the legacy FEniCS pixi
  environment, `finmag.set_logging_level('DEBUG')` still leaves
  `df.parameters['reorder_dofs_serial']` `False` with logger level 10, and
  `finmag.util.helpers.set_logging_level is finmag.set_logging_level`. The
  pre-fix aggregate `dev/bin/verify-dolfinx-m5` exited 0 with all 32 steps
  green, fast examples 14 passed/3 skipped in 271.94s and the core smoke
  re-witnessing `{"integrator_backend": "sundials", ..., "t": 1e-12}`
  (`/tmp/finmag-p21-m5.log`); the post-review-fix aggregate on `626dfcc8`
  again ran all 32 steps green, with fast examples 14 passed/3 skipped in
  238.28s and the core smoke re-witnessing `{"integrator_backend": "sundials",
  ..., "t": 1e-12}`. That run executed while these five documentation files
  were mid-edit, so the wrapper's own tracked-file-cleanliness guard tripped
  and forced its exit status to 1 -- a false alarm from concurrent doc edits
  (no `src/` or `pixi.toml` file changed during the run), not a source
  regression (log `/tmp/finmag-p21-m5-final.log`). The owed clean-tree run was
  then executed on the committed, fully clean worktree and **exited 0 with all
  32 steps green**: focused import gate 19 passed/3 skipped in 21.16s, fast
  examples 14 passed/3 skipped in 252.11s, core smoke
  `{"integrator_backend": "sundials", ..., "t": 1e-12}`, and the
  tracked-file-cleanliness guard silent (log `/tmp/finmag-p21-m5-clean.log`).
  This is the authoritative P2.1 aggregate result. It does not rerun or
  reclassify the frozen FULL lane. [Claude Opus 4.8]
- **Review:** APPROVE WITH FOLLOW-UPS. Two findings were acted on before the
  commit was amended (`82967eb7` → `626dfcc8`): flipping `set_logging_level` to
  `requires_legacy_dolfin=False` silently dropped the `_prepare_legacy_dolfin()`
  dof-ordering side effect in the legacy lane (fixed by keeping the flag `True`
  while leaving the name out of `_LEGACY_ONLY_FEATURES`, the same treatment
  `example` gets, and verified empirically as above); and the
  `finmag.util.helpers` re-export was covered only by a test that skips in every
  materialised environment (fixed by folding `test_example_dolfinx.py` into the
  legacy `src-import-pytest` task as well).
- **Stated limits:** the guarded legacy-resolution path is forward-looking
  insurance, not a contract exercised today — importing `finmag.example` already
  required `dolfinx` before this slice (via `sphere_inside_airbox` →
  `finmag.field`), so no materialised environment can exercise it; it is not
  verified. `dolfinx.mesh.create_box` matches legacy `df.BoxMesh` exactly on
  vertex count and on 6 tetrahedra per cuboid cell, but the intra-cuboid
  diagonal orientation convention is unverified without legacy dolfin
  installed; any difference would be a permutation of the 6 tets within each
  cuboid, a discretisation-error-level difference, not a resolution or physics
  change. This slice did not rerun the FULL example lane
  (`FINMAG_EXAMPLE_FULL=1`, recorded baseline 12 passed/5 failed) and makes no
  claim about it.
- **Deferred follow-ups:** `from finmag.example.normal_modes import disk`
  (`src/finmag/sim/sim_test.py:1506`) is a real submodule import, not attribute
  access, so it still yields a raw `ModuleNotFoundError` in the DOLFINx
  environment; it belongs with the normal-modes port slice. `nanowire`'s `name`
  parameter is accepted but never forwarded to `sim_with`, so the simulation is
  always `unnamed` — faithful to master `b5015c5a`, a latent upstream bug and
  its own slice if the owner wants it corrected. `finmag.util.helpers` as a
  whole still imports legacy `dolfin` at module scope and remains unimportable
  in the DOLFINx environment; its full port is a separate slice.

### [x] P2.2 `sim_with` MacroGeometry

- Wire nx/ny/spacing arguments to the existing dense-FK MacroGeometry path.
- **Non-goals:** no Treecode selector, GCR or Demag2D work.
- **Completed in `2cf4647f`** (witness corrections `f97d4088`/`571df59f`,
  guard hardening `899d3906`): `sim_with(nx=, ny=, spacing_x=, spacing_y=)`
  forwards its four arguments to `MacroGeometry(nx=nx, ny=ny, dx=spacing_x,
  dy=spacing_y)` on the already-ported dense-FK path, matching the legacy
  contract (`git show b5015c5a:src/finmag/sim/sim.py`, lines 1443-1447)
  exactly: no `unit_length` scaling, and `spacing_*` is the tile PITCH
  (centre-to-centre translation of the image lattice) in mesh coordinate
  units, not a gap. A new guard, `_reject_touching_macro_geometry`, refuses by
  name any pitch at or below the mesh extent on an active axis (`pitch <=
  extent`) -- both the legacy exactly-touching default and the overlapping
  case -- because the ported periodic BEM double-counts the solid angle at
  coincident/interpenetrating tile boundary nodes.
- **P2.2a false-positive correction (`f97d4088`, completed by `571df59f`):**
  three PBC-path tests were counting a demonstrably broken configuration (the
  touching `MacroGeometry` default, pitch == mesh extent) as passing physics
  evidence. `test_pbc_out_of_plane_thin_film_analytic_limit` asserted `|Nz -
  1| < 1e-3` on a 40x40x2 slab tiled with that default; the configuration
  produces a periodic BEM with non-finite entries, the phi_2 Krylov solve
  fails with `KSP_DIVERGED_NANORINF`, phi_2 stays identically zero, and `H =
  -grad(phi_1) = -M` exactly -- which trivially satisfies `|Nz - 1| < 1e-3`
  (measured 2.085e-10) while the same configuration also reports the
  unphysical `Nx = 1.000000` in-plane. `test_pbc_image_sum_converges_1d` and
  `test_pbc_end_to_end_runs` used the same touching default and were corrected
  in the follow-up commit, which also corrected a stale "plateau near
  -0.1007" comment that was itself a broken-path number. All three now use a
  non-coincident pitch `extent * (1 + 1e-6)`, assert BEM finiteness and the
  row-sum identity (`sum_j B_ij == -1`) as explicit preconditions, and (where
  applicable) a monotonic approach to the analytic limit; the touching
  expectation is retained as a strict `xfail`
  (`test_pbc_out_of_plane_thin_film_analytic_limit_touching_tiles`), and the
  underlying defect is pinned directly by
  `test_pbc_coincident_tile_spacing_produces_a_non_finite_bem`.
- **Evidence:** cross-geometry equivalence (the legacy `demag_pbc_test.py`
  acceptance check) -- a 3x-tiled 20nm cube (`nx=3, spacing_x=20.001`)
  reproduces a directly-meshed 60x20x20nm bar to 0.08% in Hx (in-plane) and
  0.011% in Hz (out-of-plane), against the legacy bounds of 1%/2%; thin-film
  analytic limits in both directions at the gapped pitch (`Nz`: 0.709851 ->
  0.983976 -> 0.990758; `Nx`: 0.054464 -> 0.007920 -> 0.004588, for nx=ny =
  1, 3, 5), with monotonic approach plus BEM-finiteness/row-sum
  preconditions -- independently reproduced by review, which confirmed the
  gap-limit is continuous from the right (`Nz -> ~0.9908` as gap -> 0+), so
  the 1e-6 gap is real physics, not an artefact; `nx=1, ny=1` reduces to plain
  FK demag to 1.005e-15 relative; the corrected gapped 1D image-sum sequence
  (nx = 1, 3, 5, 9): -0.3337, -0.0638, -0.0245, -0.0078, decaying toward 0
  (replacing the broken-path "plateau near -0.1007"). Focused gates after all
  four commits: `dolfinx-src-simulation-pytest` 44 passed (was 37),
  `dolfinx-src-treecode-pytest` 22 passed/1 xfailed (was 20 passed),
  `dolfinx-src-demag-pytest` 18 passed (unchanged), `dolfinx-src-import-pytest`
  19 passed/3 skipped (unchanged).
- **Review:** APPROVE WITH FOLLOW-UPS. The touching-case diagnosis, the
  honesty of the replacement witnesses, the cross-geometry witness, exactness,
  mutation-resistance and scope were all independently verified. Finding 1 (a
  second touching-default witness, in `test_pbc_image_sum_converges_1d` /
  `test_pbc_end_to_end_runs`) and finding 2 (the guard rejected only
  exact-touching, not overlapping, pitches) were acted on in `571df59f` and
  `899d3906`. Finding 3 (the guard over-rejects a vanishingly small ~5e-10 gap
  band that would technically work) is deliberately left as benign
  conservatism -- the error message steers users to `extent * (1 + 1e-6)`.
- **New divergence recorded:** D17 in `acceptance-register.md` -- the by-name
  refusal of `pitch <= extent` is a deliberate divergence from the legacy
  default (which computed touching/overlapping tiles); disposition *pending
  owner decision*.
- **Non-goal confirmed:** no Treecode factory selector, GCR or Demag2D work;
  no demag-algorithm change. The coincident-node BEM defect itself is NOT
  fixed -- it is refused by name and pinned by a strict `xfail` plus a direct
  regression test, which remains a demag-algorithm change out of scope here.
  [Claude Sonnet 5]

### [x] P2.3 Callable pin masks

- Restore coordinate-to-dof callable pin selection without changing indexed
  pins or adding MPI stepping.
- **Completed in `df5fc1a1`:** `Simulation.__set_pins` previously raised a
  by-name `_deferred` `NotImplementedError` for a callable pin mask; it now
  resolves the callable to node indices exactly as legacy did (`git show
  b5015c5a:src/finmag/sim/sim.py` lines ~906-918): per-point over the
  owned-node coordinate array `self.llg._m_field.coords_and_values()[0]`, in
  RAW MESH UNITS (`unit_length` is NOT applied), a truthy return marks that
  node pinned, and the resulting index is the position in that same
  coordinate-ordered `xxx` array that `LLG._pins` consumes. Indexed pins,
  `LLG.set_pins`, MPI stepping and `field.py` are all unchanged; the
  serial-only pin contract stays. Coordinate mapping was directly verified on
  a `[0,30]x[0,10]x[0,10]` box (`unit_length=1e-9`): a "pin z<=zmin" callable
  resolved to `sim.llg.pins == [0,1,2,5,8,9,12,13]`, exactly the z==0 face;
  after `run_until(2e-12)` every pinned node's magnetisation delta measured
  0.0 exactly, while the unpinned nodes' max delta measured
  0.03844512461343788.
- **Evidence:** RED first -- the 4 new callable tests failed before the
  source change with the by-name `_deferred` `NotImplementedError`; the
  indexed-pin regression test (`test_indexed_pins_unchanged_by_callable_support`)
  passed both before and after, confirming no change to the existing index
  path. Focused gate `dolfinx-src-simulation-pytest` went from 44 passed to
  49 passed (5 new: intended-sites-and-holds-them, mesh-units-not-metres,
  selecting-nothing, callable-and-index-list-equivalent, plus the indexed-pin
  regression guard). Neighbour `dolfinx-src-llg-pytest` 16 passed, unchanged.
  The clean-tree aggregate `dev/bin/verify-dolfinx-m5` on the committed
  worktree **exited 0 with all 32 steps green**: simulation gate 49 passed,
  fast examples 14 passed/3 skipped in 216.63s, core smoke
  `{"integrator_backend": "sundials", "m_average": [0.9802592114540473,
  0.17662000907877634, 0.08889756808631089], "max_unit_norm_deviation":
  2.763425760221594e-06, "t": 1e-12, "t_target": 1e-12}`, tracked-file
  cleanliness guard silent (log `/tmp/finmag-p23-m5-clean.log`). This slice
  did not rerun the FULL example lane (`FINMAG_EXAMPLE_FULL=1`, recorded
  baseline 12 passed/5 failed) and makes no claim about it. [Claude Sonnet 5]
- **Review:** APPROVE, nothing blocking. The reviewer independently
  confirmed the coordinate->index mapping two ways: structurally (both
  `_pins` indexing and `coords_and_values` use the identical
  `_owned_vertex_to_dof` permutation) and empirically with an asymmetric
  single-corner selection -- the callable selected index 3, and exactly
  node 3 at (30,10,10) was frozen while the other 15 nodes moved (a
  permutation error would have frozen a different physical node instead).
  Units (mesh, not metres), the per-point contract, truthiness, the
  serial-only guard and the indexed-pin regression were all confirmed. No
  follow-ups.

### [x] P2.4 Correct discrete-time Zeeman energy

- Rebuild/update the energy consistently with the field after interval changes.
- Preserve an explicit regression showing the legacy stale-energy behavior.
- **Completed in `ff906f11` (register D3):** `DiscreteTimeZeeman.update()`
  previously rebound `self.H` to a brand-new `Field` on each interval crossing,
  bypassing `set_value()`; the cached UFL energy form `self.E` (built once in
  `setup()` against the original `H` `Function`) kept assembling the
  discarded setup-time field forever, so `compute_energy()` froze at
  `E(H(t=0))` -- measured a permanent freeze at `-8.042477193189932e-23` --
  while `compute_field()`/`energy_density()` stayed current: 16.7% relative
  error at the first crossing, 50% at t=1ns, unbounded in general (100% wrong
  when `H(0)=0`, wrong sign when `m.H(t)` flips). The fix routes the interval
  refresh through `self.set_value(self.field_function(t))`, exactly as the
  base `TimeZeeman.update` does: `self.H` is written in place and `self.E` is
  re-formed against it. Field values are provably unchanged by the fix
  (measured max|difference| = 0.0 in `compute_field()`/`average_field()`
  dof arrays between the old rebind path and the new `set_value` path, for
  both a constant-vector and a nested spatially-varying callable contract),
  so dynamics, effective field and every `.ndt` trajectory stay bit-identical;
  only `compute_energy()` is corrected. The legacy stale value is preserved as
  an explicit divergence pin (`_D3_LEGACY_STALE_ENERGY =
  -8.042477193189932e-23`) rather than silently dropped.
- **Evidence:** focused gate `dolfinx-src-timezeeman-pytest` went from 28
  passed to 32 passed (the stale-energy quirk pin was replaced by the D3
  divergence pin plus 5 new tests: corrected analytic energy tracking across
  interval updates, the field-invariance guard on both input contracts, a
  switch-off energy guard, and an explicit analytic-energy assertion on the
  continuous `TimeZeeman` path). Neighbours: `dolfinx-src-energies-pytest` 45
  passed; `dolfinx-src-simulation-pytest` 37 passed.
  `test_oracle_discrete_time_zeeman_sequence_matches_legacy` passes
  unchanged -- the committed fixture `timezeeman_oracle.json` does NOT
  numerically discriminate this defect (its correct energies are ~1e-37 J
  against `atol=1e-18`), so the fix passes it unchanged; the real numerical
  record of the legacy value is now the D3 divergence pin. The clean-tree
  aggregate `dev/bin/verify-dolfinx-m5` on the committed worktree **exited 0
  with all 32 steps green**: focused import gate 19 passed/3 skipped in
  20.87s, timezeeman folded in at 32 passed, fast examples 14 passed/3
  skipped in 207.38s, core smoke
  `{"integrator_backend": "sundials", "m_average": [0.9802592114540473,
  0.17662000907877634, 0.08889756808631089], "max_unit_norm_deviation":
  2.763425760221594e-06, "t": 1e-12, "t_target": 1e-12}`, tracked-file
  cleanliness guard silent (log `/tmp/finmag-p24-m5-clean.log`). This slice
  did not rerun the FULL example lane (`FINMAG_EXAMPLE_FULL=1`, recorded
  baseline 12 passed/5 failed) and makes no claim about it. [Claude Sonnet 5]
- **Review:** APPROVE, nothing blocking. The reviewer independently
  reproduced the multi-crossing and sign-flip energy tracking (rel err
  ≤2e-14, energy correctly flips sign); the field bit-invariance (max|
  difference| = 0.0 on both surfaces); the analytic energy values; that the
  divergence pin hard-pins the real legacy value and that `E_now/E_legacy ==
  2.0` is a robust structural fact (energy linear in H, fixture field doubles
  t=0 -> 1e-9), not a coincidence; and that re-forming `self.E` costs ~7.7us
  (~0.5% of an update), negligible. The only follow-up noted was docs
  reconciliation, discharged by this same commit's companion documentation
  update.

### [ ] P2.5 Correct hysteresis stage relaxation

- Ensure every applied-field stage independently relaxes.
- Test switching/loop physics, not merely result lengths.
- Document why corrected output differs from master.

### [x] P2.6 Reject conflicting STT modes

- Keep each mode independently working.
- Raise a clear error when configuration would enable both modes.
- **Completed in `de518888` (register D11):** `LLG.use_slonczewski` and
  `use_zhangli` each gain a guard, evaluated BEFORE either method mutates any
  flag, that raises `ValueError` if the OTHER mode's `do_*` flag is already
  set (`use_slonczewski` checks `self.do_zhangli`; `use_zhangli` checks
  `self.do_slonczewski`). Each guard keys ONLY on the sibling flag, so
  re-tuning the SAME mode (a second `use_slonczewski` call to change J/P, or
  a second `use_zhangli` call to change P/beta) and disable-then-switch (flip
  the active `do_*` flag off, then configure the other mode) both remain
  allowed -- exactly the previous last-call-wins convenience, minus the
  silent cross-mode clobber. The `ValueError` messages name both
  "Slonczewski" and "Zhang-Li" and tell the caller to disable one before
  enabling the other, following the in-file D10 pin-range `ValueError`
  precedent. `Simulation.set_stt`/`set_zhangli` inherit the guard by routing
  through these setters, with no duplicated logic in `sim.py`. STT physics
  and the `if do_slonczewski ... elif do_zhangli` dispatch in `llg.py` are
  unchanged.
- **Evidence:** RED first -- the 2 conflict tests failed before the source
  change with "DID NOT RAISE ValueError"; the 5 allowed-behaviour tests
  (same-mode reconfigure for each mode, disable-then-switch, each mode alone
  activating) already passed pre-change, confirming the guard does not
  over-reach. Focused gate `dolfinx-src-stt-pytest` went from 14 passed to
  21 passed (+7 new tests). Neighbours: `dolfinx-src-llg-pytest` 16 passed
  and `dolfinx-src-simulation-pytest` 49 passed, both unchanged. Each mode
  alone still produces a nonzero STT dm/dt contribution (`‖dmdt‖` ~=
  2.46e10 for Slonczewski, ~3.5e10 for Zhang-Li). [Claude Sonnet 5]
- **Review:** APPROVE WITH FOLLOW-UPS, nothing blocking. The reviewer
  verified the guard runs before any mutation in both directions (a rejected
  call leaves the first mode's full configuration intact); same-mode
  reconfigure is preserved for both modes; both call orders raise with
  messages naming both modes; each mode alone still works; STT physics and
  dispatch are untouched; the `Simulation`-layer paths inherit the guard;
  the tests assert real behaviour (flag state, not merely "raises"); the RED
  evidence reproduces; and the diff is scope-clean (`llg.py` +14 lines,
  `test_stt_dolfinx.py` +83 lines, nothing else touched). Two out-of-scope
  defects were recorded rather than fixed, as register row **D18**:
  `Simulation.toggle_stt(new_state)` (`src/finmag/sim/sim.py:888`) writes
  `self.llg.do_slonczewski` directly, bypassing these guarded setters, so
  `set_zhangli(...)` followed by `toggle_stt(True)` can still leave BOTH
  `do_slonczewski` and `do_zhangli` True -- the dispatch then silently
  prefers Slonczewski, the very silent precedence D11 set out to close,
  still reachable through this one path; separately, `toggle_stt(False)`
  branches on `if new_state:` / `else:`, so a falsy explicit `False` falls
  into the flip branch instead of forcing the flag off, and so does not
  reliably disable. D11's approved scope was the `use_*` configuration path
  only, so both are correctly left for a follow-up slice rather than folded
  into this one.

Each P2 slice gets a RED test, the narrow focused gate, closest legacy
invariants from P0.1, and the aggregate verifier. Do not combine these into one
public-interface commit.

## Priority 3 — Selected serial physics and Field completeness

1. Probe the exact legacy `Simulation(pbc=)` contract under `dev/dolfinx`.
2. Implement serial function-space PBC from the accepted probe; do not combine
   it with MacroGeometry or MPI.
3. Port `Field.cross`, `Field.dot` and scalar coercion.
4. Port `Field.from_generic_vector` separately with ordering and MPI-ownership
   tests.
5. Port spatially varying cubic axes with oracle and analytic witnesses; retain
   the owner-approved correct K2 physics.

## Priority 4 — Selected I/O and convenience surface

Implement as separate reviewable slices:

1. HDF5 readback/round-trip with a DOLFINx-native documented format.
2. Region/submesh field output.
3. Magnetisation initialisers, one formula family per slice.
4. `Simulation.probe_field*`.
5. LLG `M`/`M_average` compatibility.
6. `length_scales` and `mesh_info`.
7. logging/instance/shutdown helpers needed by selected scripts.
8. backend-neutral NumPy/Matplotlib helpers, followed by an optional PyVista
   adapter with headless tests. Do not reproduce the mencoder wrapper.

## Priority 5 — Mesh and external-reference validation

### [ ] P5.1 Netgen necessity probe

- Identify a selected test/geometry that Gmsh and the `.geo` subset cannot
  represent or validate.
- If none exists, leave Netgen deferred. If one exists, propose a separate
  conversion design before source edits.

### [ ] P5.2 Comparison data

- Restore OOMMF, Nmag and Magpar comparisons separately using checked-in data.
- Compare by coordinates and physical invariants, not raw node ordering.
- Do not resurrect `nsim` merely to reproduce historical execution.

### [ ] P5.3 Legacy tests

- Port tests in small feature-family clusters driven by P0.1.
- Skip/xfail only after the positive replacement test is ported, or for a named
  Later/Not-now capability with a linked acceptance-register reason.
- Never create a single mechanical whole-suite rewrite.
- These feature-family slices may run in parallel with other SR1 work, but SR1
  is not complete until every selected owner-Now legacy-test family has a
  positive DOLFINx witness or an explicitly linked owner disposition.

## Priority 6 — Full scientific acceptance and SR1 handoff

1. Run the full-resolution examples after preceding fixes land.
2. Add mesh-matched oracle or convergence/quantitative trajectory evidence for
   standard problem 4; its broad switching window remains qualitative only.
3. Execute minimal and realistic demag/dynamics examples from a clean install.
4. Publish supported API, validation limits, Later and Not-now lists.
5. Declare SR1 only when all selected Now rows are working or explicitly moved
   by a new owner prioritisation decision, P0.2 is recorded, and the P5.3
   legacy-test-family gate is satisfied.

## Deferred beyond SR1

Normal modes/eigensolvers; ringdown and FFT/PSD; legacy NEB; a separate GNEB
PR; MPI stepping; DMI D2D; possible Treecode factory exposure; compiled
Equation performance; and movie-generation conveniences.

Thermal SLLG, LLB, nonlocal STT, Demag2D, GCR, FixedEnergyDW, specialist mesh
generators, nmesh conversion, old energy assembly algorithms, batch tooling and
Mercurial helpers are **Not now**, not permanently deleted. Revisit them during
later full-parity planning.

[Codex GPT-5]

[P2.1 updates: Claude Opus 4.8]

[P2.3 updates: Claude Sonnet 5]

[P2.4 updates: Claude Sonnet 5]

[P2.6 updates: Claude Sonnet 5]
