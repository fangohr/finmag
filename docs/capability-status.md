# Finmag DOLFINx Capability Status

**Source baseline audited:** 2026-07-23 at `dolfinx-parity` commit
`f1a1344c423e74687ddf820c1fafc056a6271fe1`

**2026-07-29 update (doc-restructure T1): header date corrected.** "Source
baseline audited" names the original P0 audit commit/date; individual rows
(e.g. C03/C04/C08/C09/C21) have been continuously updated through SR1 P4 and
the 2026-07-28 batch ratification — see each row's own evidence for its
current date. This file is not stale as of the 2026-07-23 header date alone.

**Final target:** the functionality and user-facing interface of original
`master` (`b5015c5a`), except for deviations explicitly accepted by the
repository owner.

This is the canonical current capability inventory. `docs/archive/HANDOVER.md`
was the short entry point (now retired; see `docs/README.md`);
`acceptance-register.md` is the sole owner-decision ledger. The plans,
`docs/archive/transition-notes.org`, and `docs/archive/porting_map.md`
preserve history and evidence but do not override this status table.

Maintenance rule: every implementation slice updates the affected row(s) in
this file in the same commit, naming the focused gate and baseline SHA/date in
its evidence. State changes belong here, not in a new status list. A slice may
touch multiple rows only when one source change genuinely crosses those
capability boundaries.

Status terms:

- **ported** — the stated surface works and has a current automated witness;
- **partial** — a useful subset works, but named compatibility or lifecycle
  gaps remain;
- **planned** — required for master parity but not implemented yet;
- **deferred** — intentionally scheduled after the first usable release;
- **decision-pending** — implementation depends on an owner decision in the
  acceptance register.

Validation terms distinguish **oracle** comparisons with frozen pixi commit
`ba9280934e188d7f3800e7b9865e70a9422f7687`, **analytic** physics checks,
**cross-method** agreement, **regression** tests, and **end-to-end** workflows.
“Oracle” is used only where the frozen code was actually run.

## Interim goal: serial simulator release candidate (SR1)

SR1 is the next useful stopping point. It is not full master parity. It is a
scientifically testable, deterministic, serial micromagnetic simulator with a
substantial legacy-compatible `Simulation` interface and an explicit list of
everything that is still unavailable.

SR1 is complete when all of the following are true:

- the documented pixi install, `import finmag`, `finmag.example.barmini()`, and
  a realistic demag-containing dynamics example work from a clean checkout;
- deterministic Exchange, DMI, uniaxial/cubic anisotropy, static/time-dependent
  applied fields, FK demag, treecode/MacroGeometry demag, spatially varying
  parameters, regions, and each local STT mode separately compose in serial
  simulations;
- `run_until`, `relax`, hysteresis, scheduling, NDT, field snapshots,
  VTK/XDMF write, and restart work through the documented lifecycle;
- SciPy and Sundials support construction, tolerance changes, advance,
  reset/reinitialisation, scheduling, save, and restart, or a backend is
  explicitly removed from the supported SR1 contract;
- physics is covered by oracle, analytic, cross-method, regression, and
  composed end-to-end tests appropriate to each capability;
- the full-resolution example lane (`FINMAG_EXAMPLE_FULL=1`), including
  muMAG standard problem 4, has a recorded green run as a qualitative workflow
  witness; behavioral parity needs tighter mesh-matched or convergence
  evidence. **At declaration this criterion is met 15/17 and explicitly
  qualified:** the two heaviest entries (`std_prob_4` full trace,
  `magnetic_grain` full physics) are deferred by owner decision (2026-07-28)
  to a re-run after `plans/2026-07-28-post-sr1-performance.md`. See C15;
- legacy public names needed by the supported workflow either work or raise a
  feature-specific error—never an incidental raw `dolfin` import failure;
- user documentation includes installation, a minimal simulation, a realistic
  simulation, validation limits, and a plainly visible “not implemented yet”
  section.
- every selected owner-Now legacy-test family has a positive DOLFINx witness or
  an explicitly linked owner disposition; P5.3 may execute in parallel with
  implementation slices but is an SR1 completion gate.

The owner approved correcting D3 (stale discrete-field energy) and D4
(hysteresis no-re-relax). D3 is implemented (`ff906f11`, SR1 P2.4). D4 is
discharged as a diagnosis correction (SR1 P2.5): the approved-fix
investigation found the re-relax defect NOT reproducible -- the code already
re-relaxes each stage -- so the owner directed a diagnosis correction (no
source change), not a mechanism rewrite; the on-axis oracle stall is
reclassified as a degenerate Stoner-Wohlfarth-saddle pin. D8's condition is
discharged: `81fab481` restores native Sundials as the public default after
the P1.1/P1.2 lifecycle and restart evidence. SciPy remains a fully supported
explicit opt-in.

## Current capability matrix

| ID | Capability | Status | Current evidence and boundary | SR1 |
|---|---|---|---|---|
| C01 | Install, native build, provenance, aggregate CI | **ported** | Editable pixi install with native BEM/CVODE/treecode builds and a green 33-gate verifier; `h5py` (`1a9df5eb`) was the only new SR1 runtime dependency, added for the `Field` HDF5 round-trip (see C04). See [SUPPORTED.md §1](SUPPORTED.md). | required, green |
| C02 | Package import and top-level compatibility exports | **partial** | Core imports and `finmag.example`/`set_logging_level` now work dolfin-free (`626dfcc8`); the submodule-import spelling and `finmag.util.helpers`-as-a-module gaps are the two known holes named in [SUPPORTED.md §7](SUPPORTED.md). | everyday exports are blockers; normal-mode exports later |
| C03 | Mesh generation | **partial** | Common Gmsh/OCC mesh generation replaces Netgen (with a documented `maxh` density-drift caveat) and mesh diagnostics (`mesh_info`, `mesh_size`, `length_scales`) are restored (`868f664d`); Netgen's binary backend, `nmesh_to_dolfin` and specialist generators remain unavailable. See [decisions.md §2](decisions.md) and [SUPPORTED.md §2.5](SUPPORTED.md). | common subset required; probe Netgen need before any port; specialist generators not now; mesh diagnostics now ported |
| C04 | `Field` data model and inspection | **partial** | Constants/callables/arrays, point probing, `cross`/`dot`/coercion (`c59f3438`), `from_generic_vector` (`ef92eb7d`) and HDF5 round-trip are ported, with one known minor gap left as-is (scalar-Field multiply/divide does not itself guard a mismatched-space same-node-count scramble, matching legacy); `__add__`, VTK/XDMF function readback and plotting remain blockers. See [SUPPORTED.md §2.3](SUPPORTED.md). | owner selected these missing operations for SR1; `cross`/`dot`/coercion, `from_generic_vector` and HDF5 round-trip now ported (`__add__` still deferred); plotting (see C12) and VTK/XDMF readback remain blockers |
| C05 | Spatial parameters and regions | **ported** | Varying `Ms`/`A`/`K`/`D`/axis/alpha and region energies/averages are oracle/regression-tested; region-restricted *field* output remains separate, unported I/O. See [SUPPORTED.md §2.4](SUPPORTED.md). | required, green |
| C06 | Deterministic energy interactions | **partial** | Exchange, Zeeman family, uniaxial/cubic anisotropy (including now-ported spatially varying cubic axes, `595f8335`), DMI and `ThinFilmDemag` are oracle/analytic/regression-tested; legacy matrix/project/direct methods and string-`Expression` axes are not required for SR1. See [SUPPORTED.md §2.4](SUPPORTED.md). | common interactions required, now including varying cubic axes; selected legacy surfaces deferred |
| C07 | Demagnetisation | **partial** | FK demag is oracle-validated and treecode/`MacroGeometry` is analytic/cross-method validated, with `nx`/`ny`/`spacing_*` wired through `sim_with`; the Treecode factory selector, GCR, `Demag2D` and touching/overlapping tiling (D17) remain unavailable or refused by name. See [SUPPORTED.md §2.4](SUPPORTED.md). | Treecode factory exposure later/review; D17 approved 2026-07-23 (refusal is the SR1 contract; coincident-node BEM kernel fix deferred) |
| C08 | Deterministic serial LLG and composed dynamics | **ported** | `Simulation`, `run_until`, `relax`, effective-field composition and callable/indexed pins (`df5fc1a1`) are analytic/oracle/regression-tested, and `LLG.M`/`M_average` now implement corrected, documented physics (`67e5ebe1`, D20) rather than the broken legacy versions. See [decisions.md §5.2](decisions.md) and [SUPPORTED.md §2.2](SUPPORTED.md). | required core green; callable pins now ported; `M`/`M_average` now ported with corrected physics (D20 ACCEPTed) |
| C09 | SciPy and native Sundials integration | **ported** | Both backends construct, advance, reset/reinitialise, schedule, save and restart, and native Sundials is again the public default (D8), with SciPy a fully supported explicit opt-in. See [decisions.md §3](decisions.md). | required, green; D8 discharged |
| C10 | Spin-transfer torque | **partial** | Slonczewski and Zhang-Li local STT modes run separately with oracle/analytic/cross-backend witnesses, and configuring both now raises `ValueError` naming both (D11, D18, closing the `toggle_stt` back-door); nonlocal `LLG_STT` remains unported (M5). See [SUPPORTED.md §2.5](SUPPORTED.md). | D11 and D18 both discharged for the local-mode config and `toggle_stt` paths; nonlocal not now |
| C11 | Time-dependent fields and hysteresis | **partial** | Time-dependent Zeeman, relaxation and hysteresis are exercised; the stale `DiscreteTimeZeeman` energy is fixed (D3) and the hysteresis "no-re-relax" concern is resolved as a diagnosis correction, not a defect (D4). See [decisions.md §5.1](decisions.md) and [SUPPORTED.md §2.5](SUPPORTED.md). | D4 discharged (no fix needed, diagnosis corrected); row stays partial on remaining coverage |
| C12 | Scheduler and output | **partial** | Scheduler, NDT, `.npy` snapshots, VTK/XDMF write, HDF5 round-trip (see C04), the per-region `.ndt` column (D21) and `get_field_as_dolfin_function`'s point-eval path (D26, fixed) are ported; region/submesh field output, VTK/XDMF readback, per-interaction `.ndt` columns (D27) and the `surface_3d` matplotlib-3.11 break remain blockers. See [SUPPORTED.md §2.5](SUPPORTED.md). | HDF5 round-trip, the per-region `.ndt` column (D21 ACCEPTed) and `get_field_as_dolfin_function`'s point-eval path (D26 fixed) are now ported; region/submesh field output, VTK/XDMF readback, the `surface_3d` mpl-3.11 fix and the PyVista adapter remain blockers/later; movie wrapper later/review |
| C13 | Restart | **partial** | Coordinate-aware restart v2 restores magnetisation/time on both backends and rejects v1; material/interaction metadata reapplication and full-state reconstruction are out of scope by design (D16). See [decisions.md §4](decisions.md). | metadata boundary remains |
| C14 | Point/topology utilities | **ported** | Point evaluation, `get_spherical`, skyrmion number/density are analytic/regression-tested, subject to the D22 outer-face-probe caveat. See [SUPPORTED.md §3.1](SUPPORTED.md). | required, green |
| C15 | Converted examples | **partial (declaration-qualified)** | 15 of 17 FULL-lane examples are witnessed green at full workload at SR1 declaration; the remaining two (`std_prob_4` full trace, `magnetic_grain` full physics) are deferred by owner decision to a re-run after the post-SR1 performance work. See [SUPPORTED.md §3.3](SUPPORTED.md). | declaration-qualified; completing the two deferred entries is named post-perf follow-up |
| C16 | Thermal SLLG and LLB | **deferred** | The legacy surface remains tied to unported modules/native code, retained for a later full-parity decision. See [decisions.md §8](decisions.md). | Not now; retain for later full-parity decision/work |
| C17 | Normal modes, eigensolvers, ringdown and FFT/PSD | **planned** | Top-level `NormalModeSimulation`/`normal_mode_simulation` raise a curated `NotImplementedError`, but the submodule spelling still reaches legacy `dolfin` directly (C02); required for full parity, deferred past SR1. See [decisions.md §8](decisions.md). | after SR1; required for full parity |
| C18 | Path methods | **deferred** | No public GNEB API/workflow exists on original `master`, so it is not an interface-parity requirement; legacy NEB variants remain unclassified pending inventory. See [decisions.md §8](decisions.md). | after SR1 |
| C19 | Parallel and periodic simulation surfaces | **partial** | Ownership-sensitive MPI probes and treecode periodic `MacroGeometry` work; general MPI stepping and serial function-space PBC are deferred, the latter genuinely blocked without `dolfinx_mpc` (D19). See [decisions.md §8](decisions.md). | serial `Simulation(pbc=)` deferred past SR1 (D19, blocked without `dolfinx_mpc`); MPI stepping later |
| C20 | OOMMF/Nmag/Magpar comparison workflows | **partial** | SR1 restored several checked-data Nmag/Magpar comparisons as live witnesses (exchange to `9e-8`, anisotropy to the D29-quantified ~8% mesh-drift residual); the dedicated OOMMF suite and all live reference runners remain unavailable. See [decisions.md §5.3](decisions.md) and [SUPPORTED.md §3.4](SUPPORTED.md). | selected checked-data comparisons largely restored (Nmag + Magpar witnessed, one OOMMF assertion); OOMMF comparison suite and live runners later/review |
| C21 | Legacy utilities and specialist workflows | **partial** | Common ported-workflow helpers are restored, including `probe_field`/`probe_field_along_line` (see C14), the vortex initialiser family, dolfin-free logging helpers and `Simulation`'s instance-teardown surface (D24, fixed); skyrmion initialisation and the unselected long tail of research-era surfaces remain later work. See [SUPPORTED.md §5](SUPPORTED.md). | selected checklist rows: probe_field, vortex init, logging helpers, HDF5, plotting, mesh diagnostics, instance-teardown now ported; skyrmion init and unselected long tail later/not now |
| C22 | Full public-interface parity | **partial** | A focused 45-surface audit corrected six parameter names/orderings but was not exhaustive; top-level exports, `sim_with`, backend lifecycle and the unported-module long tail remain open for a full exhaustive parity audit. See [archive/master-pixi-parity-manifest.md](archive/master-pixi-parity-manifest.md). | supported SR1 names required; exhaustive audit for full parity |

## Current high-priority failures

1. Substantially resolved in `626dfcc8` (P2.1): `finmag.set_logging_level` and
   `finmag.example` now work in the DOLFINx environment, and
   `finmag.NormalModeSimulation`, `finmag.normal_mode_simulation`,
   `example.sphere_inside_airbox` and `example.normal_modes` fail by feature
   name. What remains is the submodule-import spelling: `from
   finmag.example.normal_modes import disk` (`src/finmag/sim/sim_test.py:1506`)
   is a real submodule import, not attribute access, so it still raises raw
   `ModuleNotFoundError: No module named 'dolfin'`; it belongs with the
   normal-modes port slice. Separately, `finmag.util.helpers` as a whole still
   imports legacy `dolfin` at module scope and remains unimportable in the
   DOLFINx environment (only the `set_logging_level` re-export was moved out);
   its full port is its own slice.
2. Substantially resolved in `2cf4647f`/`571df59f`/`899d3906` (P2.2/P2.2a):
   `sim_with` now wires `nx`/`ny`/`spacing_x`/`spacing_y` through to the
   already-ported dense-FK `MacroGeometry` path instead of rejecting them.
   What remains: the Treecode factory selector (`demag_solver='Treecode'`) is
   still deferred by name at `sim_with`, and the touching/overlapping tiling
   (`pitch <= extent` on an active axis) is refused by name as a deliberate
   divergence from the legacy default, recorded as D17 in
   `acceptance-register.md` (**approved 2026-07-23** as the SR1 contract and
   **ratified 2026-07-28 (owner, batch): DEFER past SR1** -- the refusal
   stands; the coincident-node BEM kernel fix is a named full-parity backlog
   item, not a permanent exception).
3. **Superseded by the SR1 S1-S3 FULL-lane work.** The 2026-07-23 P0.2 baseline
   (three wrapper timeouts, a missing `save_m` scheduler keyword, a raw
   `dolfin` import) is now discharged: `save_m` is a registered scheduler
   shortcut (`f7517688`), the wrapper timeouts were rescaled to *measured*
   rates (`20a0a22c`, `71dfc064`, `c02809bf`), and 15 of the 17 entries are
   witnessed green at full workload. What remains open is the owner-deferred
   pair — `std_prob_4/test_std_prob_4.py` and `magnetic_grain/suess_2001.py`
   — scheduled for re-run after the post-SR1 performance plan. See C15.
4. **Performance (register P1).** The port re-does `fem.form(...)` +
   `assemble_vector(...)` on every field evaluation
   (`src/finmag/energies/energy_base.py:183`), where legacy's default
   `box-matrix-petsc` assembled the field operator once and then applied it
   (`b5015c5a:src/finmag/energies/energy_base.py:214-222`); the numpy RHS vs
   the compiled `Equation` backend (M3) compounds it. Root cause identified
   2026-07-28; the fix plan is
   `plans/2026-07-28-post-sr1-performance.md`. This is why the two FULL-lane
   entries above are sequenced after it.

These findings are status, not permission to change source. Each source fix
still follows the per-slice protocol and receives an independent review.

## Bounded SR1 work slices

| Slice | Scope and non-goals | Required evidence |
|---|---|---|
| SR1-L1 | **Completed in `3f4ed4ea` and `17f24413`:** backend-neutral reset/reinitialisation, truthful restart provenance and both-backend restart integrity. D8/public default and D16 metadata scope were not changed. | P1.1 focused 76 passed/2 skipped; P1.2 restart 27, Simulation 33, Sundials 22, SciPy 21 passed/2 skipped, LLG 16; aggregate 32 green |
| P1.3 | **Completed in `81fab481`:** restore native Sundials as the public `Simulation`/`sim_with` default; preserve explicit SciPy support. No scheduler, restart-format or physics change. | Simulation 37; Sundials 22; SciPy 21 passed/2 skipped; restart 27; core smoke witness `sundials` at `t=1e-12`; final clean main aggregate 32/32 green, examples 14 passed/3 skipped in 246.36s (`/tmp/finmag-p1-default-m5.log`) |
| SR1-A1 | **Completed in `626dfcc8` (P2.1):** `bar`/`barmini`/`nanowire` build their meshes with `dolfinx.mesh.create_box` (signatures, dimensions, discretisations and every `sim_with` keyword unchanged); `set_logging_level` moves to a stdlib-only `finmag.util.logging_helpers` re-exported from `finmag.util.helpers`; the unported normal-mode and airbox surfaces raise a curated `NotImplementedError`. Normal modes were not ported, and the rest of `finmag.util.helpers` was not ported. | RED: 10 tests failed before the source change (9 in `test_example_dolfinx.py`, 1 in `test_import_boundary.py`), independently reproduced by the reviewer from a reconstructed pre-change tree; every failure bottomed out in a genuine missing-`dolfin` import. Focused `dolfinx-src-import-pytest` 19 passed/3 skipped in 23.65s (from 9 passed/2 skipped); neighbours `dolfinx-src-simulation-pytest` 37 passed and `dolfinx-src-deferred-pytest` 1 passed/4 skipped. Legacy-lane check of the review fix: after `finmag.set_logging_level('DEBUG')`, `df.parameters['reorder_dofs_serial']` is `False`, logger level 10, and `finmag.util.helpers.set_logging_level is finmag.set_logging_level`. Pre-fix aggregate `dev/bin/verify-dolfinx-m5` exit 0, all 32 steps green, fast examples 14 passed/3 skipped in 271.94s, core smoke `{"integrator_backend": "sundials", ..., "t": 1e-12}` (`/tmp/finmag-p21-m5.log`); post-review-fix aggregate on `626dfcc8` again ran all 32 steps green (fast examples 14 passed/3 skipped in 238.28s, core smoke `{"integrator_backend": "sundials", ..., "t": 1e-12}`), but this run executed while these five documentation files were mid-edit, so the wrapper's own tracked-file-cleanliness guard tripped and forced its exit status to 1 -- a false alarm from concurrent doc edits (no `src/` or `pixi.toml` file changed during the run), not a source regression (`/tmp/finmag-p21-m5-final.log`). The authoritative clean-tree aggregate then ran on the committed, fully clean worktree and exited 0 with all 32 steps green: import gate 19 passed/3 skipped in 21.16s, fast examples 14 passed/3 skipped in 252.11s, core smoke `{"integrator_backend": "sundials", ..., "t": 1e-12}`, cleanliness guard silent (`/tmp/finmag-p21-m5-clean.log`) |
| SR1-A2 | **Completed in `2cf4647f`, `571df59f`, `899d3906` (P2.2/P2.2a):** wires `nx`/`ny`/`spacing_x`/`spacing_y` through `sim_with` to the already-ported dense-FK `MacroGeometry` path -- pitch semantics, mesh coordinate units, no `unit_length` scaling, matching the legacy contract exactly. A new guard refuses the touching-or-overlapping tiling (`pitch <= extent` on an active axis) by name (D17); the underlying coincident-node BEM defect is refused and pinned, not fixed. No Treecode factory exposure or demag-algorithm change. | Two witness corrections first removed a false-positive PBC path (`f97d4088`, completed `571df59f`): the touching-default thin-film/1D-image-sum tests were passing because a broken solve silently returns `H=-M`, not because the physics converged. Focused gates after all four commits: `dolfinx-src-simulation-pytest` 44 passed (was 37); `dolfinx-src-treecode-pytest` 22 passed/1 xfailed (was 20 passed); `dolfinx-src-demag-pytest` 18 passed (unchanged); `dolfinx-src-import-pytest` 19 passed/3 skipped (unchanged). Cross-geometry witness: a 3x-tiled 20nm cube reproduces a directly-meshed 60x20x20nm bar to 0.08% in Hx / 0.011% in Hz against legacy bounds 1%/2%. Review: APPROVE WITH FOLLOW-UPS |
| P2.3 | **Completed in `df5fc1a1`:** restores callable pin masks in `Simulation.__set_pins`, resolving the callable in the sim layer instead of raising the by-name `_deferred` `NotImplementedError`. Per-point over the owned-node coordinate array (`self.llg._m_field.coords_and_values()[0]`), raw mesh units (no `unit_length` scaling), truthy return marks the node pinned, index = position in that same coordinate-ordered `xxx` array that `LLG._pins` consumes -- matching legacy (`git show b5015c5a:src/finmag/sim/sim.py` ~906-918) exactly. Indexed pins, `LLG.set_pins`, MPI stepping and `field.py` are unchanged. | RED: the 4 callable tests failed before the source change with the by-name `_deferred` `NotImplementedError`; the indexed-pin regression test passed both before and after. Gate `dolfinx-src-simulation-pytest` 44 -> 49 passed (5 new); neighbour `dolfinx-src-llg-pytest` 16 passed, unchanged. Coordinate mapping confirmed on a `[0,30]x[0,10]x[0,10]` box (`unit_length=1e-9`): a "pin z<=zmin" callable resolved to `sim.llg.pins == [0,1,2,5,8,9,12,13]` (exactly the z==0 face); after `run_until(2e-12)` pinned-node delta-m = 0.0 exactly, unpinned max delta-m = 0.03844512461343788. Review independently confirmed the coordinate->index mapping two ways -- structurally (both `_pins` indexing and `coords_and_values` use the identical `_owned_vertex_to_dof` permutation) and empirically with an asymmetric single-corner selection (callable selected index 3; exactly node 3 at (30,10,10) was frozen while the other 15 moved -- a permutation error would have frozen a different physical node). Clean-tree aggregate `dev/bin/verify-dolfinx-m5` exit 0, all 32 steps green; simulation gate 49 passed; examples 14 passed/3 skipped in 216.63s; core smoke `{"integrator_backend": "sundials", "m_average": [0.9802592114540473, 0.17662000907877634, 0.08889756808631089], "max_unit_norm_deviation": 2.763425760221594e-06, "t": 1e-12, "t_target": 1e-12}`; cleanliness guard silent (`/tmp/finmag-p23-m5-clean.log`). Review: APPROVE, nothing blocking |
| P2.4 | **Completed in `ff906f11` (register D3):** `DiscreteTimeZeeman.update()` routes its interval refresh through `set_value()` instead of rebinding `self.H`, so the cached energy form `self.E` is re-formed and `compute_energy()` tracks the current field instead of freezing at the setup-time value. Field values are unchanged (max\|difference\| = 0.0). D4/hysteresis re-relax is untouched (separate slice, P2.5). | Focused `dolfinx-src-timezeeman-pytest` 28 -> 32 passed (1 staleness test replaced by a D3 divergence pin + 5 new); neighbours `dolfinx-src-energies-pytest` 45 passed, `dolfinx-src-simulation-pytest` 37 passed; oracle case `test_oracle_discrete_time_zeeman_sequence_matches_legacy` passes unchanged (fixture does not numerically discriminate the defect); field bit-invariance witnessed for both a constant vector and a spatially varying callable; clean aggregate `dev/bin/verify-dolfinx-m5` exit 0, all 32 steps green (`/tmp/finmag-p24-m5-clean.log`) |
| P2.6 | **Completed in `de518888` (register D11):** `LLG.use_slonczewski` and `use_zhangli` each gain a guard, checked before any flag mutation, that raises `ValueError` naming both modes if the OTHER mode's `do_*` flag is already set; re-tuning the SAME mode and disable-then-switch remain allowed since the guard keys only on the sibling flag. `Simulation.set_stt`/`set_zhangli` inherit the guard by routing through these setters. STT physics and the `if do_slonczewski ... elif do_zhangli` dispatch are unchanged. | RED: the 2 conflict tests failed before the change with "DID NOT RAISE ValueError"; the 5 allowed-behaviour tests (same-mode reconfigure x2, disable-then-switch, each mode alone x2) passed pre-change, showing the guard does not over-reach. Focused `dolfinx-src-stt-pytest` 14 -> 21 passed (+7); neighbours `dolfinx-src-llg-pytest` 16 passed and `dolfinx-src-simulation-pytest` 49 passed, both unchanged. Each mode alone still produces nonzero STT dm/dt (‖dmdt‖ ~= 2.46e10 Slonczewski, ~3.5e10 Zhang-Li). Review: APPROVE WITH FOLLOW-UPS, nothing blocking -- the reviewer confirmed the guard runs before any mutation (a rejected call leaves the first mode's config fully intact in both directions), same-mode reconfigure and disable-then-switch, both-order error messages naming both modes, and scope clean (`llg.py` +14, `test_stt_dolfinx.py` +83). Two out-of-scope defects recorded as D18, not fixed: `Simulation.toggle_stt(True)` writes `do_slonczewski` directly, bypassing this guard, so both flags can end up True and the dispatch silently prefers Slonczewski; `toggle_stt(False)` flips rather than force-disables |
| SR1-V1 | Run `FINMAG_EXAMPLE_FULL=1`; diagnose failures and make only separately reviewed minimal fixes. Treat std_prob4's broad crossing window as a qualitative workflow witness, not parity evidence. | Recorded full-lane result plus separately scoped mesh-matched oracle or convergence/quantitative trajectory evidence, then aggregate verifier |
| SR1-O1 | **Owner direction recorded:** fix D3/D4; D8 conditionally approved Sundials as the default. P1.1/P1.2 supplied the lifecycle/restart condition and P1.3 discharged it; P2.4 (`ff906f11`) discharges D3. | D8 implementation/evidence in `81fab481`; D3 implementation/evidence in `ff906f11`; D4 remains separately required (P2.5) |

[Codex GPT-5]

[P2.1 updates: Claude Opus 4.8]

[P2.6 updates: Claude Sonnet 5]

[C20 canonical-path repoint (D30): Claude Opus 4.8]
