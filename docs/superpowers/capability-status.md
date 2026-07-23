# Finmag DOLFINx Capability Status

**Source baseline audited:** 2026-07-23 at `dolfinx-parity` commit
`f1a1344c423e74687ddf820c1fafc056a6271fe1`

**Final target:** the functionality and user-facing interface of original
`master` (`b5015c5a`), except for deviations explicitly accepted by the
repository owner.

This is the canonical current capability inventory. `HANDOVER.md` is the short
entry point; `acceptance-register.md` is the sole owner-decision ledger. The
plans, `transition-notes.org`, and `dev/dolfinx/porting_map.md` preserve history
and evidence but do not override this status table.

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
  witness; behavioral parity needs tighter mesh-matched or convergence evidence;
- legacy public names needed by the supported workflow either work or raise a
  feature-specific error—never an incidental raw `dolfin` import failure;
- user documentation includes installation, a minimal simulation, a realistic
  simulation, validation limits, and a plainly visible “not implemented yet”
  section.
- every selected owner-Now legacy-test family has a positive DOLFINx witness or
  an explicitly linked owner disposition; P5.3 may execute in parallel with
  implementation slices but is an SR1 completion gate.

The owner has approved correcting D3 (stale discrete-field energy) and D4
(hysteresis no-re-relax). D8's condition is discharged: `81fab481` restores
native Sundials as the public default after the P1.1/P1.2 lifecycle and restart
evidence. SciPy remains a fully supported explicit opt-in. SR1 cannot be
declared complete until the approved D3/D4 corrections are implemented and
green.

## Current capability matrix

| ID | Capability | Status | Current evidence and boundary | SR1 |
|---|---|---|---|---|
| C01 | Install, native build, provenance, aggregate CI | **ported** | Editable pixi install; BEM, CVODE and treecode native builds; final clean main-worktree `dev/bin/verify-dolfinx-m5` on `81fab481` exited 0 with all 32 steps green (fast examples 14 passed/3 skipped in 246.36s; `/tmp/finmag-p1-default-m5.log`) | required, green |
| C02 | Package import and top-level compatibility exports | **partial** | Core `Simulation`, `sim_with`, `Field`, energies, versions and configuration import. `626dfcc8` makes `set_logging_level` and `finmag.example` (`bar`, `barmini`, `nanowire`) work with no legacy `dolfin` installed, and makes `NormalModeSimulation`, `normal_mode_simulation`, `example.sphere_inside_airbox` and `example.normal_modes` raise a curated `NotImplementedError` naming the deferred feature instead of a raw `ModuleNotFoundError`. Gate `dolfinx-src-import-pytest` 19 passed/3 skipped in 23.65s (was 9 passed/2 skipped). Submodule spellings such as `from finmag.example.normal_modes import disk` still raise raw `ModuleNotFoundError`, and `finmag.util.helpers` as a module still imports legacy `dolfin` | everyday exports are blockers; normal-mode exports later |
| C03 | Mesh generation | **partial** | Common Gmsh/OCC generators, template CSG, caching, and a subset of legacy Netgen `.geo` parsing have analytic/regression coverage. Netgen binary/nmesh and specialist generators remain unavailable | common subset required; probe Netgen need before any port; specialist generators not now |
| C04 | `Field` data model and inspection | **partial** | Constants/callables/arrays and component ordering have regression coverage; point probing, spherical conversion and topology have the focused `dolfinx-src-io-utils-pytest` gate. `from_generic_vector`, arithmetic (`cross`, `dot`, coercion), HDF5, plotting and VTK/XDMF readback remain unavailable | owner selected these missing operations for SR1; blockers |
| C05 | Spatial parameters and regions | **ported** | Varying `Ms`, `A`, `K`, `D`, axis and alpha plus region energies/averages have oracle/regression witnesses. Region-restricted field output is separate, unported I/O | required, green |
| C06 | Deterministic energy interactions | **partial** | Exchange, Zeeman family, uniaxial/cubic anisotropy, DMI and ThinFilmDemag have oracle/analytic/regression coverage. The varying-K2 witness deliberately pins divergence from the legacy indexing bug; varying cubic axes are missing. Legacy matrix/project/direct methods and string-expression compatibility are not required for SR1; D2D is later and FixedEnergyDW not now | common interactions and varying cubic axes required; selected legacy surfaces deferred |
| C07 | Demagnetisation | **partial** | FK is oracle-validated; treecode/MacroGeometry is analytic and cross-method validated. `sim_with` now wires `nx`/`ny`/`spacing_x`/`spacing_y` through the dense-FK `MacroGeometry` path (pitch semantics, mesh coordinate units, no `unit_length` scaling; `2cf4647f`/`899d3906`), refusing the touching/overlapping tiling (`pitch <= extent`) by name -- a deliberate divergence recorded as D17. The Treecode factory selector, GCR, Demag2D and specialist solver paths remain unavailable through `sim_with` | Treecode factory exposure later/review; D17 disposition pending owner |
| C08 | Deterministic serial LLG and composed dynamics | **ported** | `Simulation`, `run_until`, `relax`, effective-field composition, pins by index and deterministic trajectories have analytic/oracle/regression coverage. `df5fc1a1` (SR1 P2.3) restores callable pin masks: `Simulation.__set_pins` resolves the callable in the sim layer -- per-point over the owned-node coordinate array from `coords_and_values()[0]`, RAW MESH UNITS (no `unit_length` scaling), truthy return marks that node pinned, index = position in that same coordinate-ordered `xxx` array that `LLG._pins` consumes -- matching legacy (`git show b5015c5a:src/finmag/sim/sim.py` ~906-918) exactly. Indexed pins and `LLG.set_pins` are unchanged. Gate `dolfinx-src-simulation-pytest` 44 -> 49 passed (5 new); neighbour `dolfinx-src-llg-pytest` 16 passed, unchanged | required core green; callable pins now ported |
| C09 | SciPy and native Sundials integration | **ported** | `81fab481` restores native Sundials as the public `Simulation`/`sim_with` default. Both backends construct, advance, reset/reinitialise, schedule, save and satisfy immediate plus trajectory restart checks. The default's native construction and advance are directly witnessed; the core smoke reports `integrator_backend: sundials` at `t=1e-12`. SciPy remains a fully supported explicit opt-in. Final clean main aggregate: 32/32 green (`/tmp/finmag-p1-default-m5.log`) | required, green; D8 discharged |
| C10 | Spin-transfer torque | **partial** | Slonczewski and Zhang-Li local terms run separately with oracle/analytic/cross-backend witnesses. D11 approves an explicit error for conflicting configuration, but current source still silently makes the last call win. Separate nonlocal `LLG_STT` is deferred | blocker until D11 error semantics land; nonlocal not now |
| C11 | Time-dependent fields and hysteresis | **partial** | Time-dependent Zeeman family, relaxation and hysteresis are exercised. The D3 `DiscreteTimeZeeman` stale-energy defect is CORRECTED in `ff906f11` (SR1 P2.4): the interval refresh now routes through `set_value`, re-forming the cached energy form, with focused gate `dolfinx-src-timezeeman-pytest` 28 -> 32 passed and field bit-invariance (max\|difference\| = 0.0) proving dynamics/effective-field/`.ndt` output are unchanged. The D4 hysteresis no-re-relax defect REMAINS approved but unimplemented (SR1 P2.5, still pending), so this row is only partially discharged | blocker until the D4 fix (P2.5) lands |
| C12 | Scheduler and output | **partial** | Scheduler, NDT, coordinate-table `.npy` snapshots and VTK/XDMF write have regression/end-to-end tests. The current `.npy` contract is PASS but additional snapshot work is owner-Later. HDF5 and VTK/XDMF readback, plotting and region/submesh output are unavailable; historical movie helpers are separate | HDF5, plotting and region output are SR1 blockers; movie wrapper later/review |
| C13 | Restart | **partial** | Coordinate-aware v2 restores magnetisation/time on both backends; archives record truthful `sim.integrator_backend`, writable `sim.driver` stays in sync, v1 is rejected, and immediate plus uninterrupted-vs-restarted trajectory tests pass. Metadata reapplication/validation and full-state reconstruction remain separate D16 questions | metadata boundary remains |
| C14 | Point/topology utilities | **ported** | Point evaluation, `get_spherical`, skyrmion number and density have analytic/regression coverage in `dolfinx-src-io-utils-pytest` | required, green |
| C15 | Converted examples | **partial** | Aggregate baseline: 32 green steps; fast lane 14 passed, 3 skipped. Clean FULL baseline: 12 passed, 5 failed — three unchanged-wrapper timeouts (not physics failures), one missing scheduler `save_m` keyword, and one raw `dolfin` import through `finmag.util.helpers`. `FINMAG_EXAMPLE_FULL=1` runs the fourteen fast plus three slow entries. The std_prob4 0.10–0.18 ns crossing window is qualitative, not unchanged-behavior evidence | blocker |
| C16 | Thermal SLLG and LLB | **deferred** | Legacy surface remains tied to unported modules/native code | Not now; retain for later full-parity decision/work |
| C17 | Normal modes, eigensolvers, ringdown and FFT/PSD | **planned** | Since `626dfcc8`, top-level `finmag.NormalModeSimulation`/`normal_mode_simulation` and `example.sphere_inside_airbox`/`example.normal_modes` raise a curated `NotImplementedError` instead of reaching legacy `dolfin`; the submodule spelling `from finmag.example.normal_modes import disk` still reaches it directly. No DOLFINx workflow exists either way | after SR1; required for full parity |
| C18 | Path methods | **deferred** | Original master contains several legacy NEB implementations; they are not ported. No public GNEB API/workflow was found, so GNEB is not an interface-parity requirement; algorithmic overlap must be checked when NEB variants are inventoried | after SR1 |
| C19 | Parallel and periodic simulation surfaces | **partial** | Ownership-sensitive Field/energy MPI probes and treecode periodic macrogeometry work. General MPI stepping and `Simulation(pbc=)` function-space periodicity do not | serial `Simulation(pbc=)` is SR1 work; MPI stepping later |
| C20 | OOMMF/Nmag/Magpar comparison workflows | **planned** | Checked-in reference data supports some comparisons, but only a narrow Nmag example currently exercises it against DOLFINx; the live harnesses are not selected for SR1 | selected checked-data comparisons are SR1 work; live runners later/review |
| C21 | Legacy utilities and specialist workflows | **partial** | Common helpers used by ported workflows work. Owner-selected initialisers, HDF5/plotting/visualisation, probing and simulation helpers are missing; unselected research-era surfaces still require inventory or owner decision | selected checklist rows are SR1 work; unselected long tail later/not now |
| C22 | Full public-interface parity | **partial** | A focused 45-surface audit corrected six parameter names and ordering. It was not exhaustive; top-level exports, `sim_with`, backend lifecycle, Field long tail and unported master modules remain open | supported SR1 names required; exhaustive audit for full parity |

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
   `acceptance-register.md` (disposition pending owner decision).
3. The recorded clean FULL lane has three wrapper timeouts (`cubic_anisotropy`
   hysteresis, std_prob3 and std_prob4), an immediate missing scheduler
   `save_m` keyword in `cubic_anisotropy/sim.py`, and a raw legacy-`dolfin`
   import through `finmag.util.helpers` in `magnetic_grain/suess_2001.py`.
   The timeouts are baseline harness evidence, not physics failures.

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
| SR1-V1 | Run `FINMAG_EXAMPLE_FULL=1`; diagnose failures and make only separately reviewed minimal fixes. Treat std_prob4's broad crossing window as a qualitative workflow witness, not parity evidence. | Recorded full-lane result plus separately scoped mesh-matched oracle or convergence/quantitative trajectory evidence, then aggregate verifier |
| SR1-O1 | **Owner direction recorded:** fix D3/D4; D8 conditionally approved Sundials as the default. P1.1/P1.2 supplied the lifecycle/restart condition and P1.3 discharged it; P2.4 (`ff906f11`) discharges D3. | D8 implementation/evidence in `81fab481`; D3 implementation/evidence in `ff906f11`; D4 remains separately required (P2.5) |

[Codex GPT-5]

[P2.1 updates: Claude Opus 4.8]
