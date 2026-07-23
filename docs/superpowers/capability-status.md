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
(hysteresis no-re-relax), and restoring Sundials as the default under D8 after
lifecycle parity is demonstrated. These are now implementation and validation
tasks, not open policy questions. SR1 cannot be declared complete until the
approved corrections are implemented and green.

## Current capability matrix

| ID | Capability | Status | Current evidence and boundary | SR1 |
|---|---|---|---|---|
| C01 | Install, native build, provenance, aggregate CI | **ported** | Editable pixi install; BEM, CVODE and treecode native builds; 32-step `dev/bin/verify-dolfinx-m5`; push and PR CI green at the reconciled commit | required, green |
| C02 | Package import and top-level compatibility exports | **partial** | Core `Simulation`, `sim_with`, `Field`, energies, versions and configuration import. `NormalModeSimulation`, `normal_mode_simulation`, `set_logging_level`, and `example` can still fail through raw `dolfin` imports | everyday exports are blockers; normal-mode exports later |
| C03 | Mesh generation | **partial** | Common Gmsh/OCC generators, template CSG, caching, and a subset of legacy Netgen `.geo` parsing have analytic/regression coverage. Netgen binary/nmesh and specialist generators remain unavailable | common subset required; probe Netgen need before any port; specialist generators not now |
| C04 | `Field` data model and inspection | **partial** | Constants/callables/arrays and component ordering have regression coverage; point probing, spherical conversion and topology have the focused `dolfinx-src-io-utils-pytest` gate. `from_generic_vector`, arithmetic (`cross`, `dot`, coercion), HDF5, plotting and VTK/XDMF readback remain unavailable | owner selected these missing operations for SR1; blockers |
| C05 | Spatial parameters and regions | **ported** | Varying `Ms`, `A`, `K`, `D`, axis and alpha plus region energies/averages have oracle/regression witnesses. Region-restricted field output is separate, unported I/O | required, green |
| C06 | Deterministic energy interactions | **partial** | Exchange, Zeeman family, uniaxial/cubic anisotropy, DMI and ThinFilmDemag have oracle/analytic/regression coverage. The varying-K2 witness deliberately pins divergence from the legacy indexing bug; varying cubic axes are missing. Legacy matrix/project/direct methods and string-expression compatibility are not required for SR1; D2D is later and FixedEnergyDW not now | common interactions and varying cubic axes required; selected legacy surfaces deferred |
| C07 | Demagnetisation | **partial** | FK is oracle-validated; treecode/MacroGeometry is analytic and cross-method validated. `sim_with` exposes only FK and rejects its legacy MacroGeometry arguments; GCR, Demag2D and specialist solver paths remain unavailable | wire MacroGeometry through FK now; Treecode factory exposure later/review |
| C08 | Deterministic serial LLG and composed dynamics | **ported** | `Simulation`, `run_until`, `relax`, effective-field composition, pins by index and deterministic trajectories have analytic/oracle/regression coverage. Callable pin masks are not ported | required core green; missing pin form explicit |
| C09 | SciPy and native Sundials integration | **partial** | Both backends now advance, reset/reinitialise, and satisfy immediate plus trajectory restart checks. `3f4ed4ea` made reset backend-neutral; `17f24413` completed truthful restart provenance/integrity. The remaining D8 public-default difference is intentional until P1.3 | P1.3 default change remains |
| C10 | Spin-transfer torque | **partial** | Slonczewski and Zhang-Li local terms run separately with oracle/analytic/cross-backend witnesses. D11 approves an explicit error for conflicting configuration, but current source still silently makes the last call win. Separate nonlocal `LLG_STT` is deferred | blocker until D11 error semantics land; nonlocal not now |
| C11 | Time-dependent fields and hysteresis | **partial** | Time-dependent Zeeman family, relaxation and hysteresis are exercised. The stale-energy and no-re-relax defects remain in source, but their corrections are owner-approved under D3/D4 | blocker until approved fixes land |
| C12 | Scheduler and output | **partial** | Scheduler, NDT, coordinate-table `.npy` snapshots and VTK/XDMF write have regression/end-to-end tests. The current `.npy` contract is PASS but additional snapshot work is owner-Later. HDF5 and VTK/XDMF readback, plotting and region/submesh output are unavailable; historical movie helpers are separate | HDF5, plotting and region output are SR1 blockers; movie wrapper later/review |
| C13 | Restart | **partial** | Coordinate-aware v2 restores magnetisation/time on both backends; archives record truthful `sim.integrator_backend`, writable `sim.driver` stays in sync, v1 is rejected, and immediate plus uninterrupted-vs-restarted trajectory tests pass. Metadata reapplication/validation and full-state reconstruction remain separate D16 questions | metadata boundary remains |
| C14 | Point/topology utilities | **ported** | Point evaluation, `get_spherical`, skyrmion number and density have analytic/regression coverage in `dolfinx-src-io-utils-pytest` | required, green |
| C15 | Converted examples | **partial** | Aggregate baseline: 32 green steps; fast lane 14 passed, 3 skipped. Clean FULL baseline: 12 passed, 5 failed — three unchanged-wrapper timeouts (not physics failures), one missing scheduler `save_m` keyword, and one raw `dolfin` import through `finmag.util.helpers`. `FINMAG_EXAMPLE_FULL=1` runs the fourteen fast plus three slow entries. The std_prob4 0.10–0.18 ns crossing window is qualitative, not unchanged-behavior evidence | blocker |
| C16 | Thermal SLLG and LLB | **deferred** | Legacy surface remains tied to unported modules/native code | Not now; retain for later full-parity decision/work |
| C17 | Normal modes, eigensolvers, ringdown and FFT/PSD | **planned** | Top-level normal-mode exports currently reach legacy `dolfin`; no DOLFINx workflow | after SR1; required for full parity |
| C18 | Path methods | **deferred** | Original master contains several legacy NEB implementations; they are not ported. No public GNEB API/workflow was found, so GNEB is not an interface-parity requirement; algorithmic overlap must be checked when NEB variants are inventoried | after SR1 |
| C19 | Parallel and periodic simulation surfaces | **partial** | Ownership-sensitive Field/energy MPI probes and treecode periodic macrogeometry work. General MPI stepping and `Simulation(pbc=)` function-space periodicity do not | serial `Simulation(pbc=)` is SR1 work; MPI stepping later |
| C20 | OOMMF/Nmag/Magpar comparison workflows | **planned** | Checked-in reference data supports some comparisons, but only a narrow Nmag example currently exercises it against DOLFINx; the live harnesses are not selected for SR1 | selected checked-data comparisons are SR1 work; live runners later/review |
| C21 | Legacy utilities and specialist workflows | **partial** | Common helpers used by ported workflows work. Owner-selected initialisers, HDF5/plotting/visualisation, probing and simulation helpers are missing; unselected research-era surfaces still require inventory or owner decision | selected checklist rows are SR1 work; unselected long tail later/not now |
| C22 | Full public-interface parity | **partial** | A focused 45-surface audit corrected six parameter names and ordering. It was not exhaustive; top-level exports, `sim_with`, backend lifecycle, Field long tail and unported master modules remain open | supported SR1 names required; exhaustive audit for full parity |

## Current high-priority failures

1. Accessing `finmag.NormalModeSimulation`, `finmag.normal_mode_simulation`,
   `finmag.set_logging_level`, or `finmag.example` can expose raw
   `ModuleNotFoundError: No module named 'dolfin'` instead of either working or
   failing by feature name.
2. `sim_with` rejects treecode and MacroGeometry arguments although the direct
   `Demag(solver='Treecode')` and `MacroGeometry` implementation is ported.
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
| SR1-A1 | Repair the top-level import/error boundary: make the SR1 convenience exports work and make deferred families fail by feature name. Do not port normal modes in this slice. | `dolfinx-src-import-pytest`, clean-process probes, aggregate verifier |
| SR1-A2 | Wire the already-ported dense-FK MacroGeometry path through legacy `sim_with` nx/ny/spacing arguments. Do not add Treecode factory exposure or change demag algorithms. | RED factory tests, `dolfinx-src-treecode-pytest`, `dolfinx-src-simulation-pytest`, aggregate verifier |
| SR1-V1 | Run `FINMAG_EXAMPLE_FULL=1`; diagnose failures and make only separately reviewed minimal fixes. Treat std_prob4's broad crossing window as a qualitative workflow witness, not parity evidence. | Recorded full-lane result plus separately scoped mesh-matched oracle or convergence/quantitative trajectory evidence, then aggregate verifier |
| SR1-O1 | **Owner direction recorded:** fix D3/D4; D8 conditionally approves Sundials as the default. P1.1/P1.2 satisfied the lifecycle/restart validation condition; P1.3 has not yet changed the public default. | Owner-meeting dispositions recorded in `acceptance-register.md` on 2026-07-23; P1 evidence in `3f4ed4ea`/`17f24413` |

[Codex GPT-5]
