# Finmag DOLFINx Full-Parity Plan (Phase 2)

## Goal

Continue the direct DOLFINx port from the completed core
(`2026-07-06-dolfinx-core-port.md`, Tasks 1-12, branch `dolfinx-port` at
`ce60ae3e`) toward full functional parity with the original legacy Finmag.
The end state of the repository must offer the same functionality it
originally did; any exception requires explicit user acceptance and
documentation. Work happens on branch `dolfinx-parity`.

All Phase 1 conventions remain binding: the Per-Slice Protocol, the frozen
FEniCS-2019 oracle (`ba9280934e188d7f3800e7b9865e70a9422f7687`,
`dev/bin/run-legacy-oracle`), the fixture schema
(`docs/superpowers/specs/legacy-oracle-fixtures.md`), minimal direct edits to
`src/finmag`, no `dev/dolfinx` code copying, by-name errors for anything not
yet ported, TDD, per-slice pixi gates folded into `dev/bin/verify-dolfinx-m5`,
and documentation updates (plan checkboxes, `transition-notes.org`,
`dev/dolfinx/porting_map.md`) in every slice.

## Task 13: Port DMI directly

**Files:** `src/finmag/energies/dmi.py`, its existing tests, lazy energy
exports, focused DOLFINx tests.

- [x] Port the `DMI(D, method, name, dmi_type)` constructor semantics with
  constant scalar `D` first; spatially varying `D` defers by name.
- [x] Support the legacy `dmi_type` variants: `'auto'`/3D bulk
  (`D * inner(m, curl(m))`), `'interfacial'`, and the 1D/2D forms with their
  legacy `unit_length ** (dim - 1)` scaling conventions; any variant not
  ported must raise by name, not silently fall back. (The undocumented
  legacy `'D2D'` variant also raises by name; it is not part of the public
  `dmi_type` contract documented on the legacy class.)
- [x] Box-assemble only, matching the Task 5 energy foundation; matrix/project
  methods keep raising precisely.
- [x] Analytic checks: helix/spiral energy for bulk DMI where practical, zero
  for uniform m under bulk DMI, sign convention pinned against the legacy
  form.
- [x] Coordinate-ordered legacy oracle fixture for at least one 3D bulk case
  and one interfacial case (field + energy), per the fixture schema.
- [x] Replace the fail-forward `ModuleNotFoundError` pin for `DMI` with
  curated ported-behavior tests; update `sim_with(D=...)` to construct the
  ported DMI.
- [x] Serial + (if the energies pattern has one) two-rank probe coverage;
  new gate `dolfinx-src-dmi-pytest` folded into `verify-dolfinx-m5`.

## Task 14: Port cubic anisotropy directly

**Files:** `src/finmag/energies/cubic_anisotropy.py`, its tests, exports.

- [x] Port `CubicAnisotropy(u1, u2, K1, K2=0, K3=0, name, assemble=False)`
  with constant axes/constants; spatially varying coefficients defer by name.
- [x] Preserve the legacy u3 = u1 x u2 convention and axis normalisation
  behavior exactly; validate non-orthogonal axes the way legacy did (check
  the oracle, do not guess).
- [x] Check against the module's own analytic reference values (same
  constants/axes as `cubic_anisotropy_test.py`) and a legacy oracle fixture.
- [x] Replace the fail-forward pin; new gate `dolfinx-src-cubicanis-pytest`
  folded into `verify-dolfinx-m5`.
- [x] Fix round 1 [Claude Opus 4.8]: port the legacy-default `assemble=False`
  native analytic field (`H = -1/(mu0 Ms) dE/dm`) so a default-constructed
  `CubicAnisotropy` participates in dynamics (review Important finding). Both
  field paths now work; documented the dormant legacy K2 `energy.cc:116`
  native typo (DELIBERATE DEVIATION, USER ACCEPTANCE PENDING) with per-term
  native oracle fixtures and a box->analytic convergence check.

## Task 15: Port time-dependent Zeeman and hysteresis

**Files:** `src/finmag/energies/zeeman.py` (deferred stubs `TimeZeeman`,
`DiscreteTimeZeeman`, `TimeZeemanPython`, `OscillatingZeeman`,
`DipolarField`), `src/finmag/sim/hysteresis.py`, their tests.

- [x] Port `TimeZeeman` (time-parametrised field with `update(t)`),
  `DiscreteTimeZeeman` (update intervals/switch-off), and `OscillatingZeeman`
  on top of the ported static `Zeeman`; `TimeZeemanPython` and `DipolarField`
  were also PORTED (not deferred) directly on top of `TimeZeeman`/`Zeeman` --
  the full legacy Expression-based capability translates onto a plain Python
  `field_function(t)` callable contract (see `transition-notes.org` for the
  input-type deviation and two discovered/preserved legacy quirks: `t_off=0.0`
  is falsy-disabled, and `DiscreteTimeZeeman.update()` never rebuilds its
  cached energy form, so `compute_energy()` goes stale after the first
  interval update even as `compute_field()` stays current -- DELIBERATE
  PRESERVATION, USER ACCEPTANCE PENDING).
- [x] The `EffectiveField.add(..., with_time_update)` auto-connection for
  `TimeZeeman` (preserved in Task 6) is now exercised with the real class
  (`sim.add(OscillatingZeeman(...))` + `run_until`), including the
  `switch_off_H_ext` interplay fixed in Task 9.
- [x] Port `sim.hysteresis` / `hysteresis_loop` (+ `sim.relax`, needed by
  `hysteresis`) for the ported interactions; oracle fixture for a tiny loop
  plus a Stoner-Wohlfarth-like qualitative switching + loop-closure witness
  (constructed with fresh `Simulation`/`relax()` calls per field step, since
  legacy's own `hysteresis()`/`hysteresis_loop()` were discovered -- and
  confirmed against the frozen oracle -- to not actually re-relax
  independently after the first stage; preserved verbatim, not fixed; see
  `transition-notes.org`).
- [x] New gate `dolfinx-src-timezeeman-pytest` (merges hysteresis, runs both
  `test_timezeeman_dolfinx.py` and `test_hysteresis_dolfinx.py`) folded into
  `verify-dolfinx-m5`.

## Task 16: Field-valued material parameters and regions

**Files:** `src/finmag/energies/exchange.py`, `anisotropy.py`, `dmi.py`,
`cubic_anisotropy.py`, `physics/llg.py` (alpha), `sim/sim.py`
(`mark_regions`, region-resolved parameters), their tests.

- [x] Accept `Field`/callable spatially varying `Ms`, `A`, `K1`, axes, `D`,
  and `alpha` where legacy did, with the legacy broadcasting/DG0-vs-CG1
  placement semantics established from the oracle, not assumed. (Established:
  A/D -> DG0, K1/K2/axis/cubic K -> CG1, alpha -> CG1 nodal per-node in the
  damping term *and* `gamma_LL`; the port's spaces already matched legacy.)
- [x] Port `sim.mark_regions` and region-scoped parameter assignment /
  energies-in-regions accounting for the ported interactions. (Legacy had NO
  region-scoped `set_alpha`/param setter; region support = `mark_regions` +
  per-region `compute_energy(dx=...)` + region `m_average`. Region-restricted
  submesh field output stays deferred by name.)
- [x] Oracle fixtures for at least one spatially-varying-`Ms` and one
  two-region case; the legacy `test_spatially_varying_anisotropy` and
  `test_energies_in_regions` invariants are the behavioral contract. (Also:
  nonuniform-A, varying-alpha RHS, and a spatially-varying-K2 native-typo
  divergence pin.)
- [x] Remove the corresponding by-name deferrals; extend gates. (New gate
  `dolfinx-src-varparams-pytest` folded into `verify-dolfinx-m5`; only legacy
  string Expressions and varying cubic axes remain deferred by name.)

## Task 17: Packaging — installable finmag

**Files:** new `pyproject.toml`, `pixi.toml`, `native/Makefile` hook or build
script, `dev/bin/verify-dolfinx-m5`, install docs.

Executed as Task 21 in the Phase 3 audited ordering (see
"Phase 3: full master parity (Tasks 18-29)" below:
`18→19→20→21(packaging, resumes Task 17)→22→...`); the checkboxes below are
this task's outcome, delivered under the Task 21 label.

- [x] Add a `pyproject.toml` making `src/finmag` an installable package
  (setuptools or hatchling; src-layout) with the runtime dependency set
  derived from actual imports, not guesses. Outcome: `pyproject.toml`
  (setuptools build backend, `package-dir={"": "src"}`,
  `[tool.setuptools.packages.find]` restricted to `finmag*` under `src/`);
  `[project.dependencies]` is `numpy`/`scipy` only, verified by grepping
  module-scope imports reachable from the DOLFINx lane (conda/pixi keeps the
  FEM/MPI/SUNDIALS stack out of pip's resolver — see the README INSTALL
  section and `transition-notes.org`'s Task 21 section for the split
  rationale).
- [x] Decide and implement the native-extension story: either a build-backend
  hook that invokes the `native/Makefile` targets for the DOLFINx lane
  (`bem_arrays`), or a documented two-step install (`pip install -e .` +
  `make` task) — prefer the simplest reliable mechanism; record the decision
  and its trade-offs. Outcome: documented two-step install, no build-backend
  hook — `dolfinx-install-editable` (`python -m pip install -e .
  --no-deps --no-build-isolation`) then `dolfinx-native-build` (`make -C
  native`, using the Makefile's own default `PYTHON`), plus a
  `dolfinx-provenance-check` step; a non-editable/built wheel is an explicit
  non-goal (native `.so` extensions are linked against this specific
  conda/pixi environment's ABI, not portable). See README INSTALL section.
- [x] `pixi run -e dolfinx` tasks work against the INSTALLED package
  (editable) instead of `PYTHONPATH=src`; keep `PYTHONPATH=src` working
  during the transition (both paths gated). Outcome: all 22
  `dolfinx-src-*` pixi tasks now run against the installed editable package;
  exactly one retained task, `dolfinx-src-import-pythonpath-fallback`, keeps
  the old `PYTHONPATH=src` mechanism exercised as a fallback gate step.
- [x] `import finmag` from a fresh editable install passes the import
  boundary and version-access checks; `verify-dolfinx-m5` runs green against
  the installed package. Outcome: `dolfinx-provenance-check` asserts
  `finmag.__file__` resolves under this checkout's `src/finmag`;
  `dev/bin/verify-dolfinx-m5` runs the editable install, native build, and
  provenance check up front, then every `dolfinx-src-*` gate, green
  end-to-end.
- [x] Do not break the legacy oracle lane (its checkouts predate
  `pyproject.toml`; `run-legacy-oracle` must stay functional). Outcome:
  confirmed unaffected — `dev/bin/run-legacy-oracle`'s detached worktrees
  predate `pyproject.toml` entirely and do not need it.
- [x] Install documentation: a short INSTALL section (README or docs) with
  the exact commands for the supported DOLFINx environment. Outcome:
  `README.md`'s "Installing the DOLFINx port (pixi)" subsection, including
  the version-split note (static `pyproject.toml` `version` vs. runtime
  `finmag.__version__`) and the wheel non-goal.

[Claude Sonnet 5]

## Later phase-2 slices (sketch, sequence after Task 17)

1. Docs/examples refresh for the supported API (may fold into Task 17 if
   small).
2. Native Sundials/CVODE backend (dolfin-free rebuild following the
   `bem_arrays` pattern; LLG `sundials_*` hooks already stubbed by name).
3. STT (Slonczewski, Zhang-Li) and `llg_stt`.
4. Thermal/SLLG and LLB.
5. Normal modes; NEB.
6. PBC/treecode demag native slice; `Demag2D`, `MacroGeometry`.
7. MPI-parallel stepping (generalise the xxx serial state contract).
8. External comparison workflows (OOMMF/Nmag/Magpar), HDF5/XDMF read-back,
   plotting/spherical helpers, remaining dropped convenience surfaces
   (skyrmion initialisers, `skyrmion_number`, `mesh_info`, ...).
9. Port the undocumented legacy `DMI(dmi_type='D2D')` variant, or formally
   accept its deferral (it is not part of the public `dmi_type` contract
   documented on the legacy class; see Task 13 above and
   `dev/dolfinx/porting_map.md`) so the Definition of Full Parity checklist
   below can close. [Claude Sonnet 5]

## Definition of Full Parity

- [ ] Every capability of the original legacy Finmag either works on DOLFINx
  or has an explicit, user-accepted, documented exception.
- [ ] The legacy test suite's invariants are ported or accounted for
  file-by-file (tracked in porting_map).
- [ ] `verify-dolfinx-m5` (or its successor) covers all ported capabilities.
- [ ] Packaging allows installation and use without `PYTHONPATH=src`.

## Phase 3: full master parity (Tasks 18-29)

Approved 2026-07-21. The authoritative register and per-task detail live in
`2026-07-21-master-parity-audit.md`; tasks execute in the audited order
18→19→20→21(packaging, resumes Task 17)→22→23→24→25→26→27→28→29. All
accept-drop decisions are DEFERRED to Task 29 by user instruction — until then
every capability not already formally dropped is treated as PORT.

## Task 18: Mesh tooling bridge (Netgen/Gmsh → DOLFINx)

**Files:** `src/finmag/util/meshes.py`, `src/finmag/util/mesh_templates.py`,
`src/finmag/util/nmesh_to_dolfin.py`, pixi env deps if needed.

- [x] Establish the legacy mesh-generation surface from the pixi tip: the
  generator functions in `util/meshes.py` (box/cylinder/sphere/ellipsoid/
  elliptical cylinder/nanodisk etc. via `mesh_templates.py` CSG), the
  gmsh/netgen invocation paths, and the md5-keyed mesh-file caching contract.
  [Claude Opus 4.8]
- [x] Port the Gmsh path first: drive Gmsh via its Python API and convert to
  `dolfinx.mesh` via `dolfinx.io.gmsh.model_to_mesh` (the 0.10 module name;
  `gmshio` in older releases) — no dolfin XML intermediates. Public generator
  signatures preserved; caching contract preserved on an XDMF store
  (documented). [Claude Opus 4.8]
- [x] Port `mesh_templates.py` CSG template classes on top of the Gmsh path,
  preserving template names/parameters and `csg()`/`hash()` semantics exactly
  (the Netgen-CSG text is retained byte-for-byte as the cache key; geometry is
  built through the Gmsh OCC kernel — documented internal change). [Claude Opus 4.8]
- [x] Netgen path: DEFERRED by name (conda-forge `netgen` is not in the
  dolfinx env and the `.geo`→netgen→DIFFPACK→dolfin-XML path does not port
  cleanly; the Gmsh path covers the geometry set the legacy tests need).
  `netgen_is_usable()` returns `False`. Task 29 review item. [Claude Opus 4.8]
- [x] `nmesh_to_dolfin.py`: DEFERRED by name — it emits legacy dolfin-XML
  (unreadable by DOLFINx) and is consumed only by the unported Nmag
  reference-generation harness (`tests/**/run_nmag*.py`, Task 27 scope). Not
  ported speculatively. Task 29 review item. [Claude Opus 4.8]
- [x] Validation: analytic volume checks per generator/template within the
  legacy tolerances (box exact `TOL3`, curved within `TOL1`), the preserved
  md5 hashing/naming contract, XDMF caching determinism (cache hit vs distinct
  files), and an FK-demag-on-a-generated-sphere smoke (demag factor ~1/3).
  Vertex/cell counts are NOT pinned (Gmsh OCC meshing differs from the frozen
  Netgen tip); validated via volume/quality metrics instead — documented.
  [Claude Opus 4.8]
- [x] New gate `dolfinx-src-meshes-pytest` folded into `verify-dolfinx-m5`;
  `gmsh` + `python-gmsh` added to the dolfinx feature (pixi.lock changed —
  called out in the report). [Claude Opus 4.8]

## Task 19: Remaining optional energies

**Files:** `src/finmag/energies/thin_film_demag.py`,
`src/finmag/energies/dw_fixed_energy.py`, `src/finmag/energies/__init__.py`.

- [x] Port `ThinFilmDemag` directly (the last legacy-tested energy still
  importing `dolfin` at module scope): preserve constructor semantics and the
  legacy thin-film approximation exactly; validate against the legacy
  `thin_film_demag_test.py` invariants plus a legacy oracle fixture.
  [Claude Sonnet 5]
- [x] `FixedEnergyDW` (`dw_fixed_energy.py`): untested even on master. Port
  faithfully if the module is self-contained on the ported stack; otherwise
  convert to a curated by-name deferral with documentation (Task 29 review
  item) — do not leave the raw dolfin import either way. DECISION: curated
  by-name deferral (untested even on legacy master, legacy's own todo notes
  call it broken, and it depends on the already-deferred Treecode demag
  solver). [Claude Sonnet 5]
- [x] Update the deferred-surfaces sweep and lazy exports; energies
  `__init__` raw-import flags flipped for whatever is ported. Both
  `ThinFilmDemag` and `FixedEnergyDW` are now `requires_legacy_dolfin=False`;
  zero optional energies remain `True`. [Claude Sonnet 5]
- [x] New gate or fold into `dolfinx-src-energies-pytest` (your call,
  document); `verify-dolfinx-m5` updated if a new gate is added. Folded into
  `dolfinx-src-energies-pytest` (two new test files added to the same pixi
  command); `verify-dolfinx-m5` unchanged. [Claude Sonnet 5]

## Task 20: Native Sundials/CVODE backend

**Files:** `native/src/sundials/`, `native/src/util/`, `native/Makefile`,
`src/finmag/drivers/sundials_integrator.py`, `src/finmag/drivers/llg_integrator.py`,
`src/finmag/physics/llg.py` (`sundials_*` hooks, already stubbed by name).

- [x] Rebuild the native sundials module dolfin-free for the DOLFINx env,
  following the proven `bem_arrays` pattern (`-DFINMAG_NO_DOLFIN`, per-env
  object trees, module set in the dolfinx Makefile branch). The SUNDIALS-7
  wrapper work from M2/M3 (SUNContext lifecycle, nonlinear-solver attachment,
  modern linear-solver wiring) is the starting point — audit which parts still
  touch dolfin/np_array-with-dolfin and separate exactly as 11a did for BEM.
  DONE: audited every sundials source unit (`py_sundials_module.cc`,
  `sundials_cvode_impl.h`, `numpy_malloc.{h,cc}`, `util/np_array.{h,cc}`) —
  zero dolfin symbol-level coupling, only the shared `finmag_includes.h`
  umbrella gated by `-DFINMAG_NO_DOLFIN`/`-DFINMAG_NO_SUNDIALS`. Separation was
  a pure compile-flag + link-set change in the dolfinx-branch of
  `native/Makefile`, no wrapper source edits. [Claude Sonnet 5]
- [x] Wire `SundialsIntegrator` on the DOLFINx stack: `llg.sundials_rhs`
  (+ `jtimes`/`psetup`/`psolve` as legacy wired them), xxx-ordered state
  consistent with the ported LLG contract, serial-only guard inherited.
  DONE: `sundials_rhs` was already real (backend-neutral `solve_for`);
  `sundials_psetup`/`sundials_psolve`/`sundials_jtimes` are now real
  (transcribed verbatim / via the `_jtimes_numpy` node-local kernel from
  native `dm_precession_i`/`dm_damping_i`/`dm_relaxation_i`) for the legacy
  default `bdf_gmres_prec_id` path (SPGMR + identity preconditioner + analytic
  Jacobian-times-vector). `use_slonczewski`/`use_zhangli` (STT) remain
  out-of-scope `NotImplementedError` by name — unneeded by the default path,
  already tracked as Task 29 items. [Claude Sonnet 5]
- [x] Make `backend="sundials"` work while scipy remains available; restore
  the legacy default-backend semantics (`llg_integrator` default) ONLY if the
  sundials path passes the full validation below — otherwise keep scipy
  default and record the decision explicitly. DONE, in two steps across the
  slice and its fix round: the initial slice validated `backend="sundials"`
  as a fully working opt-in but conservatively kept the `llg_integrator`
  factory default at `"scipy"`, pending review. Fix round 1 (Task 20 review)
  restored the `llg_integrator` factory default to `"sundials"` as the plan
  directs — full validation had already passed and every explicit-backend
  in-tree caller (`Simulation` passes `backend=self.integrator_backend`
  explicitly) is unaffected. `Simulation.integrator_backend`'s own default
  stays `"scipy"` (Phase 1 Task 8/9 sanctioned it as the temporary DOLFINx
  default; flipping it would couple every `run_until` gate to the native
  build, a larger re-validation deliberately deferred) — registered as
  **USER ACCEPTANCE PENDING** in `transition-notes.org` and
  `dev/dolfinx/porting_map.md`. [Claude Sonnet 5]
- [x] Validation: the legacy `sundials_ode` test invariants (simple/stiff),
  analytic macrospin trajectory vs closed form, cross-backend agreement
  (sundials vs scipy on the Task 9/10 core workflow within declared
  tolerances), and a `barmini`-class Sim advance mirroring the pixi
  `barmini-smoke` acceptance slice. DONE: all invariants ported into
  `test_sundials_driver_dolfinx.py` and green — analytic macrospin trajectory,
  jtimes-vs-finite-difference (<1e-5 relative), cross-backend agreement
  (max|Δm| < 1e-6 at reltol=1e-8/abstol=1e-10, short `run_until(1e-11)`
  horizon — see the honest caveat in `transition-notes.org`), and the
  barmini-class default-path advance. [Claude Sonnet 5]
- [x] Env: SUNDIALS libraries into the dolfinx pixi feature (pixi.lock change
  expected — call out); native build hygiene per Task 11b conventions. DONE:
  `sundials = ">=7,<8"` added to `[feature.dolfinx.dependencies]`, resolving
  **7.8.0** (dolfinx env) vs the legacy default env's **7.2.1** — both
  SUNDIALS-7 major, no new compat shims needed beyond what the wrapper already
  carries; `pixi.lock` +126 lines. [Claude Sonnet 5]
- [x] New gate `dolfinx-src-sundials-pytest` folded into `verify-dolfinx-m5`;
  driver/deferred-sweep tests updated (sundials no longer raises by name —
  fail-forward pins flip to ported-behavior assertions incl. the Task 10 skip
  guards, which were designed for exactly this moment). DONE: 17 tests
  initially (18 after fix round 1's added bare-factory-default integration
  test), folded into `verify-dolfinx-m5`; `test_llg_dolfinx.py`'s deferred
  sweep lost 3 by-name params (now-ported hooks) and gained a positive ported
  test; the fix round additionally flipped
  `test_llg_integrator_default_backend_is_scipy` to
  `test_llg_integrator_default_backend_is_sundials` (paired with an
  availability-guarded raises-by-name counterpart covering the other branch)
  and added a docstring cross-reference from `test_construction_core_state`'s
  `Simulation`-default pin to the USER ACCEPTANCE PENDING register entry.
  [Claude Sonnet 5]

## Task 22: STT — Slonczewski and Zhang-Li

**Files:** `src/finmag/physics/llg.py` (`set_stt`/`set_zhangli` and the STT
dm/dt terms), `src/finmag/physics/llg_stt.py`, `src/finmag/sim/sim.py`
(`set_stt`/`set_zhangli` pass-throughs), their tests.

- [x] Establish the legacy STT surface at the pixi tip. DONE: legacy
  `LLG.use_slonczewski(J, P, d, p, Lambda, epsilonprime, with_time_update)` and
  `use_zhangli(J_profile, P, beta, using_u0, with_time_update)` (with
  `Simulation.set_stt`/`toggle_stt`/`set_zhangli` pass-throughs) read at the
  oracle. The compiled kernels are `calc_llg_slonczewski_dmdt` (using
  `slonczewski_xiao_i`, NOT `slonczewski_i`) and `calc_llg_zhang_li_dmdt` in
  `native/src/llg/llg.cc`; `llg_stt.py`'s `LLG_STT` is the SEPARATE nonlocal-STT
  class (`calc_llg_nonlocal_stt_dmdt`, reached only via `kernel="llg_stt"`),
  distinct from the in-LLG flags. [Claude Opus 4.8]
- [x] Port the deterministic STT terms into the ported LLG's NumPy RHS. DONE:
  `_dmdt_slonczewski_numpy` (llg.cc:207,252) and `_dmdt_zhangli_numpy`
  (llg.cc:406) transcribe the native kernels term-for-term with native physical
  constants (llg.cc:21-25); precession is unconditional as in the compiled
  kernels. The Zhang-Li `(J.grad)m` operator (`_compute_zhangli_gradient`) is
  the legacy lumped-box functional (`compute_gradient_matrix`), and per-node
  `Ms` (`_ms_nodal`) is the legacy lumped CG1 projection. Native STT kernels NOT
  rebuilt (NumPy transcription slice, documented in transition-notes). [Claude Opus 4.8]
- [x] Coordinate-ordered legacy oracle fixtures + analytic pins. DONE:
  `slonczewski_rhs.json` (gen_slonczewski_rhs.py) and `zhangli_rhs.json`
  (gen_zhangli_rhs.py; also pins the discrete `H_gradm`), both 1D IntervalMesh,
  schema v1, generated via `dev/bin/run-legacy-oracle`. Analytic pins: STT
  linear in J, sign-flip with J, Slonczewski vanishing at `m || p`, Zhang-Li
  vanishing for uniform m. [Claude Opus 4.8]
- [x] scipy/sundials integration with STT active + dynamics witnesses. DONE:
  Zhang-Li domain-wall-displacement witness (port of legacy
  `zhang_li_test.test_zhangli`), Slonczewski tilt-sign witness (m_z change flips
  with J), and scipy-vs-sundials cross-backend agreement on the Zhang-Li case
  (rtol=1e-4, atol=1e-6). [Claude Opus 4.8]
- [x] Preserve `do_slonczewski`/`do_zhangli`-era public semantics. DONE: the
  ported `LLG` re-exposes `do_slonczewski`/`do_zhangli` flags (set by `use_*`,
  toggled by `Simulation.toggle_stt`); `set_stt`/`set_zhangli` pass-throughs
  restore the legacy signatures faithfully, including the `with_time_update`
  callable contract. Only `kernel="llg_stt"` stays deferred. [Claude Opus 4.8]
- [x] `llg_stt.py` (the separate STT-LLG class): DEFERRED by name. `LLG_STT` is
  the nonlocal spin-accumulation model (distinct capability, reached only via
  `Simulation(kernel="llg_stt")`, already a by-name deferral); Task 29-registered.
  Documented in transition-notes + porting_map. [Claude Opus 4.8]
- [x] New gate `dolfinx-src-stt-pytest` (14 tests) folded into
  `verify-dolfinx-m5`. DONE; full m5 green end-to-end. [Claude Opus 4.8]
- [x] Docs: this checklist, transition-notes "STT (Task 22)" section (formulas
  cited to C++ file:line, fixture numbers, witnesses), and porting_map update
  (STT no longer deferred; `llg_stt` decision). [Claude Opus 4.8]

## Task 23: PBC and treecode demag (native slice)

**Files:** `native/src/treecode_bem/` (Cython + C sources, built via its own
`setup.py` per `native/Makefile`), `src/finmag/energies/demag/fk_demag_pbc.py`,
`demag_treecode.py`, `treecode_bem.py`, `src/finmag/util/pbc2d.py`,
`src/finmag/sim/sim.py` (`pbc` paths), `energies/demag/__init__.py`.

- [x] Build `finmag.native.treecode_bem` for the DOLFINx env. DONE: audited the
  C/Cython sources -- pure-C octree fast-summation + Lindholm BEM kernels, zero
  dolfin/Boost/SWIG coupling (`grep` clean; the built `.so` links only
  libgomp/libm; import leaves `sys.modules` dolfin-free). Rewrote `setup.py`
  from removed-on-3.12 `distutils` to `setuptools`+`cythonize`
  (`language_level=3`, NumPy-2 clean), fixed a Cython duplicate-def, added
  `treecode_bem.so` to the DOLFINx Makefile `MODULES` with per-env build-temp
  and a `CFLAGS=` clear (the conda `make` was exporting the finmag PCH
  `-include` into the C build), and added `cython` to the pixi feature
  (`pixi.lock` changed). Legacy-lane activation NOT done (not trivially safe;
  noted in porting_map as follow-up). [Claude Opus 4.8]
- [x] Port the Python demag modules that consume it. DONE: `treecode_bem.py`
  (`TreecodeBEM(FKDemag)`, `Demag(solver='Treecode')`) and `fk_demag_pbc.py`
  (`MacroGeometry` + `BMatrixPBC` + `build_periodic_bem`,
  `FKDemag(macrogeometry=...)`) ported onto the Task 11b FK foundation via two
  clean `FKDemag` hooks (`_setup_bem`/`_apply_bem`). `demag_treecode.py` NOT
  ported (uses the never-built `fast_sum_lib`, Python-2 code; dead/experimental).
  `util/pbc2d.py`/`Simulation(pbc=...)` (constrained_domain periodic *spaces*, a
  DIFFERENT capability needing `dolfinx_mpc`) stays deferred by name -- the PBC
  *demag* contract uses `pbc=None` + `MacroGeometry`, so it is unaffected.
  [Claude Opus 4.8]
- [x] Validation: cross-method + analytic (documented as the plan's accepted
  evidence exception -- the oracle env never built treecode, so NO oracle
  numbers exist). DONE: single-tile periodic BEM == golden dense FK BEM
  bit-for-bit (3.5e-17); treecode-vs-dense-FK cross-check (<1e-6 direct-sum on
  a cube -- machine precision, since compact convex boundaries never enter the
  multipole approximation regime at any practical mac/p/num_limit, see
  round-1 review fix below); sphere demag factor ~1/3; periodic image-sum
  convergence (-0.334->-0.165->-0.125->-0.109, trend-only) + analytic
  out-of-plane thin-film Nz->1 limit. The legacy `demag_pbc_test` finite-bar
  1.2% numbers (a SKIP on every modern lane) are NOT reproduced/asserted --
  EMPIRICALLY VERIFIED reason: `mesh_1`'s inferred tile spacing coincides
  exactly with its own extent, triggering coincident-node ill-conditioning
  (row sums -> -2, one exact NaN); documented in transition-notes.org.
  [Claude Opus 4.8]
  **Round-1 review fix ([Claude Sonnet 5]):** the original "~4e-5 at legacy
  default mac=0.3/p=3" figure above was overclaimed -- unreproducible on the
  box/sphere geometries the encoded tests use (always machine precision
  there). A genuine approximation-regime witness (a 50:1 aspect-ratio bar)
  DOES reach ~4.6e-5 at mac=0.3, confirming the figure on the right geometry;
  see transition-notes.org and porting_map.md for the full investigation
  (C-source citations, mac/p/correct_factor roles) and
  `test_treecode_pbc_demag_dolfinx.py` for the new encoded witness tests.
- [x] Unported variants stay by-name. DONE: `GCR` deferred (Task 29 candidate);
  `Demag2D` DEFERRED (decision: heavy MeshEditor/Expression coupling, no
  treecode dependency -- documented in porting_map + transition-notes). [Claude
  Opus 4.8]
- [x] New gate `dolfinx-src-treecode-pytest` (20 tests -- 16 baseline + 4
  round-1 review additions) folded into `verify-dolfinx-m5`; full m5 green
  end-to-end. [Claude Opus 4.8] [Claude Sonnet 5]
- [x] Docs: this checklist, transition-notes "Treecode/PBC demag (Task 23)"
  section (dolfin audit, Cython/NumPy-2 inventory, validation basis + numbers,
  Demag2D/pbc decisions, legacy-lane note), porting_map update. [Claude Opus 4.8]

## Task 30: Convert examples/ to the ported package (practical-parity witness)

Executed out of audit order by user directive (2026-07-22), immediately after
Task 23. Purpose: user-level end-to-end validation that the ported package
runs real legacy workflows with MINIMAL interface changes and the same
practical results. Interface drift discovered here is a first-class finding
to surface, not something to paper over with rewrites.

**Files:** `examples/` scripts (+ per-example run/verify wiring), a new gate.

- [x] Inventory every example and classify. DONE: all 19 subdirs inventoried
  (see the transition-notes per-example table). (b) set = `dispersion_curves`
  (T25), `llb` (T24), `nmag_example_2` (T27), `boost_python` (Boost.Python C++
  tutorials — not the finmag API), plus the nmag-comparison / mayavi-vpython
  visualisation sibling scripts (T27 / T26); each annotated with a "requires
  Task N" header. Everything else converted (a). [Claude Opus 4.8]
- [x] Convert the (a) set with MINIMAL diffs. DONE: 18 scripts converted;
  every changed line justified in the report + inline. 11 interface-drift
  findings recorded in the dedicated table (transition-notes + report),
  incl. one genuine src BUG (coordinate<->vertex matching crashed on
  Gmsh/`from_geofile` meshes) fixed minimally and flagged for review, and one
  stale-example fix (`CubicAnisotropy` signature). [Claude Opus 4.8]
- [x] Controller amendment: `from_geofile` partially un-deferred. DONE:
  `src/finmag/util/geofile.py` implements a Netgen-CSG-subset `.geo`/`from_csg`
  loader on the Task 18 OCC/Gmsh machinery (orthobrick/sphere/plane-box/cylinder
  + and/not + multi-tlo; multitranslate & oblique planes raise by name); example
  `from_geofile("*.geo")` lines are unchanged. 10 focused tests in
  `dolfinx-src-meshes-pytest`. Netgen-binary backend still deferred. [Claude Opus 4.8]
- [x] Same practical results. DONE: exchange_demag vs checked-in nmag refs
  (averages 2.9e-3, demag E 3.5e-3, exch E 2.8e-2 at honest tolerances);
  macrospin & demag field/energy analytic; std_prob_3 flower E_total 0.2985 vs
  µMAG 0.302; std_prob_4 µMAG switching window (FULL); the rest physical-sanity.
  Reduced-resolution loosenings documented in transition-notes. [Claude Opus 4.8]
- [x] New gate `dolfinx-src-examples-pytest` folded into `verify-dolfinx-m5`:
  14 fast subprocess-run examples (~3-4 min) + 3 FULL-only (FINMAG_EXAMPLE_FULL=1
  documented per-script). [Claude Opus 4.8]
- [x] Docs. DONE: these checkboxes; transition-notes "Examples conversion (Task
  30)" section (per-example table, drift table, from_geofile inventory, result
  numbers); porting_map update. Attribution [Claude Opus 4.8].

## Task 31: Public field-array component-ordering correction (user-directed)

De-registers the interleaved-ordering interface drift: all PUBLIC array
surfaces return legacy component-blocked ordering (x1..xN, y1..yN, z1..zN);
internal storage unchanged. CRITICAL-class change: a wrong ordering fix does
not error, it silently scrambles physics.

- [ ] `compute_field()` on every interaction (EnergyBase box path, Exchange,
  Zeeman + time family + DipolarField, uniaxial + cubic (both field paths),
  DMI, ThinFilmDemag, FKDemag, treecode/MacroGeometry) returns xxx-blocked
  owned arrays — implemented ONCE at the shared boundary (EnergyBase helper /
  each class's return point), not N hand-copies.
- [ ] `EffectiveField.H_eff`/`compute()`/`compute_jacobian_only()` and
  `sim.effective_field()` are xxx-blocked; `LLG.solve`'s existing raw→xxx
  H_eff conversion REMOVED (double-conversion is the failure mode — pin with
  a test that would catch it).
- [ ] `Field.get_numpy_array_debug()` returns legacy component-blocked
  ordering (legacy = dolfin local vector = blocked).
- [ ] Every ordering-sensitive test/fixture/probe updated: oracle-fixture
  array→table conversions, energies/effectivefield MPI probes, any gate
  pinning the old raw ordering — each change justified (the fixture VALUES
  never change, only the conversion applied before comparison).
- [ ] Example workarounds reverted to legacy one-liners
  (`examples/demag/test_field.py`, `examples/macrospin/
  test_macrospin_alpha_rtol.py`); drift rows #3/#6 closed in the table
  (moved to a "corrected" section, not deleted).
- [ ] Cross-backend + composed-physics + examples gates all green — these are
  the anti-scramble safety net; plus one NEW dedicated ordering contract test
  (uniform-field asymmetric case where interleaved vs blocked provably
  differ, asserted against an analytic layout).
- [ ] Docs (verification-checklist item): plan checkboxes, transition-notes
  "Ordering correction (Task 31)" incl. the de-registered drift, porting_map
  register update. Attribution.
