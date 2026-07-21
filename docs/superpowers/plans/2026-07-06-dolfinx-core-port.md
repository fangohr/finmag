# Finmag DOLFINx Direct-Port Implementation Plan

## Goal

Port Finmag to DOLFINx by validating uncertain FEM mechanics under
`dev/dolfinx` and then editing each accepted capability directly in its existing
`src/finmag` module. Do not create `dev/dolfinx/finmag`, duplicate the package,
or plan a final copy/move operation.

## Fixed Decisions

- [x] Use `src/finmag` as the only production implementation.
- [x] Keep `dev/dolfinx` as a bounded mechanics-probe lane.
- [x] Preserve the public Finmag architecture rather than promoting
  `PrototypeSimulation`.
- [x] Use the Python-3/FEniCS-2019 `pixi` commit as the immutable legacy oracle
  after direct DOLFINx source edits begin.
- [x] Prefer small direct module diffs over a parallel package and final move.

## Per-Slice Protocol

Every capability follows the same sequence:

1. Identify the exact public surface and trustworthy legacy tests.
2. State the scientific invariant, units, ordering, and tolerance.
3. Add or run a small `dev/dolfinx` probe only for unresolved DOLFINx behavior.
4. Generate a coordinate-ordered legacy reference fixture only when no analytic
   reference is practical.
5. Run the probe/reference and record RED or the missing production behavior.
6. Edit the corresponding `src/finmag` module directly.
7. Port the existing source tests in place and add only necessary DOLFINx cases.
8. Run the focused source gate, differential/analytic check, serial test, and
   any ownership-sensitive two-rank test.
9. Check `git diff --check`, tracked worktree cleanliness, and the slice's diff
   scope.
10. Commit the source capability separately from unrelated probes or cleanup.

No step copies implementation code from `dev` into `src`. The dev result is
evidence about an API or numerical operation; the production implementation is
written against the existing Finmag abstraction.

## Task 1: Correct the control plane before source edits

**Files:** `pixi.toml`, `pixi.lock`, DOLFINx verifier/CI, migration documents.

- [x] Pin `fenics-dolfinx = "0.10.*"` instead of `"*"`.
- [x] Add SciPy to the isolated DOLFINx environment.
- [x] Rename or describe the existing M4 gate as the frozen prototype gate.
- [x] Add focused `src`-port task names without referring to a staged package.
- [x] Make validation fail if it changes tracked files.
- [x] Prevent native version generation from rewriting
  `src/finmag/__version__.py` during read-only oracle validation.
- [x] Record DOLFINx, Python, NumPy, SciPy, PETSc, and MPI versions.
- [x] Run the existing 105-test prototype gate after the environment change.

Expected source diff: none.

## Task 2: Freeze the legacy oracle contract

**Oracle commit:** `ba9280934e188d7f3800e7b9865e70a9422f7687`.

- [x] Record the exact M2 and M3 commands and measured results at the oracle.
- [x] Add a helper that can run a focused legacy reference command from a
  temporary detached checkout without altering the active source tree.
- [x] Define a small fixture schema: oracle commit, mesh recipe, coordinates,
  physical parameters, units, values, and tolerances.
- [x] Require coordinate ordering rather than raw legacy dof ordering.
- [x] Document when an analytic result is sufficient and a fixture is unwanted.

Expected source diff: none.

## Task 3: Make package imports sliceable

**Files:** `src/finmag/__init__.py`, `src/finmag/energies/__init__.py`, focused
import tests. Keep `src/finmag/init.py` only as long as still needed by explicit
legacy surfaces.

Dev question: confirm that importing DOLFINx itself has no side effect needed by
the package initializer. No duplicate package probe is needed.

- [x] Inventory intentional top-level public names.
- [x] Add tests that plain `import finmag` imports neither legacy `dolfin` nor
  optional native modules.
- [x] Replace eager wildcard imports with explicit lazy public exports.
- [x] Make optional/unported features load only when requested.
- [x] Preserve top-level `Simulation`, `sim_with`, `Field`, and version access,
  plus the common names in `finmag.energies`, through the lazy boundaries.
- [x] Verify this backend-neutral change against both the oracle environment and
  the DOLFINx environment.
- [x] Run the full M3 gate for the final time against the active source before
  the first DOLFINx-only module replacement.

This is the only deliberate transition seam. Do not add a general FEM backend
selector or `try dolfin / try dolfinx` implementation branches.

## Task 4: Port `Field` directly

**Files:** `src/finmag/field.py`, the preserved legacy oracle tests
`src/finmag/field_test.py` and `src/finmag/field_setters_test.py`, and focused
production tests under `src/finmag/tests/`.

The 1,500-line DOLFIN-specific test module remains intact at the immutable
oracle instead of being mechanically rewritten. Its trustworthy invariants are
ported into a compact DOLFINx matrix covering mesh dimensions 1/2/3, scalar
and 1/2/3/4-component fields, setters, ordering, ownership, and accessors.

Dev evidence already available: blocked vector layout, callable interpolation,
coordinate matching, normalization, volume average, VTK, and XDMF.

Additional probe required before editing:

- [x] Establish owned/ghost behavior for raw arrays, averages, normalization,
  and coordinate/value export on two ranks.
- [x] Decide and test the exact compatibility conversion for legacy `xxx`
  component ordering where still required by a driver.

Direct source acceptance:

- [x] Preserve constants, callables, `set`, `from_array`, `from_field`, and
  `from_function`.
- [x] Preserve scalar/vector inspection, raw arrays, coordinate-ordered `xyz`,
  volume averages, normalization, and the underlying function/space access.
- [x] Make every local/global ownership contract explicit in method names or
  documentation.
- [x] Preserve the legacy Field tests as oracle history and port their trusted
  dimension, setter, accessor, ordering, and normalization invariants into a
  focused DOLFINx production suite without duplicating the package under
  `dev/`.
- [x] Pass serial and two-rank Field gates.
- [x] Mark expression strings, point-measure arithmetic, plotting, spherical
  conversion, and legacy HDF5 explicitly supported or explicitly unavailable.

## Task 5: Port the energy foundation and common interactions

**Files:** `src/finmag/energies/energy_base.py`, `exchange.py`, `zeeman.py`,
`anisotropy.py`, their existing tests, and lazy energy exports.

### 5a. Box-method foundation

- [x] Probe owned/ghost-safe lumped nodal volumes and derivative assembly on
  two ranks.
- [x] Port `EnergyBase` directly with box assembly as the first supported method.
- [x] Verify effective-field sign, `mu0`, `Ms`, unit-length powers, and energy
  reduction analytically.
- [x] Reject unsupported matrix/project methods precisely rather than silently
  changing algorithms.

### 5b. Zeeman

- [x] Port constant/callable field setup, field values, energy, and average.
- [x] Check `H_eff == H` and analytic energy on 2D and 3D meshes.

### 5c. Exchange

- [x] Port scalar `A` first, keeping constructor and lifecycle semantics.
- [x] Check zero constant fields, analytic linear fields, and unit scaling.

### 5d. Uniaxial anisotropy

- [x] Port constant `K1`, `K2`, and axis values first.
- [x] Check parallel/perpendicular states, axis validation, field direction, and
  legacy reference values.
- [x] Defer spatially varying coefficients to a separate slice rather than
  quietly treating them as constants.

## Task 6: Port `EffectiveField` directly

**Files:** `src/finmag/physics/effective_field.py` and its existing tests.

- [x] Preserve unique-name enforcement and the existing interaction lifecycle.
- [x] Preserve `add`, `get`, `exists`, `all`, `remove`, total field, total energy,
  and time-update callbacks.
- [x] Test registry behavior with interaction doubles independently of FEM.
- [x] Test total field and energy with all three ported interactions.
- [x] Test that removing or replacing an interaction changes dynamics, not only
  reported energy.

## Task 7: Port the deterministic LLG core

**Files:** `src/finmag/physics/llg.py` and focused LLG/equation tests.

- [x] Generate coordinate-ordered legacy RHS references for one macrospin and
  one nonuniform field case. (Macrospin uses the closed-form LL RHS; the
  nonuniform case uses an oracle fixture on a 1D interval mesh,
  `src/finmag/tests/fixtures/llg_rhs_nonuniform.json`.)
- [x] Port scalar `Ms`, scalar `alpha`, `gamma`, `set_m`, `solve`, and
  `solve_for` directly.
- [x] Drive every RHS evaluation from the complete `EffectiveField` registry.
- [x] Verify precession sign, damping direction, physical time scale, pinning
  (retained, serial), and unit-length invariance after accepted steps.
- [x] Keep native Sundials, STT, thermal dynamics, and multi-rank stepping out of
  this slice with explicit availability errors.

## Task 8: Reuse the SciPy driver

**Files:** `src/finmag/drivers/scipy_integrator.py`, `llg_integrator.py`, and
existing driver tests.

- [ ] Add SciPy to the DOLFINx environment before touching source.
- [ ] Confirm the installed VODE/BDF path against a small stiff ODE probe.
- [ ] Preserve the existing `advance_time` and tolerance interface.
- [ ] Implement real reinitialization from the current field state.
- [ ] Reject backward integration and unsuccessful integration explicitly.
- [ ] Make SciPy the temporary supported/default DOLFINx backend without
  changing the public backend-selection API.

Do not replace the integrator API or introduce `solve_ivp` merely as migration
cleanup; that can be evaluated separately after scientific parity.

## Task 9: Port the core `Simulation` directly

**Files:** `src/finmag/sim/sim.py`, focused existing simulation tests, top-level
lazy exports, and the simplest existing example needed for the core smoke.

- [ ] Port construction, scalar `Ms`, `unit_length`, name, scalar `alpha`, and
  scalar `gamma`.
- [ ] Port `set_m`, `m`, `m_field`, `m_average`, `t`, and `dmdt`.
- [ ] Port interaction add/get/list/remove and energy methods.
- [ ] Port integrator creation, tolerances, `advance_time`, `run_until`, reset,
  and reinitialization.
- [ ] Port `sim_with` for Exchange, Zeeman, and uniaxial anisotropy.
- [ ] Make demag, PBC, stochastic kernels, STT, scheduler, restart, and unported
  output fail explicitly when requested.
- [ ] Remove touched dead legacy branches rather than leaving NameErrors behind.
  Leave unrelated long-tail modules unchanged outside the active import graph.
- [ ] Run an end-to-end physical-time workflow using all configured interaction
  fields.

## Task 10: Establish the first direct-source DOLFINx gate

- [ ] `PYTHONPATH=src python -c "import finmag"` resolves the edited source.
- [ ] Focused Field, energy, EffectiveField, LLG, driver, and Simulation tests
  pass under DOLFINx.
- [ ] The core smoke advances to `1e-12 s` adaptively.
- [ ] Magnetization norms and scientific reference values meet declared
  tolerances.
- [ ] Unsupported features fail by name, not through incidental import errors.
- [ ] The gate checks that no test rewrites tracked source files.
- [ ] Documentation lists the exact supported API and deferred surfaces.

This completes the first core source port. It is not full Finmag completion.

## Task 11: Port FK demag

- [ ] Separate the array-only LLG/BEM native bindings from legacy
  SWIG-DOLFIN mesh converters and the `-ldolfin` link dependency.
- [ ] Rebuild and test the resulting native surface on Python 3.12 without
  installing legacy DOLFIN.
- [ ] Validate DOLFINx boundary mesh/marker extraction under `dev/dolfinx`.
- [ ] Validate the compiled `compute_bem_fk_from_arrays` entry point on the
  Python-3.12 toolchain.
- [ ] Port the existing FK demag modules directly in `src/finmag`.
- [ ] Compare BEM matrices, fields, and energies with coordinate-ordered legacy
  references.
- [ ] Run a DOLFINx `barmini`-class workflow with compiled FK demag.

PBC/treecode demag remains a separate native slice.

## Task 12: Port restart and required output

- [ ] Define restart ownership and mesh/parameter metadata before implementation.
- [ ] Store owned values in a stable coordinate-aware format.
- [ ] Round-trip magnetization, time, material parameters, and interactions.
- [ ] Port NDT output needed by accepted workflows.
- [ ] Port required VTK/XDMF output with explicit read/write capabilities.
- [ ] Add scheduler integration only after the underlying outputs work directly.

## Later Value-Driven Slices

Select independently after the core, FK demag, restart, and output gates:

1. DMI and cubic anisotropy production classes;
2. field-valued materials and regions;
3. native Sundials/CVODE;
4. normal modes;
5. LLB/SLLG and spin-transfer torque;
6. PBC/treecode demag;
7. external OOMMF, Nmag, and Magpar workflows.

## Definition of Migration Complete

- [ ] Accepted workflows import and run from `src/finmag` on DOLFINx.
- [ ] Compiled FK demag, restart, and required output are green.
- [ ] Scientific comparisons meet explicit tolerances.
- [ ] Unsupported historical behavior is documented and accepted.
- [ ] `dev/dolfinx` contains only useful probes and reference-generation tools.
- [ ] There is no duplicate `finmag` package and no final promotion/move diff.
