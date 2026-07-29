> **ARCHIVED (2026-07-29).** Historical record of the porting process; statements reflect their writing date. Current truth: docs/README.md.

# Finmag DOLFINx Direct-Port Design

## Decision

Finmag will be ported to DOLFINx by editing the existing modules under
`src/finmag` directly, one reviewed capability at a time.

`dev/dolfinx` remains a laboratory for small, disposable DOLFINx mechanics
probes. It must not contain a second `finmag` package, and production code must
not be developed there and copied into `src` later.

This replaces the earlier proposal to build `dev/dolfinx/finmag` and promote it
as a final migration operation.

## Why Direct Porting Is Better Here

The staged-package proposal would create two implementations with the same
public API, duplicate package structure and tests, invite drift, and finish with
a large delete/move diff that obscures which legacy behavior changed. It would
also make the exploratory `PrototypeSimulation` architecture more likely to
leak into production even though it does not match Finmag's design.

Direct in-place porting gives each accepted behavior one implementation and one
history. Reviewers can compare each changed production module with its legacy
version and the associated tests without also reviewing a package relocation.

## Current Baselines

- The immutable legacy oracle is commit
  `ba9280934e188d7f3800e7b9865e70a9422f7687` (`pixi`), which contains the
  Python-3/FEniCS-2019 implementation and its recorded M3 gate.
- The active DOLFINx exploration environment resolves to DOLFINx 0.10.0 on
  Python 3.12.
- The modules directly under `dev/dolfinx` are evidence about DOLFINx mechanics,
  not candidate production modules.
- The current prototype has known restart and multi-rank ownership defects.
  Those defects are reasons not to copy it, not tasks that must be fixed before
  the production port starts.

## Migration Rules

1. Preserve the existing `src/finmag` module names, public classes, constructor
   semantics, and scientifically relevant behavior where practical.
2. Use `dev/dolfinx` only to answer a bounded question that must be understood
   before editing a particular production module.
3. Once a probe is validated, implement the behavior directly in the matching
   `src/finmag` module. Re-express it in the production abstraction; do not copy
   prototype files or APIs wholesale.
4. Do not introduce a general `dolfin`/DOLFINx compatibility facade. The APIs
   differ too substantially, and temporary dual-backend branches would enlarge
   both the implementation and its final cleanup.
5. After the first DOLFINx-only source slice, compare against the pinned legacy
   oracle in a separate process or checkout. Do not keep the edited source
   simultaneously runnable on both FEM stacks unless a change is naturally
   backend-neutral.
6. Port existing source tests in place. Add new tests only for DOLFINx-specific
   semantics or previously untested scientific contracts.
7. Prefer analytic references. When an analytic result is unavailable, generate
   a small, versioned legacy reference fixture from the oracle.
8. Reference fields by coordinates and values, not raw legacy dof ordering.
   Store only owned DOLFINx dofs in parallel comparisons.
9. One source slice should change one coherent capability. Avoid unrelated
   cleanup, formatting, renaming, or revival of historical features.
10. Unsupported behavior must fail explicitly when requested. It must never be
    silently omitted from a simulation.

## Validation Model

### Legacy oracle

The FEniCS-2019 source stops being an executable in-tree baseline as soon as a
DOLFINx-only module is ported. Preserve its value through:

- the pinned oracle commit above;
- the recorded M3 workflow and result;
- focused legacy commands run from a temporary checkout when a new reference is
  needed;
- small JSON or NPZ fixtures containing mesh definition, physical parameters,
  coordinate-ordered results, units, tolerances, and oracle commit metadata.

`dev/bin/run-legacy-oracle` implements the detached-worktree command boundary.
The versioned JSON contract, coordinate-ordering rule, and analytic-evidence
decision are defined in
[`legacy-oracle-fixtures.md`](legacy-oracle-fixtures.md). [Codex GPT-5]

The full immutable oracle does not need to run for every DOLFINx source edit.
Each source slice instead runs its focused differential or analytic contract.

### DOLFINx probes

A dev probe should test one uncertain mechanism, for example:

- owned and ghost dof semantics;
- coordinate/value permutations;
- UFL differentiation and lumped-volume assembly;
- extraction of boundary arrays for native FK BEM;
- DOLFINx-native function output or restart primitives.

A probe is complete when the question has an executable answer. Product API,
orchestration, long-lived state, and duplicate interaction classes do not belong
in the probe lane.

### Source gates

Every direct source slice must pass:

1. focused tests for the edited production module in the DOLFINx environment;
2. an analytic or pinned-oracle scientific comparison where applicable;
3. the still-relevant low-level dev probe;
4. at least one serial test and, for ownership-sensitive FEM code, a two-rank
   test;
5. `git diff --check` and a clean-worktree check that ignores only declared test
   artifacts.

The completed M4 suite under `dev/dolfinx` is frozen as a prototype regression
witness. It runs in a DOLFINx `0.10.*`/Python 3.12 environment with SciPy and
has its own `dolfinx-prototype-pytest` task. Direct production work uses
separate `dolfinx-src-*` tasks, and the aggregate prototype verifier rejects
tracked-file mutations. A machine-readable version report records DOLFINx,
Python, NumPy, SciPy, PETSc, and MPI before the gate runs. [Codex GPT-5]

## First Permanent Source Seam

Finmag currently imports most of the application eagerly from
`src/finmag/__init__.py`. Importing any submodule therefore imports `Simulation`,
demag, utilities, native modules, and legacy `dolfin`. That prevents a focused
in-place port: even testing `finmag.field` requires the rest of the old stack.
The scope is substantial: 64 non-test Python modules currently import legacy
`dolfin` directly, so keeping every historical module live during each source
slice would defeat the goal of reviewable changes.

The first source slice should make package exports explicit and lazy. This is a
permanent import architecture improvement, not a temporary backend selector.
It must:

- make plain `import finmag` free of FEM and native-build side effects;
- preserve the intended top-level names such as `Simulation`, `sim_with`, and
  `Field` through lazy resolution;
- allow a ported submodule to be tested without importing unported subsystems;
- make optional or unported features fail only when requested;
- stop import-time validation from rewriting tracked version files.

The energy package needs the same treatment because its current `__init__`
eagerly imports demag, DMI, thermal, and other interactions.

The reviewed top-level compatibility inventory is `Simulation`, `sim_with`,
`Field`, `MacroGeometry`, `NormalModeSimulation`, `normal_mode_simulation`,
`set_logging_level`, `configuration`, `versions`, `example`, `energies`,
`timings_report`, `__version__`, `logger`, and `logging`. The last two remain
public because existing source uses them; command-line parsing, signal-handler
registration, version-report locals, and imported implementation modules were
wildcard accidents and are not public exports.

The separate `finmag.energies` inventory is `Demag`, `Demag2D`,
`MacroGeometry`, `EnergyBase`, `Exchange`, `UniaxialAnisotropy`,
`CubicAnisotropy`, `Zeeman`, `TimeZeeman`, `DiscreteTimeZeeman`,
`OscillatingZeeman`, `TimeZeemanPython`, `DMI`, `DMI_interfacial`,
`ThinFilmDemag`, and `FixedEnergyDW`. These names resolve to their existing
modules on demand; they are not promoted to the top-level `finmag` namespace.
[Codex GPT-5.6]

## Direct Source Slices

### 1. `Field`

Port `src/finmag/field.py` in place. Preserve the field behaviors required by
the core simulation: constants and callables, raw and coordinate-ordered array
access, assignment from fields/functions, scalar/vector inspection, volume
averages, normalization, and access to the underlying DOLFINx function.

Owned/ghost handling and coordinate ordering require explicit two-rank tests.
Historical expression strings, point-measure arithmetic, plotting, or obsolete
HDF5 behavior may remain unsupported initially, but their public methods should
raise precise errors if retained.

### 2. Energy foundation and common interactions

Port `src/finmag/energies/energy_base.py`, then the existing Exchange, Zeeman,
and uniaxial-anisotropy modules. Keep the interaction lifecycle:

```text
construct -> Simulation.add -> setup(m, Ms, unit_length)
          -> compute_field / compute_energy / average_field
```

The first supported field calculation is the legacy box-assembly method.
Constructor values for unsupported legacy methods may be accepted only if use
raises a clear error; they must not silently choose different mathematics.

### 3. `EffectiveField`

Port the existing named interaction registry directly. Preserve unique names,
add/get/list/remove, time callbacks, total field, and total energy. Test it with
small interaction doubles as well as the ported common interactions.

### 4. Physical LLG core

Port the core of `src/finmag/physics/llg.py` around the real total effective
field. Preserve physical units, signs, `gamma`, scalar `Ms`/`alpha`, `set_m`,
`solve`, and `solve_for`. Compare the RHS with analytic macrospin cases and
coordinate-ordered legacy references.

Native Sundials, STT, thermal dynamics, and multi-rank time integration remain
separate slices. They must not complicate the first deterministic LLG port.

### 5. Adaptive driver

Reuse and minimally adapt the existing SciPy driver because it already matches
Finmag's stateful `advance_time` interface and avoids coupling the FEM port to a
native CVODE build. Add real reinitialization and backward-time checks. Do not
redesign the integrator API during the FEM migration.

### 6. Core `Simulation`

Port `src/finmag/sim/sim.py` directly after its dependencies are ready. Keep the
core orchestration and public properties, but do not import unported scheduler,
output, demag, PBC, stochastic, STT, or visualization modules at module import
time.

In the touched module, dead legacy branches should be removed or changed to
explicit unsupported-feature errors. Untouched long-tail modules can remain in
the tree outside the active import graph until their own slices are selected.

### 7. Demag, restart, and output

Port these only after the core workflow is green:

1. native array-based FK demag and a `barmini`-class workflow;
2. restart persistence with mesh, parameters, time, and owned field values;
3. NDT and required VTK/XDMF output;
4. scheduler integration.

Normal modes, LLB/SLLG, STT, PBC/treecode demag, and external comparison tools
remain value-driven follow-up slices.

The existing `finmag.native.llg` binary cannot simply be rebuilt unchanged for
DOLFINx. Its Makefile links `libdolfin`, its Python module registers legacy
SWIG-DOLFIN converters, and its array-based LLG/BEM entry points share a binary
with legacy mesh bindings. Before FK demag is ported, split or condition the
native binding so the array-only surface builds without legacy DOLFIN while
preserving the production Python API where practical.

## Reviewability

Each source commit should contain:

- the direct edit to the existing production module;
- the smallest corresponding in-place test changes;
- at most one new legacy reference fixture;
- a short status update naming supported and unsupported behavior.

Do not combine a dev probe, several production layers, broad test cleanup, and
documentation reorganization in one commit. Dev evidence may land first; the
corresponding production port should then be a separate, easy-to-review diff.

## Completion

There is no final promotion, package copy, or directory move.

The DOLFINx migration is complete when the agreed scientific workflow matrix
runs from `src/finmag`, including compiled FK demag, restart, and required
output; differential results satisfy their declared tolerances; unsupported
historical features are documented; and `dev/dolfinx` contains only useful
mechanics witnesses rather than a second implementation.
