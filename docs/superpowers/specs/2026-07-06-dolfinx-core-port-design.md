# Finmag DOLFINx Core Port Design

## Context

Finmag's production package under `src/finmag` is a legacy FEniCS/DOLFIN
implementation. It remains the behavioral and scientific reference for the
DOLFINx migration, but it is not a runtime that the `dolfinx-port` branch must
continue to support.

The code under `dev/dolfinx` is an exploration lane. It establishes useful
DOLFINx mechanics and numerical evidence, but its `PrototypeSimulation`,
flat parameter dataclass, JSON contracts, and free-function architecture are
not the target production design.

The production port will replace the legacy implementation under `src/finmag`
with DOLFINx code while preserving the original Finmag philosophy, module
layout, public API, interaction model, and scientifically relevant behavior as
closely as practical.

## Goal

Deliver the first production-shaped DOLFINx Finmag slice with this workflow:

```python
import finmag
from finmag import Simulation
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman

sim = Simulation(mesh, Ms=8.6e5, unit_length=1e-9)
sim.set_m((1, 0, 1))
sim.add(Exchange(13e-12))
sim.add(Zeeman((0, 0, 1e5)))
sim.run_until(1e-12)
```

The workflow must use DOLFINx functions and forms, the complete configured
effective field, physical Finmag units, and an adaptive time integrator.

## Migration Rules

1. Treat legacy `src/finmag` code and trustworthy legacy tests as the
   specification.
2. Port classes and modules in place rather than promoting the prototype API.
3. Preserve public names, constructor semantics, properties, interaction
   lifecycle, and errors unless DOLFINx makes a behavior impractical.
4. Document intentional compatibility differences in tests and migration
   notes.
5. Port capabilities in scientific-value order rather than attempting to
   revive every historical subsystem.
6. Keep compiled native FK BEM construction as the future demagnetising-field
   baseline; do not replace it with the NumPy Magpar reference implementation.

## Initial Corrections

Before porting production modules:

1. Pin the isolated environment to `fenics-dolfinx = "0.10.*"`. The current
   wildcard can move the environment to a new incompatible release whenever
   the lock file is refreshed. DOLFINx 0.10 is already installed and verified,
   and upgrading is not required for the first compatibility slice.
2. Update project documentation to state that `src/finmag` is the legacy
   reference being replaced on `dolfinx-port`, not a second backend that must
   remain runnable.
3. Freeze `dev/dolfinx` as exploration evidence. Its known restart and MPI
   defects are not production blockers because its architecture will not be
   promoted.
4. Add the successor DOLFINx verification gate and prevent legacy FEniCS gates
   from defining success for the DOLFINx branch.

## Production Architecture

The runtime dependency flow remains the same as legacy Finmag:

```text
Simulation
  -> Field objects for m and Ms
  -> LLG
       -> EffectiveField
            -> named interaction objects
       -> integration RHS
  -> selected time integrator
```

### `finmag.Field`

`Field` wraps a `dolfinx.fem.Function` and its function space. The first slice
preserves:

- construction from constants and callables;
- `set`, `from_array`, `from_field`, and `from_function`;
- scalar/vector inspection;
- raw local-array access;
- mesh-vertex-ordered `xyz` array access;
- nodal vector normalisation;
- finite-element volume averages;
- `mesh`, `mesh_dim`, `value_dim`, and access to the wrapped function.

DOLFINx blocked vector layout differs from legacy DOLFIN component-blocked
layout. The public `xyz` methods preserve their legacy meaning through explicit
coordinate-based permutation. The historical `xxx` layout is retained only
where a deterministic compatibility conversion is needed by an existing
driver or test.

### Energy interactions

The first production interactions are:

- `Exchange(A, method=..., name=...)`;
- `Zeeman(H, name=..., **kwargs)`;
- `UniaxialAnisotropy(K1, axis, K2=..., method=..., name=..., assemble=...)`.

Each interaction keeps the legacy lifecycle:

```text
construct -> Simulation.add -> interaction.setup(m, Ms, unit_length)
          -> compute_field / compute_energy / average_field
```

`EnergyBase` owns common DOLFINx form differentiation, lumped nodal-volume
division, energy assembly, and MPI scalar reduction. The supported first-slice
calculation is the legacy box-assembly method. Historical method names may be
accepted for constructor compatibility, but unsupported implementations must
raise a precise error rather than silently selecting different mathematics.

### `EffectiveField`

`EffectiveField` preserves the legacy named interaction registry:

- unique-name enforcement;
- `add`, `get`, `exists`, `all`, and `remove`;
- summation of interaction fields;
- summation of interaction energies;
- optional time-update callbacks.

It receives the shared magnetisation and saturation-magnetisation `Field`
objects. Adding an interaction binds it to those fields through `setup`.

### `LLG`

`LLG` preserves the original role and relevant properties:

- owns magnetisation, saturation magnetisation, damping, gyromagnetic ratio,
  and effective-field state;
- supports `set_m`, `solve`, and `solve_for`;
- evaluates the Gilbert-form LLG equation using the complete effective field;
- uses `gamma = 2.210173e5 m/(A s)` and fields in `A/m`;
- returns derivatives in `1/s`;
- normalises the magnetisation after accepted integration steps.

The first slice supports scalar `Ms` and scalar `alpha`. Their internal
representation will not prevent a later extension to field-valued material
parameters.

### Drivers

The legacy driver abstraction remains:

```python
llg_integrator(llg, m0, backend=...)
```

The first production slice ports the SciPy VODE/BDF driver because it already
matches the legacy interface and provides adaptive physical-time integration
without coupling the FEM port to a simultaneous native CVODE build.

The native Sundials/CVODE driver is the next independent driver slice. Until
then:

- `integrator_backend="scipy"` is supported;
- requesting `"sundials"` raises an explicit availability error;
- `Simulation` defaults to `"scipy"` on the DOLFINx successor branch;
- the public backend-selection interface remains compatible.

### `Simulation`

`Simulation` remains the user-facing orchestrator. The first slice preserves:

- construction from a DOLFINx mesh, `Ms`, `unit_length`, name, kernel, and
  integrator-backend controls;
- `set_m` and the `m`, `m_field`, `m_average`, `Ms`, `alpha`, `gamma`, `t`,
  and `dmdt` surfaces needed by the core workflow;
- interaction `add`, lookup, listing, removal, and total-energy methods;
- `effective_field`, `create_integrator`, `set_tol`, `advance_time`,
  `run_until`, `reinit_integrator`, and `reset_time`;
- the `sim_with` convenience constructor for supported first-slice
  interactions.

Demag-related `sim_with` arguments remain accepted only when they can fail with
a clear unsupported-feature error. They must not silently omit demag.

## Data Flow

1. `Simulation` creates DOLFINx scalar and three-component Lagrange spaces.
2. It creates `Field` objects for magnetisation and material state.
3. `Simulation.add(interaction)` delegates to `EffectiveField.add`.
4. `EffectiveField.add` calls `interaction.setup(m, Ms, unit_length)`.
5. The driver asks `LLG.solve_for(y, t)` for an ODE derivative.
6. `LLG` writes `y` into the shared magnetisation field.
7. `EffectiveField.compute(t)` updates time-dependent interactions and sums
   every configured interaction field.
8. `LLG` computes the Gilbert RHS from `m`, total `H_eff`, `alpha`, and
   `gamma`.
9. The adaptive driver advances to the requested physical time.
10. The accepted state is copied back to the shared `Field`, normalised, and
    exposed through `Simulation`.

No production `relax` or `run_until` path may use only the applied field while
reporting energies from other interactions.

## Parallel Scope

The first time-integration slice is serial, matching the practical limitation
of legacy Finmag's time integration. `Simulation` requires a communicator of
size one and raises `NotImplementedError` otherwise.

Form assembly and field helpers should still use correct owned/ghost semantics
so that later MPI support does not inherit the prototype's double-counting and
zero-ghost-volume defects.

## Error Handling

- Duplicate interaction names raise `ValueError`.
- Unknown interaction lookup/removal uses a dedicated compatibility error.
- Invalid field dimensions, zero-vector normalisation, invalid material
  constants, and non-positive unit lengths fail before form assembly.
- Integrating backwards in time raises `RuntimeError`.
- Unsupported kernels, drivers, demag, PBC, stochastic dynamics, spin-transfer
  torque, and multi-rank stepping raise explicit errors.
- Optional native modules fail when the feature is requested, not during base
  `import finmag`.

## Testing Strategy

All production behavior is implemented test-first under `src/finmag`. Tests
are ported from or directly compared with the legacy suite where trustworthy.

The first gate covers:

1. `import finmag` in the DOLFINx environment.
2. `Field` constants, callables, array ordering, normalisation, and volume
   averages.
3. Analytic Exchange, Zeeman, and uniaxial-anisotropy energy and field values.
4. `EnergyBase` effective-field sign, units, and nodal-volume scaling.
5. Interaction add/get/list/remove and duplicate-name behavior.
6. Total effective-field and total-energy summation.
7. LLG direction, physical scaling, and unit-length preservation.
8. SciPy driver time advancement, backward-time rejection, and
   reinitialisation after magnetisation changes.
9. `Simulation` construction, properties, interaction management, and
   `run_until`.
10. The accepted end-to-end user workflow.

Prototype tests remain available as exploratory evidence, but passing them is
not a substitute for the production gate.

## CI and Verification

The active DOLFINx workflow will:

- create the pinned DOLFINx 0.10 environment;
- import the production `finmag` package from `src`;
- run the production DOLFINx core tests;
- run the end-to-end core simulation smoke test.

Legacy FEniCS workflows and scripts remain in repository history or as manual
reference tools, but they do not gate the DOLFINx successor branch after the
corresponding production modules are replaced.

## Deferred Slices

The following are intentionally outside this first implementation:

1. native Sundials/CVODE integration;
2. FK demagnetising field using compiled array-based BEM construction;
3. restart persistence and legacy restart compatibility;
4. scheduler, NDT, VTK/XDMF, and field checkpoint output;
5. field-valued material parameters and regions;
6. DMI and cubic anisotropy production interaction classes;
7. PBC and treecode demag;
8. normal modes, LLB/SLLG, and spin-transfer torque;
9. multi-rank time integration;
10. external OOMMF, Nmag, and Magpar comparisons.

These slices follow the roadmap order: core simulation and common energies,
demag, restart/data I/O, normal modes, additional dynamics, then external
validation tooling.

## Acceptance Criteria

The slice is complete when:

- DOLFINx is constrained to `0.10.*`;
- `PYTHONPATH=src python -c "import finmag"` succeeds in the DOLFINx
  environment without importing legacy `dolfin`;
- the accepted example advances to `1e-12 s` using all configured interaction
  fields;
- magnetisation remains unit length within the test tolerance;
- interaction energies and effective fields match analytic or trusted legacy
  references;
- unsupported features fail explicitly;
- the production DOLFINx core test and smoke gates pass from a clean checkout.
