> **ARCHIVED (2026-07-29).** Historical record of the porting process; statements reflect their writing date. Current truth: docs/README.md.

# DOLFINx M4 Prototype

This directory contains the frozen, isolated M4 DOLFINx prototype. It is intentionally
separate from `src/finmag`: the code here explores a reduced Finmag-on-DOLFINx
path without changing the green legacy FEniCS-2019 M2/M3 code. [Codex
gpt-5.5 high]

The M4 suite is now a regression witness, not the place where the production
port grows. New port tests target the existing modules under `src/finmag`
through separately named `dolfinx-src-*` tasks. [Codex GPT-5]

Task 4 has now replaced `src/finmag/field.py` directly with its DOLFINx
implementation. The adapter in this directory was extended only to settle the
pre-source ownership/ordering questions and is frozen at that accepted
behavioral contract; it was not copied into production and does not own the
active API. Production Field checks use `dolfinx-src-field-pytest` and
`dolfinx-src-field-mpi`. [Codex GPT-5.6]

Task 5 uses `energy_box_probe.py` as a bounded pre-source witness for energy
assembly. It is not an energy package to copy into `src`: it records the exact
DOLFINx 0.10 reverse-add/forward-scatter ordering required for complete lumped
nodal volumes and derivative vectors that was validated before the existing
production energy modules were edited directly. [Codex GPT-5.6]

The accepted probe has now been implemented directly in
`src/finmag/energies`: `EnergyBase`, static `Zeeman`, scalar-`A` `Exchange`,
and constant-`K1`/`K2` `UniaxialAnisotropy` are DOLFINx-native. Production
checks use `dolfinx-src-energies-pytest` and `dolfinx-src-energies-mpi`; this
directory remains evidence, not the implementation source. This first box
slice is restricted to blocked three-component CG1 magnetisation spaces.
[Codex GPT-5.6]

`porting_map.md` records how this exploration lane should be evaluated against
the existing Finmag API, data structures, and tests before the matching
production module is edited directly under `src/finmag`.

This directory must not grow a second `finmag` package. A successful probe is
evidence for a focused in-place source change; prototype modules are not copied
or moved into production.

## Supported Scope

- DOLFINx 0.10.0 import and a tiny mesh/function smoke test.
- Vector-valued Lagrange magnetisation fields on simple DOLFINx meshes.
- Constant magnetisation setup and nodal vector inspection.
- A narrow DOLFINx-backed `Field` compatibility adapter for constants,
  callables, scalar/vector inspection, nodal values, and volume averages.
- `set()`/`from_field()`/`from_function()` dispatch on another `DOLFINxField`
  or `fem.Function`, mirroring legacy `Field.set`'s type-based dispatch
  (`from_field` interpolates between different function spaces; 
  `from_function` requires a matching space). [GitHub Copilot / Claude
  Sonnet 5]
- `set_random_values()`, `is_constant()`/`as_constant()`, `normalise()`
  (per-node unit-length nodal normalisation), `coords_and_values()`, and
  `allclose()`, matching their legacy `Field` counterparts (with `allclose`
  and `normalise` using different implementation approaches internally, but
  matching outcomes/semantics; see code docstrings for the differences).
  [GitHub Copilot / Claude Sonnet 5]
- `save_pvd()`/`close_pvd()` (via `dolfinx.io.VTKFile`) and
  `save_xdmf()`/`close_xdmf()` (via `dolfinx.io.XDMFFile`) for basic,
  write-only, Paraview-viewable file output. These use DOLFINx-native
  formats, not the external `dolfinh5tools` package/format that legacy
  `Field.save_hdf5` depends on, and there is no read-back/round-trip support
  in this DOLFINx version (0.10.0 has no `XDMFFile.read_function`).
  [GitHub Copilot / Claude Sonnet 5]
- Raw dof-array (`from_array`) and mesh-vertex-ordered
  (`get/set_with_ordered_numpy_array_xyz`) field value access, matching
  legacy `Field`'s coordinate/value ordering methods. Coordinate matching is
  done explicitly (not by assuming dof index equals vertex index) and is
  only supported for one-dof-per-vertex spaces (e.g. not `DG0`). The legacy
  flat component-blocked `"xxx"` view is an explicit transpose of the flat
  per-node `"xyz"` view; it is retained for driver compatibility rather than
  treated as a DOLFINx-native storage layout. Rank-local raw and ordered views
  contain owned dofs only and refresh ghosts after mutation. [Codex GPT-5.6]
- Energy assembly for exchange, constant-field Zeeman, and constant-axis
  uniaxial anisotropy.
- `energy_box_probe.py` checks the production-port box contract in serial and
  on two ranks: assembled vectors reverse-add ghost contributions and then
  forward-refresh completed owner values before owned coefficients are read.
  The probe verifies scalar/vector lumped volumes, `H_eff == H` for Zeeman,
  the `-1/(mu0*Ms)` sign, constant `K1`/`K2` anisotropy, exchange
  `unit_length**-2` field scaling, and analytical total energies. [Codex
  GPT-5.6]
- Bulk (3D, T-symmetry) DMI energy assembly, matching the legacy
  `dmi_type='auto'` 3D case (`D * inner(m, curl(m))`). Interfacial and
  1D/2D DMI variants are not covered yet. [GitHub Copilot / Claude Sonnet 5]
- Cubic anisotropy energy assembly for constant `K1`/`K2`/`K3` and constant
  axes, matching legacy `finmag.energies.cubic_anisotropy.CubicAnisotropy`'s
  analytic form; checked directly against that module's own reference test
  values. Spatially varying cubic-anisotropy fields are not covered yet.
  [GitHub Copilot / Claude Sonnet 5]
- DMI and cubic anisotropy are wired into `RelaxationParameters` and
  `PrototypeSimulation.energy_terms()`, both defaulting to inert values
  (`dmi_constant=0.0`, `cubic_anisotropy_K1=K2=K3=0.0`). Since `dmi_energy`
  only supports 3D meshes and both example/wrapper meshes are 2D unit
  squares, a non-zero `dmi_constant` there raises explicitly rather than
  silently doing the wrong thing; this is a tested boundary, not a bug.
  [GitHub Copilot / Claude Sonnet 5]
- Explicit normalized LLG stepping in a constant effective field.
- ``nodal_volume()`` (lumped/"box" nodal volumes) and
  ``effective_field_values()``, computing the true energy-derived effective
  field ``H_eff = -1/(mu0*Ms) * dE/dm`` via the same finite-element "box
  method" used by legacy ``finmag.energies.energy_base.EnergyBase``
  (assemble the weak-form derivative, divide by lumped nodal volume).
  Validated directly: for a uniform applied field with all other terms off,
  this recovers ``H_eff == field`` exactly. [GitHub Copilot / Claude Sonnet 5]
- ``effective_field_llg_step()``, an LLG stepper driven by that computed
  effective field (recomputed from the current magnetisation on every
  call), so exchange/anisotropy/DMI/cubic-anisotropy contributions actually
  drive the dynamics rather than only changing the reported energy. This
  fixes the limitation flagged when DMI/cubic anisotropy were first wired
  into the simulation wrappers. ``PrototypeSimulation.step()`` (and
  ``run_until``/``relax``/``relaxation_trace``/``relaxation_summary`` and
  their JSON-writing counterparts) accept an opt-in
  ``use_effective_field=True`` to use it; the default remains the original
  fixed-field stepper for backward compatibility. [GitHub Copilot / Claude
  Sonnet 5]
- Prototype simulation time tracking and a narrow `run_until(...)` probe.
- A small end-to-end relaxation example with JSON summary output.
- A small restart-state round-trip example with JSON output.
- A hand-written JSON summary validator with `schema_version = 1`.
- Small dataclasses for relaxation parameters and JSON-compatible results.
- Early validation for the reduced relaxation-example parameter set.
- A reduced `PrototypeSimulation` wrapper for the first M5 core-path sketch.
- JSON-compatible state summaries from the reduced simulation wrapper.
- A JSON-compatible relaxation summary from the reduced simulation wrapper.
- JSON-compatible per-step relaxation traces from the reduced simulation
  wrapper, using absolute prototype simulation times.
- JSON summary writing from the reduced simulation wrapper.
- Narrow JSON restart-state round trips for the reduced `PrototypeSimulation`
  unit-square path, including prototype time. These tests use default
  DMI/cubic-anisotropy parameters; the current reader drops non-default values
  for those newer fields and must not be treated as a production restart
  implementation.

## Explicit Non-Scope

- This is not a drop-in replacement for `finmag.Simulation`.
- `PrototypeSimulation` is not legacy `finmag.Simulation` compatibility.
- No demagnetising field implementation is provided.
- The `Field` adapter does not implement legacy's `from_expression`
  (DOLFINx has no `dolfin.Expression`/`UserExpression` equivalent; use a
  Python callable with `set()`/`f.interpolate()` instead), the point-measure
  arithmetic operators (`__add__`/`__mul__`/`__div__`/`cross`/`dot`), Paraview
  plotting, `get_spherical()`, or `dolfinh5tools`-format HDF5.
  [GitHub Copilot / Claude Sonnet 5]
- Non-zero DMI is only usable on a 3D mesh; the current example/wrapper
  meshes are 2D, so DMI stays at its inert default (`0.0`) there in practice.
- `explicit_llg_step` (the default stepper) only precesses/damps toward a
  fixed applied field argument; it does not derive an effective field from
  the full energy functional. `effective_field_llg_step` (and
  `PrototypeSimulation.step(..., use_effective_field=True)`) now provide the
  fix: exchange/anisotropy/DMI/cubic-anisotropy contributions genuinely
  drive the dynamics through the box-method effective field, not just the
  reported energy. The default remains the fixed-field stepper for backward
  compatibility; the energy-decrease guarantee under `relax()`/
  `run_relaxation_example()` still only holds for the default fixed-field
  path (or the effective-field path in genuinely energy-minimising regimes),
  not for arbitrary parameter/stepper combinations. [GitHub Copilot / Claude
  Sonnet 5]
- No general Finmag restart format, legacy NDT tables, VTK/XDMF output, or full
  scheduler-driven data I/O is provided.
- `average_nodal_vector()` is an owned-only MPI nodal statistic. It is not a
  volume-weighted finite-element average and is retained only as prototype
  output evidence; production `Field.average()` has the FEM-integral contract.
- No adaptive or production-grade time integrator is provided; `run_until(...)`
  is a bounded explicit-step probe only.
- No finite-element projection of general effective fields is provided.
- Matrix/project/direct energy algorithms, spatially varying `A`, `K1`, `K2`,
  anisotropy axes or `Ms`, `DipolarField`, and all time-dependent Zeeman
  variants remain deferred from the direct Task 5 source slice.
- No compatibility guarantee is made for legacy Finmag public APIs.
- The dataclasses are a prototype API sketch, not a stable public API.
- No attempt is made here to port rarely used historical features.

## Frozen M4 Completion Status

The reduced M4 prototype criteria are now represented in this directory:

- one end-to-end DOLFINx example runs through `dolfinx-example`;
- supported scope is documented in this file;
- unsupported subsystems are listed explicitly above;
- CI-facing verification is available through `dev/bin/verify-dolfinx-m4`.
- the environment is pinned to DOLFINx `0.10.*`, and `dolfinx-versions` emits
  the Python, NumPy, SciPy, PETSc, and MPI runtime versions as JSON.

This does not make M4 a production port. It means the intended reduced
prototype has enough coverage and documentation to serve as a stable starting
point for M5 expansion work. [Codex gpt-5.5 high]

## Verification

Run the isolated M4 checks with:

```bash
PIXI_HOME=/tmp/pixi-cache \
XDG_CACHE_HOME=/tmp/pixi-cache \
PIXI_CACHE_DIR=/tmp/pixi-cache \
dev/bin/verify-dolfinx-m4
```

The wrapper records the runtime versions, runs the import and smoke probes, the
frozen `dolfinx-prototype-pytest` suite, the JSON-output relaxation example,
and the restart-state round-trip example. It also fails if any of those checks
mutates a tracked file. The examples write
`/tmp/finmag-dolfinx-relaxation-summary.json` and
`/tmp/finmag-dolfinx-restart-state.json` by default. [Codex gpt-5.5 high]
