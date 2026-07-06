# DOLFINx M4 Prototype

This directory contains the isolated M4 DOLFINx prototype. It is intentionally
separate from `src/finmag`: the code here explores a reduced Finmag-on-DOLFINx
path without changing the green legacy FEniCS-2019 M2/M3 code. [Codex
gpt-5.5 high]

`porting_map.md` records how this exploration lane should be evaluated against
the existing Finmag API, data structures, and tests before any code is promoted
into `src/finmag`. [Codex gpt-5.5 high]

## Supported Scope

- DOLFINx 0.10.0 import and a tiny mesh/function smoke test.
- Vector-valued Lagrange magnetisation fields on simple DOLFINx meshes.
- Constant magnetisation setup and nodal vector inspection.
- A narrow DOLFINx-backed `Field` compatibility adapter for constants,
  callables, scalar/vector inspection, nodal values, and volume averages.
- Raw dof-array (`from_array`) and mesh-vertex-ordered
  (`get/set_with_ordered_numpy_array_xyz`) field value access, matching
  legacy `Field`'s coordinate/value ordering methods. Coordinate matching is
  done explicitly (not by assuming dof index equals vertex index) and is
  only supported for one-dof-per-vertex spaces (e.g. not `DG0`). There is no
  DOLFINx equivalent of legacy's component-blocked `"xxx"` ordering, so that
  variant is intentionally not provided. [GitHub Copilot / Claude Sonnet 5]
- Energy assembly for exchange, constant-field Zeeman, and constant-axis
  uniaxial anisotropy.
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
  unit-square path, including prototype time.

## Explicit Non-Scope

- This is not a drop-in replacement for `finmag.Simulation`.
- `PrototypeSimulation` is not legacy `finmag.Simulation` compatibility.
- No demagnetising field implementation is provided.
- Non-zero DMI is only usable on a 3D mesh; the current example/wrapper
  meshes are 2D, so DMI stays at its inert default (`0.0`) there in practice.
- `explicit_llg_step` only precesses/damps toward a fixed applied field
  argument; it does not derive an effective field from the full energy
  functional (exchange/anisotropy/DMI/cubic-anisotropy gradients are not fed
  back into the stepper). Adding a non-Zeeman energy term therefore changes
  the reported `energy_terms()`/`total_energy()` value, but is not guaranteed
  to still produce a monotonic energy decrease under `relax()`/
  `run_relaxation_example()`. [GitHub Copilot / Claude Sonnet 5]
- No general Finmag restart format, legacy NDT tables, VTK/XDMF output, or full
  scheduler-driven data I/O is provided.
- No adaptive or production-grade time integrator is provided; `run_until(...)`
  is a bounded explicit-step probe only.
- No finite-element projection of general effective fields is provided.
- No compatibility guarantee is made for legacy Finmag public APIs.
- The dataclasses are a prototype API sketch, not a stable public API.
- No attempt is made here to port rarely used historical features.

## M4 Completion Status

The reduced M4 prototype criteria are now represented in this directory:

- one end-to-end DOLFINx example runs through `dolfinx-example`;
- supported scope is documented in this file;
- unsupported subsystems are listed explicitly above;
- CI-facing verification is available through `dev/bin/verify-dolfinx-m4`.

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

The wrapper runs the import probe, the smoke probe, all tests below
`dev/dolfinx`, the JSON-output relaxation example, and the restart-state
round-trip example. The examples write
`/tmp/finmag-dolfinx-relaxation-summary.json` and
`/tmp/finmag-dolfinx-restart-state.json` by default. [Codex gpt-5.5 high]
