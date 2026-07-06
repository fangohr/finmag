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
- Energy assembly for exchange, constant-field Zeeman, and constant-axis
  uniaxial anisotropy.
- Bulk (3D, T-symmetry) DMI energy assembly, matching the legacy
  `dmi_type='auto'` 3D case (`D * inner(m, curl(m))`). Interfacial and
  1D/2D DMI variants are not covered yet. [GitHub Copilot / Claude Sonnet 5]
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
