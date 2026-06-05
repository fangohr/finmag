# DOLFINx M4 Prototype

This directory contains the isolated M4 DOLFINx prototype. It is intentionally
separate from `src/finmag`: the code here explores a reduced Finmag-on-DOLFINx
path without changing the green legacy FEniCS-2019 M2/M3 code. [Codex
gpt-5.5 high]

## Supported Scope

- DOLFINx 0.10.0 import and a tiny mesh/function smoke test.
- Vector-valued Lagrange magnetisation fields on simple DOLFINx meshes.
- Constant magnetisation setup and nodal vector inspection.
- Energy assembly for exchange, constant-field Zeeman, and constant-axis
  uniaxial anisotropy.
- Explicit normalized LLG stepping in a constant effective field.
- A small end-to-end relaxation example with JSON summary output.
- A hand-written JSON summary validator with `schema_version = 1`.
- Small dataclasses for relaxation parameters and JSON-compatible results.
- Early validation for the reduced relaxation-example parameter set.

## Explicit Non-Scope

- This is not a drop-in replacement for `finmag.Simulation`.
- No demagnetising field implementation is provided.
- No restart format, VTK/XDMF output, or full data I/O is provided.
- No adaptive or production-grade time integrator is provided.
- No finite-element projection of general effective fields is provided.
- No compatibility guarantee is made for legacy Finmag public APIs.
- The dataclasses are a prototype API sketch, not a stable public API.
- No attempt is made here to port rarely used historical features.

## Verification

Run the isolated M4 checks with:

```bash
PIXI_HOME=/tmp/pixi-cache \
XDG_CACHE_HOME=/tmp/pixi-cache \
PIXI_CACHE_DIR=/tmp/pixi-cache \
dev/bin/verify-dolfinx-m4
```

The wrapper runs the import probe, the smoke probe, all tests below
`dev/dolfinx`, and the JSON-output relaxation example. The example writes
`/tmp/finmag-dolfinx-relaxation-summary.json` by default. [Codex gpt-5.5 high]
