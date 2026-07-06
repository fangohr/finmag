# DOLFINx Compatibility Porting Map

This map links the isolated `dev/dolfinx` exploration lane back to the legacy
Finmag design that the real DOLFINx port should preserve where practical. It is
not an implementation plan for a clean-room rewrite. It is a checklist for
deciding when a DOLFINx prototype result is ready to be promoted into
`src/finmag`. [Codex gpt-5.5 high]

## Porting Rule

The production port should keep the existing Finmag public API, data model, and
test expectations unless DOLFINx or the modern software stack makes that
impractical. In particular, `PrototypeSimulation` is an exploration object; it
must not become the production API by accident. [Codex gpt-5.5 high]

## Legacy Surfaces To Preserve

- `src/finmag/sim/sim.py`: `Simulation` construction, `set_m`, interaction
  management, `run_until`, `restart`, scheduling, output helpers, and the
  existing names for core simulation state.
- `src/finmag/field.py`: the `Field` abstraction around scalar/vector function
  spaces, value setting, averaging, and file output.
- `src/finmag/energies/`: energy classes such as `Exchange`,
  `UniaxialAnisotropy`, `CubicAnisotropy`, `DMI`, `Zeeman`, and demag variants
  should keep their user-facing constructor semantics where possible.
- `src/finmag/drivers/`: time integration and relaxation should preserve the
  high-level behaviour of the existing Sundials/scipy driver interfaces, even
  if the DOLFINx implementation has to change internally.
- `src/finmag/util/fileio.py` and `src/finmag/sim/sim_savers.py`: NDT, field,
  restart, and VTK/XDMF-style output conventions are part of the compatibility
  contract, not just implementation details.
- Existing tests under `src/finmag/tests`, `src/finmag/energies/*_test.py`,
  `src/finmag/drivers/tests`, and comparison tests are the behavioural
  acceptance baseline for the port. [Codex gpt-5.5 high]

## Current Exploration Evidence

- Mesh/function basics: `dev/dolfinx/prototype.py` and
  `test_dolfinx_smoke.py` exercise DOLFINx import, mesh creation, scalar and
  vector function spaces, constant vector fields, and nodal inspection.
- Field compatibility: `field_adapter.py` probes the first legacy
  `finmag.field.Field` behaviours with DOLFINx, including scalar/vector
  distinction, constants, callables, nodal values, normalisation, and volume
  averages.
- Energy assembly: `exchange_energy`, `zeeman_energy`, and
  `uniaxial_anisotropy_energy` exercise representative form assembly, MPI
  reduction, unit-length scaling, and zero-coefficient edge cases.
- Bulk DMI: `dmi_energy` covers the legacy 3D `dmi_type='auto'` case
  (`D * inner(m, curl(m))`), with the matching `unit_length ** (dim - 1)`
  scaling convention and a 3D-mesh guard. Interfacial and 1D/2D DMI variants
  are still not covered. [GitHub Copilot / Claude Sonnet 5]
- Time stepping: `explicit_llg_step` gives a transparent nodal explicit step
  that tests DOLFINx function mutation and normalisation, but it is not a
  production driver.
- Simulation time: `PrototypeSimulation.time` and `run_until(...)` are a small
  compatibility-shaped probe for the legacy `Simulation.run_until` concept.
  They deliberately use the prototype explicit stepper and should not be
  treated as a production driver. [Codex gpt-5.5 high]
- Reduced output: `relaxation_example.py`, `PrototypeSimulation` summaries,
  and relaxation traces exercise JSON output contracts only. Trace records use
  absolute prototype simulation times, but they do not cover legacy NDT tables,
  VTK/XDMF, or scheduled output. [Codex gpt-5.5 high]
- Reduced restart: `restart_state` and `restart_example.py` prove a JSON
  round trip for a tiny unit-square prototype, preserving time, parameters, and
  nodal magnetisation values. This is not the legacy Finmag restart format.
  [Codex gpt-5.5 high]
- FK demag baseline: the FEniCS-2019/pixi M3 path still computes the
  Fredkin-Koehler BEM matrix through the compiled `finmag.native.llg`
  extension, using `compute_bem_fk` or the DOLFIN-2019-compatible
  `compute_bem_fk_from_arrays` entry point. The Python/NumPy Magpar code is a
  reference/comparison path, not the production FK BEM implementation. [Codex
  gpt-5.5 high]

## Promotion Criteria

Before moving any `dev/dolfinx` code into `src/finmag`, check that:

- the target legacy API surface is identified;
- the existing FEniCS-2019/Python-3 tests that define the expected behaviour
  are known;
- any DOLFINx-driven API or behaviour change is explicitly documented;
- the implementation fits the existing `Simulation`, `Field`, energy, driver,
  restart, or output abstraction;
- the new DOLFINx code has an executable witness or regression test;
- unsupported legacy behaviour is tracked as deliberate non-scope, not simply
  omitted. [Codex gpt-5.5 high]

## Near-Term Gaps

- The DOLFINx-backed `Field` adapter is only a narrow probe; it does not yet
  cover all legacy setters, coordinate/value ordering, HDF5/PVD output, or
  integration with `finmag.Simulation`.
- There is no DOLFINx-backed `finmag.Simulation` compatibility path yet.
- Demag is not covered by the current DOLFINx prototype lane. The production
  DOLFINx port should either reuse/port the array-based native FK BEM routines
  or provide a replacement with equivalent tests; a pure Python/NumPy BEM path
  may be acceptable for tiny references, but not as the assumed production
  implementation. [Codex gpt-5.5 high]
- DMI, cubic anisotropy, variable material parameters, regions, PBC, scheduler
  output, legacy restart files, and production integrators are not covered by
  the current prototype lane. Bulk 3D DMI is now a partial exception (see
  `dmi_energy`); interfacial/1D/2D DMI, cubic anisotropy, regions, PBC,
  scheduler output, legacy restart files, and production integrators remain
  uncovered. [GitHub Copilot / Claude Sonnet 5]
- PBC/treecode demag depends on the separate `finmag.native.treecode_bem`
  extension, which is still missing in the pixi path and should remain tracked
  separately from the FK BEM baseline. [Codex gpt-5.5 high]
- The current traces and restart states are useful witnesses, but they are not
  replacements for Finmag's NDT, VTK/XDMF, and restart conventions. [Codex
  gpt-5.5 high]
