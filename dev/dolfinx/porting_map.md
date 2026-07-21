# DOLFINx Compatibility Porting Map

This map links the isolated `dev/dolfinx` exploration lane back to the legacy
Finmag design that the real DOLFINx port should preserve where practical. It is
not an implementation plan for a clean-room rewrite. It is a checklist for
deciding when the evidence is strong enough to edit the matching production
module directly under `src/finmag`.

## Porting Rule

The production port should keep the existing Finmag public API, data model, and
test expectations unless DOLFINx or the modern software stack makes that
impractical. In particular, `PrototypeSimulation` is an exploration object; it
must not become the production API by accident. [Codex gpt-5.5 high]

Do not create `dev/dolfinx/finmag`, copy these prototype modules into `src`, or
defer review to a final package move. Use a probe to answer a bounded mechanics
question, then implement the accepted behavior once in the existing source
module with its tests.

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
  averages. It also covers raw dof-array access (`from_array`) and
  flat mesh-vertex-ordered access (`get/set_with_ordered_numpy_array_xyz`),
  matching dofs to vertices by coordinate rather than assumed index equality,
  with an explicit guard for spaces that don't have one dof per vertex (e.g.
  `DG0`). The legacy component-blocked `"xxx"` ordering is now provided as an
  explicit compatibility conversion rather than mistaken for a DOLFINx-native
  storage layout.
  It further covers `from_field`/`from_function` dispatch through `set()`,
  `set_random_values`, `is_constant`/`as_constant`, `normalise`,
  `coords_and_values`, `allclose`, `mesh_dim`, and basic write-only
  `save_pvd`/`save_xdmf` file output via DOLFINx-native `VTKFile`/`XDMFFile`
  (not the external `dolfinh5tools` format, and with no read-back support in
  this DOLFINx version).

  The exact array contract established before the direct source edit is:

  - legacy `as_array()`/`get_numpy_array_debug()` use the owned
    `GenericVector.get_local()` range; on two legacy ranks the scalar owned
    lengths are 6/10 for 16 global CG1 dofs, while each mesh partition exposes
    10 local vertices including ghosts;
  - DOLFINx `Function.x.array` contains owned blocks followed by ghosts. On the
    same 4-by-4 vertex grid the two ranks own 7/9 blocks, have 5/3 ghosts, and
    each stores 12 blocks (`36` scalar entries for block size three);
  - raw compatibility methods therefore expose or accept owned scalar entries
    only, and every mutation calls `scatter_forward()` to refresh ghosts;
  - `xyz` is the legacy flat per-node view
    `[x1, y1, z1, x2, y2, z2, ...]`; `coords_and_values()` alone returns the
    two-dimensional `(n_owned, components)` value table;
  - `xxx` is the flat component-blocked view
    `[x1, x2, ..., y1, y2, ..., z1, z2, ...]`, computed exactly as
    `xyz.reshape(-1, components).T.reshape(-1)`, with the transpose conversion
    reversed when setting;
  - rank-local ordered exports contain owned vertices only. Gathering their
    coordinate/value rows gives 16 unique global coordinates with analytic
    values; ghosts never appear twice. A future global ordered API must gather
    and coordinate-sort owned rows before component blocking, not concatenate
    per-rank `xxx` arrays. [Codex GPT-5.6]

- Field legacy witnesses: `field_test.py` directly checks construction and the
  underlying function/space, scalar/vector and value dimensions, mesh access,
  analytic scalar/vector averages, nodal normalization, coordinate/value
  shapes, and flat `xyz`/`xxx` values for dimensions one through four.
  `field_setters_test.py` checks constant, callable/expression, function,
  generic-vector, same-space Field, and cross-space Field assignment while
  retaining the wrapped function object. Raw-array ownership is defined by
  legacy `GenericVector.get_local()` and production consumers rather than a
  strong dedicated legacy unit test. [Codex GPT-5.6]

- Field `xxx` consumers: `physics/llg.py`, `drivers/llg_integrator.py`, the NEB
  implementations, and `Field.np` require component-blocked state. DOLFINx's
  interleaved blocked storage does not remove that API requirement. The SciPy
  driver currently initializes from raw `as_array()` but calls an LLG setter
  that interprets its argument as `xxx`; Task 8 should route both directions
  through the explicit state ordering rather than preserve this ambiguity.
  Multi-rank stepping remains outside the initial driver slice. [Codex GPT-5.6]
- Energy assembly: `exchange_energy`, `zeeman_energy`, and
  `uniaxial_anisotropy_energy` exercise representative form assembly, MPI
  reduction, unit-length scaling, and zero-coefficient edge cases.
- Task 5 box-method witness: `energy_box_probe.py` closes an ownership gap in
  the earlier prototype helper. A DOLFINx linear-form vector must call
  `scatter_reverse(InsertMode.add)` to accumulate contributions into owners and
  then `scatter_forward()` to refresh ghost copies before any ghost-inclusive
  inspection. Production interaction arrays remain owned-only, but this order
  ensures their owner values include every adjacent cell. The probe passes on
  a 16-vertex unit square with serial ownership 16/0 and two-rank ownership
  7/9 plus 5/3 ghosts. [Codex GPT-5.6]
- Task 5 legacy surface: `EnergyBase`, `Exchange`, and uniaxial anisotropy used
  to default to `box-matrix-petsc` and advertised NumPy-matrix, project, and
  direct paths. The DOLFINx source slice deliberately changes the default to
  the only supported `box-assemble` algorithm; every historical
  matrix/project/direct request raises `NotImplementedError` instead of being
  silently remapped. [Codex GPT-5.6]
- Task 5 import boundary: `finmag.energies` now bypasses the legacy `dolfin`
  bridge for `EnergyBase`, `Exchange`, static `Zeeman`, and
  `UniaxialAnisotropy`. The time-Zeeman names resolve to explicit deferred
  stubs without pretending that installing `dolfin` restores their removed
  implementation. Demag, DMI, cubic anisotropy, and the other untouched
  exports retain the real legacy dependency. The ported modules no longer
  import `finmag.util.meshes`, `finmag.util.helpers`, or `finmag.native`.
  [Codex GPT-5.6]
- Bulk DMI: `dmi_energy` covers the legacy 3D `dmi_type='auto'` case
  (`D * inner(m, curl(m))`), with the matching `unit_length ** (dim - 1)`
  scaling convention and a 3D-mesh guard. Interfacial and 1D/2D DMI variants
  are still not covered. [GitHub Copilot / Claude Sonnet 5]
- Cubic anisotropy: `cubic_anisotropy_energy` covers the legacy constant-axis,
  constant-`K1`/`K2`/`K3` case from
  `finmag.energies.cubic_anisotropy.CubicAnisotropy`, and its test is checked
  directly against that module's own analytic reference values (same
  constants/axes as `cubic_anisotropy_test.py`), not just a self-derived case.
  Spatially varying cubic-anisotropy `Field` coefficients are not covered.
  [GitHub Copilot / Claude Sonnet 5]
- Wiring: `RelaxationParameters`/`PrototypeSimulation.energy_terms()` now
  include `dmi_constant` and `cubic_anisotropy_K1`/`K2`/`K3`/`u1`/`u2`
  fields, both defaulting to inert values. `dmi_energy` requires a 3D mesh, and
  both the reduced example and `PrototypeSimulation.unit_square` are 2D, so a
  non-zero `dmi_constant` there raises explicitly; this is covered by an
  explicit test rather than left as a silent gap. [GitHub Copilot / Claude
  Sonnet 5]
- Time stepping: `explicit_llg_step` gives a transparent nodal explicit step
  that tests DOLFINx function mutation and normalisation, but it is not a
  production driver; it only precesses/damps toward a fixed applied-field
  argument. `effective_field_llg_step` now provides a real fix: it computes
  `H_eff = -1/(mu0*Ms) * dE/dm` via the standard finite-element box method
  (assemble the weak-form derivative of the total energy, divide by lumped
  nodal volume) and uses that to drive the step, so exchange/anisotropy/DMI/
  cubic-anisotropy contributions genuinely affect the trajectory.
  `PrototypeSimulation.step()` (and the relax/trace/summary methods built on
  it) expose this as an opt-in `use_effective_field=True`, defaulting to the
  original fixed-field behaviour. [GitHub Copilot / Claude Sonnet 5]
- Simulation time: `PrototypeSimulation.time` and `run_until(...)` are a small
  compatibility-shaped probe for the legacy `Simulation.run_until` concept.
  They deliberately use the prototype explicit stepper and should not be
  treated as a production driver. [Codex gpt-5.5 high]
- Reduced output: `relaxation_example.py`, `PrototypeSimulation` summaries,
  and relaxation traces exercise JSON output contracts only. Trace records use
  absolute prototype simulation times, but they do not cover legacy NDT tables,
  VTK/XDMF, or scheduled output. [Codex gpt-5.5 high]
- Reduced restart: `restart_state` and `restart_example.py` prove a JSON
  round trip for a tiny unit-square prototype, preserving time and nodal
  magnetisation values for the default parameter set. Deserialization currently
  drops non-default DMI and cubic-anisotropy parameters, so this is neither a
  complete prototype parameter round trip nor the legacy Finmag restart format.
- MPI nodal output: the previous `average_nodal_vector` reduced owned and ghost
  entries together; a two-rank nonuniform field returned
  `[1.38888889, 2.37037037, 4]` instead of
  `[1.38888889, 2.38888889, 4]`. The bounded helper now reduces owned nodal
  blocks only. This nodal statistic remains distinct from `Field.average()`,
  whose FEM form integrates owned cells and performs one MPI reduction.
  `field_ownership_probe.py` checks both contracts in serial and on two ranks.
  [Codex GPT-5.6]
- FK demag baseline: the FEniCS-2019/pixi M3 path still computes the
  Fredkin-Koehler BEM matrix through the compiled `finmag.native.llg`
  extension, using `compute_bem_fk` or the DOLFIN-2019-compatible
  `compute_bem_fk_from_arrays` entry point. The Python/NumPy Magpar code is a
  reference/comparison path, not the production FK BEM implementation. [Codex
  gpt-5.5 high]
  The current extension still links `libdolfin` and registers SWIG-DOLFIN
  converters, so its array entry point must be separated from those legacy
  bindings before the extension can be rebuilt for the DOLFINx environment.

## Direct-Edit Readiness Criteria

Before editing the matching module in `src/finmag`, check that:

- the target legacy API surface is identified;
- the existing FEniCS-2019/Python-3 tests that define the expected behaviour
  are known;
- any DOLFINx-driven API or behaviour change is explicitly documented;
- the intended implementation fits the existing `Simulation`, `Field`, energy,
  driver, restart, or output abstraction;
- the DOLFINx mechanic has an executable witness or regression test;
- coordinate ordering and owned/ghost semantics are explicit where relevant;
- unsupported legacy behaviour is tracked as deliberate non-scope, not simply
  omitted;
- the production change will be written directly rather than copied from the
  prototype.

## Near-Term Gaps

- `src/finmag/field.py` is now the direct DOLFINx production implementation.
  It covers constant/vectorized/pointwise-callable assignment, Function and
  Field assignment (including cross-space interpolation), owned raw arrays,
  flat owned `xyz`/`xxx` views, coordinates, global FEM averages (including
  passed subdomain measures), collective normalization, scalar constants,
  random/allclose helpers, and VTK/XDMF output. The focused parity matrix spans
  mesh dimensions 1/2/3 and scalar plus 1/2/3/4-component fields. Collective
  participation is required for reductions and ghost-refreshing mutations.
  The `dev` adapter remains only a frozen witness. [Codex GPT-5.6]
- Production `Field` explicitly rejects `from_expression` (DOLFINx has no
  legacy `Expression`/`UserExpression` equivalent), point probing and
  point-measure arithmetic, legacy plotting, `get_spherical`, and
  `dolfinh5tools` HDF5. XDMF is write-only here. Integration with the still
  legacy `finmag.Simulation` is deferred to its direct port task. [Codex
  GPT-5.6]
- Ordered arrays and `coords_and_values()` are currently rank-local owned
  views. A separate collective, globally coordinate-sorted export should be
  added only for a concrete output/restart consumer; multi-rank ODE state is
  not claimed by the first Field/driver slices. [Codex GPT-5.6]
- There is no DOLFINx-backed `finmag.Simulation` compatibility path yet.
- The Task 5 box assembly and common interactions are now direct production
  code with focused serial/two-rank source tests. Spatially varying `A`, `K1`,
  `K2`, anisotropy axes and `Ms`, matrix/project/direct energy methods,
  region/PBC interaction behavior, `DipolarField`, and time-dependent Zeeman
  variants remain outside this slice. The ported interactions are currently
  direct-use building blocks with DOLFINx `Field`; `Simulation`, hysteresis,
  LLB, and normal-mode consumers remain unported. [Codex GPT-5.6]
- Task 6: `EffectiveField` is now a direct DOLFINx production module. It keeps
  the exact registry API (`add`/`get`/`exists`/`all`/`remove`, unique-name
  `ValueError`, `UnknownInteraction`), total field/energy accumulation, the
  `with_time_update` callback contract (including the "no t given" error), and
  the automatic `TimeZeeman.update` auto-connection (`isinstance` still works
  against the deferred `TimeZeeman` stub without constructing one). `H_eff` is
  now sized from `Field.as_array().size` instead of the legacy
  `vector().local_size()`, matching every ported interaction's owned-array
  contract. `get_dolfin_function` no longer imports
  `finmag.util.helpers.vector_valued_function` (which pulls in legacy
  `dolfin`); it is reimplemented directly with `Field(m.functionspace,
  interaction.compute_field()).f` and explicitly rejects the historical
  (already-unused) `region` argument rather than silently ignoring it.
  `Simulation`, hysteresis, LLB, and normal-mode consumers remain unported.
  [Claude Sonnet 5]
- The production box foundation deliberately requires a blocked
  three-component CG1 magnetisation space. Some higher-order Lagrange row-sum
  lumped weights are non-positive, so accepting arbitrary elements would fail
  late and potentially asymmetrically across ranks. Invalid volume detection
  is collective as a second guard. [Codex GPT-5.6]
- Demag is not covered by the current DOLFINx prototype lane. The production
  DOLFINx port should either reuse/port the array-based native FK BEM routines
  or provide a replacement with equivalent tests; a pure Python/NumPy BEM path
  may be acceptable for tiny references, but not as the assumed production
  implementation. [Codex gpt-5.5 high]
- DMI, cubic anisotropy, variable material parameters, regions, PBC, scheduler
  output, legacy restart files, and production integrators are not covered by
  the current prototype lane. Bulk 3D DMI and constant-axis cubic anisotropy
  are now partial exceptions (see `dmi_energy` and `cubic_anisotropy_energy`);
  interfacial/1D/2D DMI, spatially varying cubic anisotropy, variable material
  parameters, regions, PBC, scheduler output, legacy restart files, and
  production integrators remain uncovered. [GitHub Copilot / Claude Sonnet 5]
- PBC/treecode demag depends on the separate `finmag.native.treecode_bem`
  extension, which is still missing in the pixi path and should remain tracked
  separately from the FK BEM baseline. [Codex gpt-5.5 high]
- The current traces and restart states are useful witnesses, but they are not
  replacements for Finmag's NDT, VTK/XDMF, and restart conventions. [Codex
  gpt-5.5 high]
