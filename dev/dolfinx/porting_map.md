# DOLFINx Compatibility Porting Map

> **Chronological evidence map.** This file contains valuable per-slice
> mechanics and source links, but earlier gap lists are not necessarily current.
> Later addenda supersede earlier entries. Use
> `../../docs/superpowers/capability-status.md` for current status and
> `../../docs/superpowers/acceptance-register.md` for owner decisions. [Codex GPT-5]

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

- Field `xxx` consumers: `drivers/llg_integrator.py`, the NEB
  implementations, and `Field.np` require component-blocked state. DOLFINx's
  interleaved blocked storage does not remove that API requirement. The SciPy
  driver used to initialize from raw `as_array()` but call an LLG setter
  that interprets its argument as `xxx`. Multi-rank stepping remains outside
  the initial driver slice. [Codex GPT-5.6]
- Task 8 update: `drivers/scipy_integrator.py` is now the direct DOLFINx
  production `ScipyIntegrator`. It resolves the raw/`xxx` ambiguity noted
  above by seeding and writing back the ODE state exclusively through
  `Field.get_ordered_numpy_array_xxx()`/`set_with_ordered_numpy_array_xxx()`
  in both directions, matching what `llg.solve_for` has always expected.
  `advance_time` now rejects `t < cur_t` with `ValueError` and an
  unsuccessful `scipy.integrate.ode` step with `RuntimeError` (replacing a
  bare `assert`), and `reinit()` does real work: it rebuilds the underlying
  VODE integrator seeded from the current field state at the current time,
  mirroring the Sundials reinit contract instead of only logging
  "not supported". `drivers/llg_integrator.py`'s public `backend=` argument
  and shape are unchanged; only its default value changes from
  `"sundials"` to `"scipy"`, making the ported driver the working default
  while an explicit `backend="sundials"` request keeps raising `ImportError`
  by name (native Sundials/CVODE remains unported). [Claude Sonnet 5]
  Task 20 update: sundials is no longer a gap. Native Sundials/CVODE is
  ported and fully validated on the DOLFINx stack (`SundialsIntegrator`
  wired against the ported `LLG`, `bdf_gmres_prec_id` default path exercised
  end-to-end). Fix round 1 (Task 20 review) restored `llg_integrator`'s
  `backend` default from `"scipy"` back to `"sundials"`, matching the legacy
  default semantics -- the initial slice had left it at `"scipy"` pending
  review even though full validation had already passed. An explicit
  `backend="scipy"` remains fully supported. `Simulation.integrator_backend`
  is a separate default and deliberately stays `"scipy"` -- DELIBERATE
  DEVIATION, USER ACCEPTANCE PENDING (see the register entry in the Task 9
  update below and `transition-notes.org`'s "Native Sundials/CVODE on DOLFINx
  (Task 20)" section). **Superseded by `81fab481` (P1.3): native Sundials is
  now the public `Simulation`/`sim_with` default; D8 is discharged and SciPy
  remains an explicit opt-in.** [Claude Sonnet 5]
- Task 7 update: `physics/llg.py` is no longer an unported `xxx` consumer. The
  direct DOLFINx port makes `solve`/`solve_for` and the `m` setters
  unambiguously component-blocked coordinate-ordered `xxx`, and routes the
  registry's raw owned-order `H_eff` through `Field.from_array` +
  `get_ordered_numpy_array_xxx()` into the same ordering before the node-local
  update, so the raw/`xxx` ambiguity is resolved on the LLG side (Task 8 should
  match it from the driver). Deferred surfaces raising `NotImplementedError` by
  name: Slonczewski/Zhang-Li STT (`use_slonczewski`/`use_zhangli`), native
  CVODE `sundials_jtimes`/`sundials_psetup`/`sundials_psolve`, spatially varying
  `alpha`, and multi-rank ODE state (serial-only guard on `solve`/`solve_for`/
  `sundials_m` when `comm.size > 1`, matching the note above that multi-rank
  stepping is not claimed). `sundials_rhs` is kept as a backend-neutral
  `solve_for` adapter. The deterministic dm/dt is transcribed from the native
  `calc_llg_dmdt` (`native/src/llg/llg.cc`), including the `c*(1-|m|^2)*m` norm-
  relaxation term. [Claude Opus 4.8]
  Task 20 update: `sundials_jtimes`/`sundials_psetup`/`sundials_psolve` are no
  longer by-name deferrals -- all three are real implementations of the
  legacy `bdf_gmres_prec_id` default path (SPGMR + identity preconditioner +
  analytic Jacobian-times-vector, the node-local kernel transcribed from
  native `dm_precession_i`/`dm_damping_i`/`dm_relaxation_i`), validated to
  <1e-5 relative against a finite-difference of the rhs. Spatially varying
  `alpha` was already ported by Task 16 (unrelated to this task). The only
  remaining by-name items unchanged from this list: Slonczewski/Zhang-Li STT
  (`use_slonczewski`/`use_zhangli`) and the multi-rank ODE state guard.
  [Claude Sonnet 5]
  Task 22 update: Slonczewski/Zhang-Li STT (`use_slonczewski`/`use_zhangli`) are
  no longer by-name deferrals -- both are ported as pure-NumPy transcriptions of
  the native `calc_llg_slonczewski_dmdt`/`slonczewski_xiao_i` and
  `calc_llg_zhang_li_dmdt` kernels (`native/src/llg/llg.cc`; the compiled STT
  kernels are NOT rebuilt, same decision as the Task 7 `_dmdt_numpy` and Task 20
  `_jtimes_numpy` transcriptions). The Zhang-Li `(J.grad)m` gradient is the
  legacy lumped-box operator (`compute_gradient_matrix`) assembled as a linear
  functional; per-node `Ms` is the legacy lumped CG1 projection. Pinned by two
  coordinate-ordered oracle fixtures (`slonczewski_rhs.json`, `zhangli_rhs.json`
  -- the latter also pins the discrete `H_gradm`), analytic direction/scaling
  pins, a Zhang-Li domain-wall-displacement witness, a Slonczewski tilt-sign
  witness, and a scipy-vs-sundials cross-backend check. Note: the port uses
  coordinate-ordered arrays throughout (legacy had raw dof-ordered `J`/`Ms`/`p`
  alongside coordinate-ordered `m`/`H`/`alpha`, a latent inconsistency); this is
  a DELIBERATE DEVIATION with identical results for all pinned cases
  (documented in `transition-notes.org` Task 22). The only remaining by-name
  `LLG` item is the multi-rank ODE state guard. The separate NONLOCAL STT class
  `llg_stt.LLG_STT` (native `calc_llg_nonlocal_stt_dmdt`; reached only via
  `Simulation(kernel="llg_stt")`) stays deferred by name -- a distinct
  spin-accumulation capability, Task 29-registered. [Claude Opus 4.8]
- Task 7 addendum: legacy `set_pins` logged `logger.error(...)` for
  out-of-range pin indices and silently kept the previous `_pins` array
  unchanged; the ported `set_pins` instead raises `ValueError` for the same
  condition, a deliberate fail-fast deviation covered by
  `test_out_of_range_pins_raise`. [Claude Sonnet 5]
- Task 9 update: `src/finmag/sim/sim.py` is now the direct DOLFINx port of the
  core `Simulation`. It preserves the core public surface (construction on a
  DOLFINx mesh with scalar `Ms`/`unit_length`/`name`/scalar `alpha`/`gamma`;
  `set_m`/`m`/`m_field`/`m_average`/`t`/`dmdt`; the interaction registry and
  energy accessors; lazy integrator creation, `set_tol`, `advance_time`,
  `run_until`, `reset_time`, `reinit_integrator`; and `sim_with` for Exchange /
  Zeeman / UniaxialAnisotropy) driven through the ported
  `LLG`/`EffectiveField`/`Field`/`ScipyIntegrator` stack, with no legacy
  `dolfin`, `finmag.native`, scheduler, table writer, or per-simulation log
  file. `finmag.__init__` marks `Simulation`/`sim_with` as no longer requiring
  the legacy dolfin bridge, and `finmag/sim/__init__.py`'s legacy `.init`
  logging bootstrap is guarded so the package imports in the DOLFINx
  environment. Deliberate deviations: `integrator_backend` defaults to
  `"scipy"`; `m`/`dmdt` return component-blocked `xxx` arrays; `t` reports
  `0.0` until an integrator exists (no lazy-create-to-read-clock); `set_tol`
  reinits the SciPy driver; `reset_time` reseeds it (no `t0` kwarg; superseded
  for backend-neutral reset semantics by the P1.1 entry below); `Volume`
  uses a DOLFINx assemble. Deferred by name when requested (import and core
  paths stay clean): PBC, `parallel=True`, `sllg`/`llg_stt` kernels, `sim_with`
  demag (default `"FK"`, pass `demag_solver=None`) and DMI (`D`), STT,
  scheduler, restart, NDT/VTK/field output, regions, point probing, `relax`,
  hysteresis, normal modes, and callable `pins` masks. Long-tail modules
  (`sim_helpers`, `sim_savers`, `hysteresis`, `magnetisation_patterns`, the
  legacy scheduler) are untouched and no longer on the core import graph.
  [Claude Opus 4.8]
  Task 15 update: `relax` and `hysteresis`/`hysteresis_loop` are now ported
  (`Simulation.relax`/`hysteresis`/`hysteresis_loop` bound directly to the
  untouched legacy `sim_relax.py`/`hysteresis.py` modules, whose only edit in
  this slice is a local dolfin-free reimplementation of their
  `finmag.util.helpers` import). `magnetisation_patterns` and the legacy
  per-simulation table writer remain untouched/off the core import graph.
  [Claude Sonnet 5]
  Task 20 register entry, DELIBERATE DEVIATION, USER ACCEPTANCE PENDING:
  `Simulation.integrator_backend` default is `"scipy"` on DOLFINx vs legacy
  `"sundials"` -- temporary, pending native-default hardening or user
  acceptance. Native Sundials/CVODE is fully ported and validated (Task 20;
  `llg_integrator`'s own factory default was restored to `"sundials"` in the
  Task 20 review's fix round 1), so `integrator_backend="sundials"` works
  end-to-end when requested explicitly. `Simulation`'s default is kept at
  `"scipy"` regardless, because flipping it would couple every `run_until`
  gate across the M5 suite to the native sundials build and CVODE
  trajectory -- a larger re-validation deliberately not done in this slice.
  Mirrored in `transition-notes.org`'s "Native Sundials/CVODE on DOLFINx
  (Task 20)" section and the Task 20 section of
  `docs/superpowers/plans/2026-07-21-dolfinx-full-parity.md`; cross-referenced
  from `test_simulation_dolfinx.py::test_construction_core_state`'s docstring.
  **Superseded by `81fab481` (P1.3): native Sundials is now the public
  `Simulation`/`sim_with` default; D8 is discharged and SciPy remains an
  explicit opt-in.** [Claude Sonnet 5]
- Task 9 removed-surfaces addendum: a handful of legacy `Simulation` public
  names were dropped outright rather than deferred by name --
  `initialise_helix_2D`, `initialise_skyrmions`,
  `initialise_skyrmion_hexlattice_2D`, `initialise_vortex` (from
  `magnetisation_patterns`), `skyrmion_number`/
  `skyrmion_number_density_function` (from `sim_helpers`), `length_scales`/
  `mesh_info` (from `sim_details`), `profile`, `close_logfile`, and the
  `instances_*` management family (`instances_list_all`,
  `instances_delete_all`, `instances_delete_all_others`,
  `instances_alive_count`) plus `shutdown`; and, on the ported `LLG` itself,
  the legacy full-moment `M`/`M_average` properties (superseded by the unit
  `m`/`m_average` contract) and the STT enable flags
  `do_slonczewski`/`do_zhangli` (STT stays deferred behind
  `set_stt`/`set_zhangli` raising by name). They now raise a plain
  `AttributeError` rather than a by-name `NotImplementedError`, accepted by
  review as a judgment-call deferral. [Claude Sonnet 5]
  Task 22 update: STT is no longer deferred at the `Simulation` surface either.
  `set_stt`/`toggle_stt`/`set_zhangli` are ported pass-throughs to the LLG
  `use_slonczewski`/`do_slonczewski` toggle/`use_zhangli` surfaces (restoring
  the legacy signatures faithfully), and the ported `LLG` re-exposes the
  `do_slonczewski`/`do_zhangli` flags (set by the `use_*` methods and toggled by
  `toggle_stt`). The Task 9 "STT stays deferred" and "STT" entries above are
  superseded for the in-LLG torques; only `kernel="llg_stt"` (nonlocal STT)
  remains deferred. [Claude Opus 4.8]
  Task 26a update: `skyrmion_number`/`skyrmion_number_density_function` are no
  longer dropped -- restored directly against DOLFINx/UFL in
  `finmag.sim.sim_helpers` (thin `Simulation.skyrmion_number`/
  `skyrmion_number_density_function` delegators added, matching the style of
  the existing `sim_helpers.*` pass-throughs), transcribing legacy's
  `-1/(4*pi) integral(m.(dm/dx x dm/dy))` formula verbatim (2D: whole domain;
  3D: top surface only, via `dolfinx.mesh.locate_entities_boundary` +
  `meshtags` + a `ufl.Measure("ds", ...)`, the DOLFINx equivalent of legacy's
  `SubDomain`/`ds[markers]`). See `transition-notes.org`'s "I/O utility parity
  (Task 26a)" section for the full validation. `initialise_helix_2D`/
  `initialise_skyrmions`/`initialise_skyrmion_hexlattice_2D`/
  `initialise_vortex` (from `magnetisation_patterns`) and everything else in
  this Task 9 addendum remain dropped/untouched -- out of scope for 26a. Also
  Task 26a: `Field.probe`/`Field.__call__` (point evaluation) and
  `Field.get_spherical` are restored (see `field.py`'s Near-Term-Gaps entry
  below); `Simulation.probe_field`/`probe_field_along_line` remain deferred by
  name (they depend on region-restricted `get_field_as_dolfin_function`,
  out of scope for 26a). [Claude Sonnet 5]
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
  Task 15 update: `TimeZeeman`/`DiscreteTimeZeeman`/`TimeZeemanPython`/
  `OscillatingZeeman`/`DipolarField` are now `requires_legacy_dolfin=False`
  and dolfin-clean -- constructing any of them builds the ported DOLFINx
  class directly (see `transition-notes.org`'s Task 15 section for the
  `field_function(t)` input-contract deviation and two preserved legacy
  quirks). No `_DeferredZeeman` names remain in `finmag.energies.zeeman`.
  [Claude Sonnet 5]
- Bulk DMI: `dmi_energy` covers the legacy 3D `dmi_type='auto'` case
  (`D * inner(m, curl(m))`), with the matching `unit_length ** (dim - 1)`
  scaling convention and a 3D-mesh guard. Interfacial and 1D/2D DMI variants
  are still not covered. [GitHub Copilot / Claude Sonnet 5]
  Task 13 update: `src/finmag/energies/dmi.py` is now the direct DOLFINx
  production port of `DMI` (constant scalar `D` only; spatially varying `D`
  deferred by name), covering `dmi_type='auto'`/`'1d'`/`'2d'`/`'3d'` (bulk,
  transcribed from `finmag.util.helpers.times_curl`) and `'interfacial'`
  (transcribed from `finmag.energies.dmi.DMI_interfacial`), each preserving
  the legacy `unit_length ** (dim - 1)` scaling convention exactly (verified
  against hand-derived affine fields, not just asserted). The undocumented
  legacy `dmi_type='D2D'` branch is not ported this slice and raises
  `NotImplementedError` naming `D2D` explicitly. Chirality/sign convention is
  pinned two ways: an exact affine "twist" field (transcribed from
  `finmag.tests.test_dmi_terms`) and a genuine sinusoidal helix, where
  negating the wavevector is shown to exactly negate the assembled energy as
  an algebraic FEM-level identity (`E(k) + E(-k) == 0.0` to machine
  precision), sidestepping the discretisation error a direct comparison to
  the continuum analytic value would carry. Two coordinate-ordered legacy
  oracle fixture cases (`bulk_3d`, `interfacial`; schema v1,
  `src/finmag/tests/fixtures/dmi_oracle.json`/`gen_dmi_oracle.py`) confirm
  faithful transcription against the frozen legacy class to near machine
  precision (energy relative error `0.0`/`7.1e-15`; field relative error
  `~5-8e-16`). `sim_with(D=...)` now constructs the ported `DMI`. Gate:
  `dolfinx-src-dmi-pytest`. [Claude Sonnet 5]
- Cubic anisotropy: `cubic_anisotropy_energy` covers the legacy constant-axis,
  constant-`K1`/`K2`/`K3` case from
  `finmag.energies.cubic_anisotropy.CubicAnisotropy`, and its test is checked
  directly against that module's own analytic reference values (same
  constants/axes as `cubic_anisotropy_test.py`), not just a self-derived case.
  Spatially varying cubic-anisotropy `Field` coefficients are not covered.
  [GitHub Copilot / Claude Sonnet 5]
  Task 14 update: `src/finmag/energies/cubic_anisotropy.py` is now the direct
  DOLFINx production port of `CubicAnisotropy` (constant scalar
  `K1`/`K2`/`K3` and constant `u1`/`u2` axes only; spatially varying
  coefficients/axes deferred to Task 16). `u3 = u1 x u2` and the legacy
  non-normalisation/non-orthogonality-checking of `u1`/`u2` are preserved
  exactly (confirmed against the frozen legacy source, the legacy module's
  own not-quite-unit/not-quite-orthogonal test axes, and this file's own
  prior probe note above -- not guessed). The legacy `assemble` flag only
  ever gated *field* computation (energy is always box-assembled in both
  legacy and here): `assemble=True` reuses the same box-assemble weak-form
  derivative `EnergyBase` already provides; the legacy-default
  `assemble=False` uses the native `compute_cubic_field` analytic field.
  Legacy's own `sim_with` never had cubic-anisotropy parameters, so
  `Simulation.add(CubicAnisotropy(...))` is the full integration surface.
  Cubic-symmetry easy/hard-axis ordering for `K1>0` (`<100>` easy, `<111>`
  hard, exact `K1/4`/`K1/3` energy differences from `<100>`) is verified
  directly from the transcribed form, not assumed. One coordinate-ordered
  legacy oracle fixture case (`cubic_3d`; schema v1,
  `src/finmag/tests/fixtures/cubic_anisotropy_oracle.json`/
  `gen_cubic_anisotropy_oracle.py`, using the exact
  `cubic_anisotropy_test.py` constants/axes plus a nonuniform unit-norm `m`)
  confirms faithful transcription against the frozen legacy class to near
  machine precision (energy relative error `8.3e-14`; field relative error
  `1.4e-15`). Gate: `dolfinx-src-cubicanis-pytest`. [Claude Sonnet 5]

  Task 14 fix round 1 [Claude Opus 4.8]: the initial slice ported only
  `assemble=True` and raised `NotImplementedError` on `compute_field()` for
  the legacy-*default* `assemble=False`, which made a default-constructed
  `CubicAnisotropy` unable to participate in dynamics -- a functional
  regression the Task 14 review flagged (legacy *capability* is the parity
  bar, not just legacy test usage). Remedy 2: `_compute_field_analytic` now
  ports the native `compute_cubic_field` path as a NumPy transcription of the
  hand-derived closed form `H = -1/(mu0 Ms) dE/dm` (chain rule through
  `a=u1.m`, `b=u2.m`, `c=u3.m`), restoring the legacy default's *capability*
  with the legacy default's *discretisation* (exact nodal analytic field, not
  the box-assemble derivative). Both flags now support `compute_field()` and
  dynamics; they differ only in discretisation and converge together under
  refinement (box-vs-analytic rel. L2 gap for `all_nonzero` at n=2,4,8 =
  0.616/0.355/0.138). Companion native-field fixture
  `cubic_anisotropy_native_oracle.json` (`gen_cubic_anisotropy_native_oracle.py`,
  same oracle commit; K1/K2/K3-only + all-nonzero cases; Ms on CG1 because the
  native routine needs a per-vertex Ms array) matches the ported analytic
  field to ~2e-15 per case. DELIBERATE DEVIATION, USER ACCEPTANCE PENDING:
  the legacy native routine has a real K2 typo at
  `native/src/llg/energy.cc:116` (`hz[i] += K2[2]*(...)` -- fixed index `2`
  where every other line uses per-node `K2[i]`); it is numerically dormant
  for the constant K2 this slice supports (uniform nodal K2 array so
  `K2[2] == K2[i]`; the `k2_only` oracle matches the *correct* derivation to
  2.7e-15), so the legacy K2 native-field bug is not reproduced. The port
  implements the correct per-node field unconditionally. NOW LIVE (Task 16):
  spatially varying K2 is supported, so this deviation is active -- the port's
  correct `hz` diverges from the legacy native `K2[2]` field by up to ~2.59e5
  A/m while `hx`/`hy` and the energy match; pinned executably in
  `test_variable_params_dolfinx.py::test_k2_varying_diverges_from_legacy_native_in_hz_only`
  against `cubic_k2_varying_oracle.json` (still USER ACCEPTANCE PENDING). See
  `transition-notes.org` Task 14 "assemble flag" and Task 16 sections for the
  full derivation, C++ comparison, and per-case numbers.
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
  Task 11a update: the array entry point is now separated from the legacy
  bindings. The legacy `finmag.native.llg` module still links `libdolfin` and
  registers SWIG-DOLFIN converters (unchanged, for the FEniCS-2019 lane), but
  the array-only Lindholm/BEM kernels now live in a shared, dolfin-free header
  (`native/src/llg/bem_arrays.h`) reused by a new standalone
  `finmag.native.bem_arrays` extension. That module exposes
  `compute_bem_fk_from_arrays`, `compute_bem_gcr_from_arrays`,
  `compute_lindholm_L`/`_K` with **no** dependency on libdolfin, the
  SWIG-DOLFIN converters, or dolfin headers (compiled with
  `-DFINMAG_NO_DOLFIN -DFINMAG_NO_SUNDIALS`). It builds and imports on Python
  3.12 in the `dolfinx` pixi env without legacy DOLFIN installed, and
  reproduces the legacy compiled BEM matrix bit-for-bit (gate
  `dolfinx-src-native-bem-pytest`). The remaining Task 11 work (DOLFINx
  boundary-mesh probes, porting `src/finmag/energies/demag`, oracle BEM
  comparisons, barmini FK-demag workflow) is Task 11b. [Claude Opus 4.8]
  Task 11b update: FK demag is no longer a gap. `src/finmag/energies/demag/
  fk_demag.py` is now the direct DOLFINx port of the Fredkin-Koehler solver
  (preserving the two-potential discrete formulation; only the FEM-API layer --
  DOLFINx spaces, PETSc KSP, boundary extraction, dof maps -- changed), wired
  into `Simulation.add(Demag())` and `sim_with(demag_solver="FK")` (the
  default). The boundary-node ordering fed to the compiled
  `compute_bem_fk_from_arrays` is built by `boundary_bem_arrays`: BEM-local node
  `i` is a CG1 boundary dof of S1, `coords[i]`/`b2g[i]` are its coordinate/
  S1-dof-index (the whole gather/scatter stays in S1 dof space), and the
  boundary triangles are explicitly oriented **outward** (the sorted-index
  winding DOLFINx returns from facet->vertex connectivity is not consistent and
  yields a wrong-though-row-sum-`-1` BEM); triangle vertices are resolved to
  BEM-local indices by coordinate, so the mapping is robust to submesh/entity
  reordering. This is pinned bit-for-bit (atol 1e-13) against the 11a golden
  matrix through both the explicit Kuhn cube and `create_unit_cube`. Field and
  energy are validated against coordinate-ordered frozen-oracle references (cube
  energy rel 4.7e-8, barmini rel 2.3e-9; pointwise ~5e-6 at the standard 1e-6
  Krylov tolerance on both sides) plus the analytic cube demag factor (avg H =
  -Ms/3, E = mu0 Ms^2 V/6).
  `solver_type='LU'`, `Demag2D`/`GCR` raise `NotImplementedError` by name.
  Task 23 update: `Demag(solver='Treecode')` and `MacroGeometry` /
  `FKDemag(macrogeometry=...)` are now PORTED (see the Task 23 row below);
  `Demag2D`/`GCR` stay deferred by name. Gate: `dolfinx-src-demag-pytest`
  (18 tests; the Task 11b Treecode/MacroGeometry/macrogeometry-argument
  fail-forward pins flipped to ported-behavior assertions, net -1).
  [Claude Opus 4.8]

- `energies/demag/treecode_bem.py` (`TreecodeBEM`),
  `energies/demag/fk_demag_pbc.py` (`MacroGeometry` + `BMatrixPBC` +
  `build_periodic_bem`), `native/src/treecode_bem/` (Cython + C):
  **PORTED (Task 23).** The `treecode_bem` Cython extension (pure-C octree
  fast-summation + Lindholm BEM kernels; audited dolfin/Boost/SWIG-free) is
  rebuilt for the DOLFINx env (setuptools+cythonize `setup.py`, NumPy-2 clean,
  `treecode_bem.so` added to the DOLFINx Makefile `MODULES`, `cython` added to
  the pixi feature -> `pixi.lock` changed). `TreecodeBEM`
  (`Demag(solver='Treecode')`) rides the ported `FKDemag`, replacing the dense
  BEM matvec with the `FastSum` fast-summation of the SAME operator.
  `MacroGeometry` / `FKDemag(macrogeometry=...)` build the periodic image-sum
  BEM. Validation is cross-method + analytic (NO treecode oracle fixtures
  exist -- the oracle env never built the module): single-tile periodic BEM ==
  golden dense FK BEM bit-for-bit (3.5e-17); treecode-vs-dense-FK cross-check
  at `<1e-6` in the direct-sum limit on a cube.

  **Round-1 review correction ([Claude Sonnet 5]):** the approximation-regime
  claim here previously read "~4e-5 at legacy default mac=0.3/p=3" without
  qualification; that number is NOT reproducible on the box/sphere geometries
  the encoded tests actually use -- on those *compact convex* boundaries
  treecode-vs-FK agreement is machine precision (`~1e-14`) at every
  `mac`/`p`/`num_limit` combination swept (boundaries up to 1178 nodes),
  because the octree's far-field acceptance test
  (`treecode_bem_I.c: mac_square * R > tree->radius_square`) is never
  satisfied when query points sit on the same compact surface every cluster
  is bounded by (near-field covers the whole sum; see transition-notes.org for
  the full C-source citation). The figure IS reproducible on a genuine
  approximation-regime witness -- a 50:1 aspect-ratio bar -- where it measures
  `4.628e-05` at `mac=0.3` (monotonic with `mac` over `{0.7,0.5,0.3,0.1}`);
  `p` is confirmed (from source: the multipole arrays are hard-coded to a
  fixed 35-term/4th-order expansion regardless of the `p` argument) and
  empirically to have zero effect on accuracy. Sphere demag factor ~1/3;
  periodic image-sum convergence (trend-only, no closed-form plateau exists;
  see transition-notes.org) + analytic out-of-plane thin-film limit; two
  `Simulation.add(Demag(...))` end-to-end smokes (Treecode solver,
  non-coincident `MacroGeometry`).
  `Demag2D` DEFERRED (heavy MeshEditor/Expression coupling, no treecode
  dependency; Task 29). `demag_treecode.py` (uses the never-built `fast_sum_lib`,
  Python-2 code) NOT ported (dead/experimental). Legacy-lane treecode activation
  left as a follow-up (not trivially safe). Gate: `dolfinx-src-treecode-pytest`
  (20 tests -- 16 baseline + 4 round-1 review additions). [Claude Opus 4.8]
  [Claude Sonnet 5]

- `util/pbc2d.py` (`PeriodicBoundary1D/2D`), `Simulation(pbc='1d'/'2d')`:
  **DEFERRED by name (Task 29 candidate).** These are `dolfin.SubDomain`
  `constrained_domain` periodic *function spaces* -- a DIFFERENT capability from
  MacroGeometry demag; DOLFINx has no `constrained_domain`, so periodic spaces
  need `dolfinx_mpc` (not in the env). The behavioral PBC-demag contract
  (`demag_pbc_test.py`) uses `pbc=None` and gets periodicity from
  `MacroGeometry`, so the demag capability is complete without `dolfinx_mpc`.
  `pbc2d.py` stays dormant (imported only by still-deferred LLB physics).
  [Claude Opus 4.8]

  Review round 1, Finding 1 (2026-07-21): the ~5e-6 residual above was
  *demonstrated*, not merely asserted, to be the oracle fixture's own frozen
  1e-6 Krylov tolerance rather than a systematic method difference.
  Tightening only the port's KSP rtol to 1e-12 left the pointwise gap
  unchanged at 6.4e-6 (energy 4.7e-8 -> 3.3e-9), falsifying the original
  "Krylov solver tolerance" wording as stated. Regenerating the cube datum
  at the oracle with *both* legacy Krylov solves tightened to 1e-12 and
  comparing against the port at the same tightened tolerance collapses the
  pointwise gap to ~1.4e-12 and the energy gap to ~5.5e-14 (fixture case
  `cube_tight_tolerance`, `gen_fk_demag_oracle.py`,
  `test_oracle_cube_tight_tolerance_isolates_krylov_residual`). The fixture
  is now schema-v1 conforming (`docs/superpowers/specs/legacy-oracle-fixtures.md`)
  with a real, checked-in generator and per-quantity tolerances stored in the
  fixture. [Claude Sonnet 5]

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
  legacy `Expression`/`UserExpression` equivalent) and point-measure
  arithmetic (`__add__`/`__mul__`/`__truediv__`/`cross`/`dot`/
  `coerce_scalar_field`, which relied on legacy's `dP`-measure assembly), and
  legacy plotting and `dolfinh5tools` HDF5 remain deferred. XDMF is
  write-only here. Point probing and `get_spherical` were originally on this
  rejected list too, but Task 26a restored both directly (see the Task 26a
  update just below) -- do not confuse "point probing" here with the still
  fully deferred "point-measure arithmetic" (the `Field.__add__`-family
  stubs), which are unrelated capabilities that happen to share the word
  "point". Integration with the still legacy `finmag.Simulation` is deferred
  to its direct port task. [Codex GPT-5.6]
  Task 16 update: `Field.from_function` now *interpolates* a `dolfinx.fem.
  Function` into this Field's space when the two spaces differ, instead of
  only supporting an identical-space dof-for-dof copy (mirroring `from_field`,
  which already interpolated between compatible spaces). Legacy raised a hard
  error on any space/size mismatch (no interpolation path existed for a raw
  `Function` argument); same-space assignment is unchanged (still a
  dof-for-dof copy), so this is a *tested superset* of the legacy contract,
  not a behaviour change to any previously working call -- it only newly
  accepts calls legacy rejected (e.g. placing a DG0-valued coefficient
  `Function` into a CG1 `Field`). The cross-space case is the same code path
  `test_from_field_interpolates_between_compatible_spaces`
  (`test_field_dolfinx.py`) exercises for `from_field`; no existing test
  relied on the old hard error. [Claude Sonnet 5]
  Task 26a update: `Field.probe(point)`/`Field.__call__(point)` and
  `Field.get_spherical()` are restored, no longer legacy-only failures. Point
  evaluation is a genuine mechanism restoration: DOLFINx `fem.Function`
  objects are not directly callable at a point (the one thing DOLFINx
  actually forced here), so a shared module-level
  `finmag.field.evaluate_at_point(function, point)` helper (bb_tree +
  compute_colliding_cells + `Function.eval`) reproduces the point-in-cell
  search, usable on both a `Field` (via `.probe`/`.__call__`) and any raw
  `dolfinx.fem.Function`. `get_spherical` needed no forced-change workaround
  at all -- it is a pure analytic nodal computation (`theta = atan2(m_r,
  m_z)`, `phi = atan2(m_y, m_x)`), computed directly as NumPy instead of
  reproducing legacy's `dP`-point-measure assembly trick (mathematically
  identical result). Neither restoration touches the still-deferred
  point-measure-arithmetic stubs noted just above. See
  `src/finmag/tests/test_io_utils_dolfinx.py` and transition-notes.org's "I/O
  utility parity (Task 26a)" section for the full validation. [Claude Sonnet 5]
- Ordered arrays and `coords_and_values()` are currently rank-local owned
  views. A separate collective, globally coordinate-sorted export should be
  added only for a concrete output/restart consumer; multi-rank ODE state is
  not claimed by the first Field/driver slices. [Codex GPT-5.6]
- Task 9 delivered the DOLFINx-backed core `finmag.Simulation`/`sim_with`
  (see the Task 9 update above). FK demag (Task 11b, below) and the
  scheduler/restart/NDT/VTK output surfaces (Task 12, below) are now ported
  and supported; PBC, the non-FK demag variants, stochastic/STT kernels,
  regions, hysteresis, and normal-mode consumers remain unported and fail by
  name when requested. [Claude Sonnet 5]
  Task 15 update: `Simulation.relax`/`hysteresis`/`hysteresis_loop` are now
  ported (bound exactly the way legacy did, over the untouched legacy
  `sim_relax.py`/`hysteresis.py` modules -- only their `finmag.util.helpers`
  import was replaced by a local dolfin-free reimplementation in
  `sim_helpers.py`). See `transition-notes.org`'s Task 15 section for an
  important finding: legacy's own `hysteresis()`/`hysteresis_loop()`, when
  actually exercised (including against the frozen oracle), do not achieve
  an independent re-relaxation after their first stage -- preserved
  verbatim, not fixed (DELIBERATE DEVIATION, USER ACCEPTANCE PENDING). PBC,
  non-FK demag, stochastic/STT kernels, regions, and normal-mode consumers
  remain unported. [Claude Sonnet 5]
- Task 10 established the first aggregated direct-source DOLFINx gate
  (`dev/bin/verify-dolfinx-m5`), running every `dolfinx-src-*` focused pytest
  gate and MPI probe plus a new core physical-time smoke
  (`dolfinx-src-core-smoke`, `run_until(1e-12)` on the Task 9 workflow) and a
  new aggregated deferred-surfaces sweep (`dolfinx-src-deferred-pytest`). See
  `transition-notes.org`'s "First Direct-Source DOLFINx Gate (Task 10, 'M5')"
  section for the exact command, naming rationale, and measured results. This
  is an aggregation/verification milestone over Tasks 3-9, not a new porting
  slice; the underlying deferred-surface list above and the Task 9 update's
  "Judgment-call deferrals" are still the authoritative content this gate
  checks.
- The deferred-surfaces sweep surfaced one pre-existing, out-of-scope gap
  rather than fixing it: directly constructing a still-fully-unported
  `requires_legacy_dolfin=True` optional energy class (`Demag`, `DMI`,
  `CubicAnisotropy`, `ThinFilmDemag`, `FixedEnergyDW`, `Demag2D`,
  `MacroGeometry`) surfaces the raw `ModuleNotFoundError: No module named
  'dolfin'` rather than a curated by-name error, because these modules still
  import legacy `dolfin` at module scope and are not ported at all yet. This
  is distinct from the `Simulation`/`sim_with`-mediated demag/DMI *request*
  paths, which already raise a curated `NotImplementedError` before ever
  reaching these modules. Fixing the direct-construction case (e.g. a curated
  `NotImplementedError` at each class's `__init__`, or lazily deferring the
  `import dolfin` past a by-name guard) is left to whichever later slice
  actually ports or explicitly stubs each class (Task 11 for demag; "Later
  Value-Driven Slices" item 1 for DMI/cubic anisotropy), rather than being
  patched incidentally by this aggregation-only gate. [Claude Sonnet 5]
  Task 11b update: `Demag`/`Demag2D`/`MacroGeometry` are now
  `requires_legacy_dolfin=False` and dolfin-clean -- `Demag()` builds the ported
  DOLFINx `FKDemag`; `Demag2D`/`MacroGeometry`/non-FK solvers raise curated
  by-name `NotImplementedError`. The remaining
  `DMI`/`CubicAnisotropy`/`ThinFilmDemag`/`FixedEnergyDW` classes are still the
  documented direct-construction gap. [Claude Opus 4.8]
  Task 13 update: `DMI` is now `requires_legacy_dolfin=False` and
  dolfin-clean -- constructing it directly builds the ported DOLFINx `DMI`
  (see the Task 13 update to the "Bulk DMI" entry above for the full form/
  scaling/chirality/fixture evidence). The undocumented legacy
  `dmi_type='D2D'` variant and spatially varying `D` raise curated by-name
  `NotImplementedError`. The remaining `CubicAnisotropy`/`ThinFilmDemag`/
  `FixedEnergyDW` classes are still the documented direct-construction gap.
  [Claude Sonnet 5]
  Task 16 update: spatially varying `D` is now SUPPORTED (callable/Field/
  Function, placed in DG0 as legacy did); only the `dmi_type='D2D'` variant and
  legacy string Expressions remain deferred by name. [Claude Opus 4.8]
  Task 14 update: `CubicAnisotropy` is now `requires_legacy_dolfin=False` and
  dolfin-clean -- constructing it directly builds the ported DOLFINx
  `CubicAnisotropy` (see the Task 14 update to the "Cubic anisotropy" entry
  above for the full form/axis-handling/assemble-flag/fixture evidence).
  Spatially varying `K1`/`K2`/`K3`/`u1`/`u2` raise curated by-name
  `NotImplementedError`. Only `ThinFilmDemag`/`FixedEnergyDW` remain the
  documented direct-construction gap. [Claude Sonnet 5]
  Task 16 update: spatially varying cubic `K1`/`K2`/`K3` are now SUPPORTED
  (CG1-placed; the analytic assemble=False path uses per-node mass-lumped K
  arrays exactly as legacy fed the native routine, which makes the K2 native
  typo LIVE -- see the "Cubic anisotropy" K2 deviation note above). Spatially
  varying cubic axes `u1`/`u2` remain deferred by name. [Claude Opus 4.8]
- The Task 5 box assembly and common interactions are now direct production
  code with focused serial/two-rank source tests. Spatially varying `A`, `K1`,
  `K2`, anisotropy axes and `Ms`, matrix/project/direct energy methods,
  region/PBC interaction behavior, `DipolarField`, and time-dependent Zeeman
  variants remain outside this slice. The ported interactions are currently
  direct-use building blocks with DOLFINx `Field`; at the time of this Task 5
  slice, hysteresis, LLB, and normal-mode consumers remained unported
  (`Simulation` itself is ported as of Task 9/11b/12, above). [Codex GPT-5.6]
  Task 15 update: `DipolarField` and every time-dependent Zeeman variant
  (`TimeZeeman`/`DiscreteTimeZeeman`/`TimeZeemanPython`/`OscillatingZeeman`)
  are now ported, and `hysteresis`/`hysteresis_loop`/`relax` are now ported
  on `Simulation` (see the Task 9/6 updates above). Spatially varying
  material parameters and regions (Task 16) and LLB/normal-mode consumers
  remain outside this slice. [Claude Sonnet 5]
  DELIBERATE DEVIATION, USER ACCEPTANCE PENDING: the legacy `TimeZeemanPython`
  scalar-spatial-envelope-with-vector-`time_fun` branch (not exercised by any
  legacy test) is not ported; whole-branch review fix round 2 adds a by-name
  gate (`TimeZeemanPython.setup` probes `time_fun(0.0)` and raises a named
  `NotImplementedError` instead of a generic `ValueError`/`TypeError`
  surfacing later), pinned in
  `test_timezeeman_dolfinx.py::test_time_zeeman_python_vector_time_fun_is_deferred_by_name`.
  DELIBERATE DEVIATION, USER ACCEPTANCE PENDING (also carried in
  `transition-notes.org`'s Task 15 section): `DiscreteTimeZeeman.update`
  bypasses `set_value()`, so the cached energy form built once in `setup()`
  never sees a later interval update -- `compute_energy()` silently goes
  stale while `compute_field()`/`energy_density()` stay current, discovered
  building the Task 15 oracle fixture and preserved verbatim, not fixed
  (the `hysteresis`/`hysteresis_loop` no-independent-re-relaxation deviation
  noted in the `Simulation` entry above carries the same USER ACCEPTANCE
  PENDING status). [Claude Sonnet 5]
  Task 16 update: spatially varying `A`/`K1`/`K2`/anisotropy axis/`Ms` are now
  SUPPORTED with the established DG0 (A) / CG1 (K1/K2/axis) / caller-space (Ms)
  placement; spatially varying LLG `alpha` is per-node in the damping term and
  `gamma_LL`; `sim.mark_regions` + per-region `compute_energy(dx=...)` /
  `m_average_in_region` are ported (region-restricted submesh field output
  stays deferred). Axis contract split (`axis_coefficient` in
  `energy_base.py`, register clarification): a *constant* uniaxial-anisotropy
  axis is normalised to a unit vector (restoring the intended legacy cosine
  contract -- a pre-Tier-1/Task 5 deviation from the letter of the legacy
  class); a *spatially varying* axis (Task 16) is used exactly as given, with
  no per-node normalisation -- bug-compatible with legacy, which never
  renormalised an interpolated varying axis either. See
  `transition-notes.org`'s Task 16 "Axis contract" section. Gate:
  `dolfinx-src-varparams-pytest`
  (`test_variable_params_dolfinx.py`, oracle fixtures
  `variable_params_oracle.json`/`spatially_varying_alpha_rhs.json`/
  `cubic_k2_varying_oracle.json`). Only legacy string Expressions, varying
  cubic axes, and matrix/project/direct energy methods remain deferred by
  name; LLB/normal-mode consumers remain outside. [Claude Opus 4.8]
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
  (already-unused) `region` argument rather than silently ignoring it. At the
  time of this Task 6 slice, hysteresis, LLB, and normal-mode consumers
  remained unported (`Simulation` itself is ported as of Task 9/11b/12,
  above). [Claude Sonnet 5]
  Task 15 update: the auto-connection is now exercised with the *real*
  `TimeZeeman`/`OscillatingZeeman` classes (`sim.add(OscillatingZeeman(...))`
  + `run_until`), not only the `FakeTimeZeeman` double used until this slice
  (`test_timezeeman_dolfinx.py::
  test_real_timezeeman_field_changes_during_run_until`); `hysteresis` is now
  ported (see below). LLB and normal-mode consumers remain unported.
  [Claude Sonnet 5]
- The production box foundation deliberately requires a blocked
  three-component CG1 magnetisation space. Some higher-order Lagrange row-sum
  lumped weights are non-positive, so accepting arbitrary elements would fail
  late and potentially asymmetrically across ranks. Invalid volume detection
  is collective as a second guard. [Codex GPT-5.6]
- Demag: FK (Fredkin-Koehler) demag is now ported directly to DOLFINx via the
  array-based native FK BEM routines (Task 11b; see the FK demag baseline note
  above). Task 23 ports the treecode-accelerated FK solver
  (`Demag(solver='Treecode')`) and periodic macro-geometry demag
  (`MacroGeometry`/`macrogeometry=`) on the rebuilt `finmag.native.treecode_bem`
  Cython extension. The pure Python/NumPy Magpar BEM remains reference-only.
  The GCR and 2D (`Demag2D`) variants are still unported and raise by name; the
  `Simulation(pbc='1d'/'2d')` constrained-domain path stays deferred (needs
  `dolfinx_mpc`). [Codex gpt-5.5 high; updated Claude Opus 4.8]
- DMI, cubic anisotropy, variable material parameters, regions, PBC, scheduler
  output, legacy restart files, and production integrators are not covered by
  the current prototype lane. Bulk 3D DMI and constant-axis cubic anisotropy
  are now partial exceptions (see `dmi_energy` and `cubic_anisotropy_energy`);
  interfacial/1D/2D DMI, spatially varying cubic anisotropy, variable material
  parameters, regions, PBC, scheduler output, legacy restart files, and
  production integrators remain uncovered. [GitHub Copilot / Claude Sonnet 5]
- PBC/treecode demag depends on the separate `finmag.native.treecode_bem`
  extension. Task 23 update: this extension is now BUILT for the DOLFINx env
  (Cython/NumPy-2, dolfin-free) and consumed by the ported `TreecodeBEM` and
  `MacroGeometry` surfaces; it is no longer missing in the pixi path. [Codex
  gpt-5.5 high; updated Claude Opus 4.8]
- The current traces and restart states are useful witnesses, but they are not
  replacements for Finmag's NDT, VTK/XDMF, and restart conventions. [Codex
  gpt-5.5 high]
- Task 12 update: restart, NDT, VTK/XDMF, and the scheduler are now ported
  directly to DOLFINx. `src/finmag/sim/sim.py` gains real `save_restart_data`/
  `restart`, `save_averages`/`save_ndt`, `save_field`/`save_m`, `save_vtk`/
  `save_field_to_vtk`, and `schedule`/`unschedule`/`clear_schedule`, with
  `run_until` now driving the ported `finmag.scheduler.Scheduler` event loop
  (unchanged, backend-free Python; only its `sim.py` callers were rewired).
  `run_until` with no schedule is behaviourally identical to the Task 9-11
  direct-advance path. `src/finmag/util/fileio.py` is preserved as the NDT
  read/write contract with two minimal DOLFINx touchpoints: the `aeon` profiler
  dependency is made optional and `np.NAN` -> `np.nan` for NumPy 2. Restart
  format decision (deliberate deviation, documented in `sim_helpers.py` and
  transition-notes): the legacy npz stored the magnetisation as a raw
  backend-dof-ordered owned array (`get_numpy_array_debug()`), which silently
  misassigns on a reordered rebuild of the same mesh; the port instead stores a
  coordinate-aware v2 layout (owned vertex coordinates + coordinate-ordered
  values + mesh hash + `Ms`/`alpha`/`gamma`/`unit_length` + interaction list +
  `simtime`/`driver='scipy'`) and remaps by coordinate on load, correctly
  restoring a same-recipe rebuild and loudly rejecting a genuine mesh mismatch
  with `ValueError`. VTK/XDMF route through the write-only `Field.save_pvd`/
  `save_xdmf`; read-back stays unavailable by name (`Field.save_hdf5` raises).
  `sim_helpers.py` is now DOLFINx-import-clean (the dolfin-dependent
  skyrmion/submesh/normal-mode helpers, already dropped from the ported
  `Simulation` in Task 9, were removed). Gate:
  `dolfinx-src-restart-output-pytest`. [Claude Opus 4.8]

- P1 lifecycle resolution: `3f4ed4ea` (P1.1) removes the SciPy-only reset
  assumption by passing `t0` through the driver factory without disturbing
  SciPy's positional factory slots; a trajectory-continuity regression pins
  reset semantics. `17f24413` (P1.2) writes truthful
  `sim.integrator_backend` provenance to v2 archives, keeps writable
  `sim.driver` synchronized, and validates immediate plus
  uninterrupted-vs-restarted trajectories on SciPy and Sundials. V2-only
  rejection, metadata handling and mesh-mismatch semantics were deliberately
  unchanged. Focused P1.1: 76 passed/2 skipped. P1.2: restart 27,
  Simulation 33, Sundials 22, SciPy 21 passed/2 skipped, LLG 16; aggregate
  verifier 32 steps green. `81fab481` (P1.3) then restores native Sundials as
  the public `Simulation`/`sim_with` default, while retaining explicit SciPy
  support. The default genuinely constructs and advances `SundialsIntegrator`;
  the core smoke JSON records `"integrator_backend": "sundials"` at `t=1e-12`.
  Focused P1.3: Simulation 37, Sundials 22, SciPy 21 passed/2 skipped,
  restart/output 27; final clean main-worktree aggregate 32 steps green. The
  final fast lane was 14 passed/3 skipped in 246.36s
  (`/tmp/finmag-p1-default-m5.log`). The frozen P0.2 FULL lane
  remains 12 passed/5 failed.
  [Codex GPT-5.6]

- `src/finmag/util/meshes.py` and `src/finmag/util/mesh_templates.py` are now
  the direct DOLFINx port of the mesh-generation surface (Task 18). The Netgen
  CSG + CLI + dolfin-XML toolchain is replaced by the Gmsh Python API
  (OpenCASCADE kernel) converted to `dolfinx.mesh` in-memory via
  `dolfinx.io.gmsh.model_to_mesh` (the DOLFINx 0.10 module name; `gmshio` in
  older releases). Public generator names/signatures are preserved
  (`box`/`sphere`/`cylinder`/`nanodisk`/`elliptic_cylinder`/
  `elliptical_nanodisk`/`ellipsoid`/`truncated_cone`/`ring`/`pair_of_disks`),
  as are the template classes (`Sphere`/`Box`/`EllipticalNanodisk`/`Nanodisk`/
  `MeshSum`/`MeshDifference`) with their `csg_string()`/`hash()`/
  `generic_filename()` semantics *exactly* (the Netgen-CSG text is retained
  byte-for-byte as the md5 cache key -- the frozen `test_hash` digests still
  hold -- while geometry is built through Gmsh OCC from stored parameters).
  `gmsh` is lazily imported so `import finmag` never pulls it in
  (plain-import boundary intact). The md5-of-CSG caching contract is preserved
  on a DOLFINx-native XDMF store (`.xdmf`+`.h5`; first *internal* XDMF read,
  not general read-back -- Task 26). `mesh_volume`/`num_vertices`/
  `order_of_magnitude` are ported dolfinx-native. Deferred by name (Task 29
  review items): the Netgen backend (`netgen_is_usable()` returns `False`),
  `nmesh_to_dolfin.py` (legacy dolfin-XML emitter, consumed only by the
  unported Nmag harness -- Task 27), the textual `from_geofile`/`from_csg`
  entry points, the multi-region airbox generators, the 2D gmsh-script
  helpers, and the dolfin-based analysis/plotting utilities. `gmsh` +
  `python-gmsh` added to the dolfinx pixi feature (`pixi.lock` changed). Gate:
  `dolfinx-src-meshes-pytest`. [Claude Opus 4.8]
- Task 18 fix round 1 (review findings on commit `867c9ca1`): the
  dolfin-based analysis/plotting deferrals above are, explicitly, `mesh_info`,
  `mesh_quality`, `nodal_volume`, `longest_edges`, `mesh_size`,
  `mesh_size_plausible`, `describe_mesh_size`, `print_mesh_info`,
  `mesh_is_periodic`, `build_mesh`, `embed3d`, `line_mesh`, `plot_mesh`,
  `plot_mesh_with_paraview`, `plot_mesh_regions` -- the last five
  (`mesh_size_plausible`/`describe_mesh_size`/`print_mesh_info`/
  `plot_mesh_with_paraview`/`plot_mesh_regions`) had been deleted outright
  (bare `AttributeError`) instead of getting by-name stubs like their
  siblings; they now do. `ring(with_middle_plane=True)` was a legacy kwarg
  accepted but silently ignored; it now raises `NotImplementedError` by name
  (`with_middle_plane=False`, the default, is unaffected) -- both were
  by-name-deferral narrowings the Phase 3 review rule requires fixing. A new
  cache-drift guard test (`test_csg_occ_cache_drift_guard`) pins that no
  `Sphere`/`Box`/`EllipticalNanodisk`/`MeshSum`/`MeshDifference` constructor
  parameter can move the OCC geometry without also moving the md5 hash. The
  legacy `test_mesh_sum` TOL2 fused-vs-separate-spheres cross-check is now
  ported into `test_template_mesh_sum_volume` (Gmsh OCC measures ~5.24e-6
  relative deviation for r1,r2,r3=10,18,12 at maxh=2.0, within the port's
  declared `TOL2=1e-5`). Gate: `dolfinx-src-meshes-pytest`, 34 tests.
  [Claude Sonnet 5]
- Task 19 update: `ThinFilmDemag` and `FixedEnergyDW` were the last two
  optional energy classes still raw-importing legacy `dolfin` at module
  scope. `ThinFilmDemag` (`Hi = -strength_i * m_i` for a single axis, with
  `strength` defaulting to a box-averaged `Ms` -- the same lumped nodal-
  average math `EnergyBase`'s box-assemble path uses elsewhere) is now
  PORTED directly: constructor semantics, the `setup(m, Ms, unit_length)`
  contract, and both the default (`Ms`-average) and explicit-
  `field_strength` branches are transcribed exactly, validated against the
  legacy `thin_film_demag_test.py` invariants plus two legacy oracle
  fixture cases (`thin_film_demag_oracle.json`, regenerated by
  `gen_thin_film_demag_oracle.py`: a spatially-varying-`Ms` default-strength
  case and an explicit-scalar-`field_strength` case). `compute_field()`
  returns the legacy component-blocked, owned-vertex-ordered `xxx` layout
  (restored across all public field-array surfaces by Task 31, below); the
  interim Task 19 interleaved-order deviation is CORRECTED, so this is no
  longer a serialisation deviation.
  `compute_energy()` keeps legacy's literal `0` (this approximation has no
  associated energy functional). `FixedEnergyDW` -- untested even on legacy
  master (no `dw_fixed_energy_test.py` exists anywhere in the legacy tree,
  and legacy's own todo notes record "the computation with the
  FixedEnergyDW class is broken") -- is converted to a curated by-name
  `NotImplementedError` deferral instead of a faithful port (Task 29 review
  item): it round-trips a hand-duplicated mesh through a bespoke dolfin-XML
  writer/reader that legacy's own docstring already flags as broken. (It also
  constructs `Demag(solver='Treecode')`, which Task 23 has since ported -- so
  the deferral now rests solely on the broken dolfin-XML round-trip and the
  total absence of any legacy test, not on the demag solver.) Neither module imports legacy
  `dolfin` any more; zero `finmag.energies` public names remain
  `requires_legacy_dolfin=True` (`energies/__init__.py`'s `_LAZY_EXPORTS`
  table). Legacy never wired `ThinFilmDemag` into `Simulation`/`sim_with`
  (only ever `sim.add(ThinFilmDemag())` directly, in the one notebook that
  used it), so no `sim_with` parameter was added. Gate: folded into
  `dolfinx-src-energies-pytest` (two new test files,
  `test_thin_film_demag_dolfinx.py` and `test_dw_fixed_energy_dolfinx.py`,
  added to the same pixi command rather than a new gate); 45 tests total.
  [Claude Sonnet 5]
- Task 21 update: packaging is no longer a gap. `src/finmag` installs as a
  regular editable Python package in the `dolfinx` pixi environment via
  `pyproject.toml` (setuptools, src-layout); every `dolfinx-src-*` gate now
  runs against that installed package. `PYTHONPATH=src` is no longer the
  mechanism the port relies on -- it is retained only as one fallback gate
  step (`dolfinx-src-import-pythonpath-fallback`) exercised alongside the
  installed-package path, not as the primary install story. See
  `transition-notes.org`'s "Packaging (Task 21)" section and `README.md`'s
  "Installing the DOLFINx port (pixi)" subsection for the exact commands and
  the dependency-split/version-split/wheel-non-goal rationale. [Claude
  Sonnet 5]
- Task 30 (examples conversion, practical-parity witness): the legacy
  `examples/` scripts now run on the ported package. 18 scripts converted with
  minimal diffs (import/mesh/Expression-callable/print fixes only); the (b) set
  (`dispersion_curves` T25, `llb` T24, `nmag_example_2`/nmag-comparison T27,
  `boost_python` C++ tutorials, mayavi/vpython viz T26) is annotated
  "requires Task N", not force-ported. Each converted script embeds its own
  validation (checked-in reference data for `exchange_demag`; analytic for
  `macrospin`/`demag`; µMAG for `std_prob_3/4`; physical-sanity otherwise) and
  is run as a subprocess by the new `dolfinx-src-examples-pytest` gate
  (14 fast + 3 FULL-only), folded into `verify-dolfinx-m5`. 11 interface-drift
  findings are recorded in `transition-notes.org` ("Examples conversion (Task
  30)"), the user's headline deliverable. Two `src/finmag` changes beyond the
  documented example diffs, both flagged for review: (1) **controller-approved**
  — `finmag.util.meshes.from_geofile`/`from_csg` un-deferred for the Netgen-CSG
  subset the examples use (new `src/finmag/util/geofile.py`, built on the Task
  18 OCC/Gmsh path; netgen-BINARY backend still deferred), so `from_geofile`
  lines stay untouched; (2) a genuine robustness BUG fix in
  `field.py::_owned_vertex_to_dof` (exact 12-decimal coordinate matching crashed
  on Gmsh/`from_geofile` meshes; replaced with a tolerance-based nearest-vertex
  match that is exact for `create_box` and robust to generator FP noise). No
  legacy oracle tests touched; `pixi.lock` untouched. [Claude Opus 4.8]
- Task 31 (public field-array component-ordering correction, user-directed):
  the interleaved-ordering interface drift is DE-REGISTERED. Every PUBLIC
  field-array surface -- each interaction's `compute_field()`,
  `EffectiveField.H_eff`/`compute()`/`compute_jacobian_only()`,
  `sim.effective_field()`, and `Field.get_numpy_array_debug()` (with
  `set_with_numpy_array_debug` kept as its exact inverse) -- now returns the
  legacy component-blocked ordering `[x1..xN, y1..yN, z1..zN]`, the
  owned-vertex-coordinate-ordered `xxx` view (`xyz.reshape(-1,3).T.reshape(-1)`,
  the same ordering `sim.m`/`llg.m` already return). Internal DOLFINx storage
  (node-interleaved backend order, still via `Field.as_array()`) is UNCHANGED.
  A single shared helper `finmag.field.owned_raw_to_blocked(functionspace,
  raw_array)` (reusing the canonical `_owned_vertex_to_dof` permutation) applies
  the conversion at each class's public return; `EnergyBase`/`CubicAnisotropy`/
  `FKDemag` keep an internal raw `_compute_field_raw()` for order-invariant means
  and Function writes. `LLG.solve`/`sundials_jtimes` DROPPED their former
  raw->xxx `H_eff` conversion (double-conversion was the failure mode).
  Drift rows #3 (`compute_field` ordering) and #6 (`get_numpy_array_debug`
  ordering) in the Task 30 table are CORRECTED; both example workarounds
  reverted to their legacy one-liners. New anti-scramble gate
  `test_ordering_contract_dolfinx.py` (4 tests, demonstrably RED pre-fix); full
  `verify-dolfinx-m5` green. See `transition-notes.org`'s "Public field-array
  component-ordering correction (Task 31)" section. [Claude Fable 5]
- SR1 P2.1 (core example/logging import boundary, `626dfcc8`): `finmag.example`
  and `finmag.set_logging_level` now work in the DOLFINx environment with no
  legacy `dolfin` installed. `bar`/`barmini`/`nanowire` build their box meshes
  with `dolfinx.mesh.create_box`, preserving every signature, dimension,
  discretisation and `sim_with` keyword (coordinates stay in nm,
  `unit_length=1e-9`); `nanowire`'s unused `S1`/`S3` spaces are dropped.
  `set_logging_level` moves into a new stdlib-only
  `finmag.util.logging_helpers`, re-exported from `finmag.util.helpers` so the
  legacy spelling keeps working; its lazy export stays flagged as needing legacy
  dolfin so the historical dof-ordering preparation still runs where dolfin IS
  installed. The unported `NormalModeSimulation`, `normal_mode_simulation`,
  `example.sphere_inside_airbox` and `example.normal_modes` are resolved at the
  lazy import boundary and raise a curated `NotImplementedError` naming the
  deferred feature instead of leaking `ModuleNotFoundError: No module named
  'dolfin'`. Gate: new `src/finmag/tests/test_example_dolfinx.py` (clean
  subprocesses, so `sys.modules` is an exact witness) folded into both
  `dolfinx-src-import-pytest` and the legacy `src-import-pytest` -- the latter
  is the only lane where the `finmag.util.helpers` re-export check can execute.
  RED was 10 failing tests, reviewer-reproduced; the focused gate went from
  9 passed/2 skipped to 19 passed/3 skipped in 23.65s, with
  `dolfinx-src-simulation-pytest` 37 passed and `dolfinx-src-deferred-pytest`
  1 passed/4 skipped unchanged. Stated limits: the guarded legacy-resolution
  path is forward-looking insurance no materialised environment can exercise
  (not verified), and the intra-cuboid diagonal orientation convention of
  `create_box` versus legacy `df.BoxMesh` is unverified without legacy dolfin
  (vertex counts and 6 tets per cuboid do match). Deferred: the
  `from finmag.example.normal_modes import disk` submodule spelling, the
  `nanowire` `name` argument never reaching `sim_with` (faithful to master
  `b5015c5a`), and the full port of `finmag.util.helpers`. See
  `transition-notes.org`'s "P2.1 core example/logging import boundary"
  section. [Claude Opus 4.8]
- SR1 P2.2 (`sim_with` MacroGeometry wiring, `2cf4647f`, witness corrections
  `f97d4088`/`571df59f`, guard hardening `899d3906`): `sim_with(nx=, ny=,
  spacing_x=, spacing_y=)` now forwards its four arguments unchanged into
  `MacroGeometry(nx=nx, ny=ny, dx=spacing_x, dy=spacing_y)` on the
  already-ported dense-FK path (Task 23 above), matching the legacy contract
  (`git show b5015c5a:src/finmag/sim/sim.py`, lines 1443-1447) exactly: no
  transformation, no `unit_length` scaling. `spacing_x`/`spacing_y` are the
  tile PITCH (centre-to-centre translation of the image lattice) in mesh
  coordinate units, not a gap; the docstring now says so explicitly. A
  `MacroGeometry` is built only when at least one of the four arguments is
  given (numerically inert for the 1x1 case, agrees with plain FK to
  1.005e-15 relative; preserves the import boundary for the plain default).
  A new guard, `_reject_touching_macro_geometry`, refuses by name any pitch
  at or below the mesh extent on an active axis (`pitch <= extent`) --
  hardened in `899d3906` to also cover the overlapping case (`pitch <
  extent`), not just the exactly-touching legacy default -- because the
  ported periodic BEM double-counts the solid angle at coincident/
  interpenetrating tile boundary nodes (row sums reach -2/-3; ~158% error on
  a cube; non-finite on a flat slab, failing the phi_2 solve with
  `KSP_DIVERGED_NANORINF` and silently returning `H = -M`). This is a
  DELIBERATE DIVERGENCE from the legacy default, recorded as D17 in
  `acceptance-register.md`, disposition pending owner decision; fixing the
  underlying coincident-node BEM defect itself is a demag-algorithm change
  and stays out of scope here -- it is refused by name and pinned (strict
  `xfail` plus a direct regression test), not fixed.

  A preceding correction, P2.2a (`f97d4088`, completed `571df59f`), found and
  replaced three PBC-path tests that were counting the touching
  `MacroGeometry` default as passing physics evidence when in fact that
  configuration's periodic BEM is non-finite, the phi_2 Krylov solve fails
  with `KSP_DIVERGED_NANORINF`, and the silently-returned `H = -M` happened
  to satisfy the old assertions. The replacement witnesses use a
  non-coincident pitch `extent * (1 + 1e-6)` with explicit BEM-finiteness and
  row-sum preconditions; the touching expectation is retained as a strict
  `xfail`, and the defect is pinned directly by a new
  `test_pbc_coincident_tile_spacing_produces_a_non_finite_bem`.

  Evidence: cross-geometry equivalence (legacy `demag_pbc_test.py` acceptance
  check) -- a 3x-tiled 20nm cube (`nx=3, spacing_x=20.001`) reproduces a
  directly-meshed 60x20x20nm bar to 0.08% in Hx / 0.011% in Hz against legacy
  bounds 1%/2%; thin-film analytic limits both directions at the gapped
  pitch (`Nz`: 0.709851 -> 0.983976 -> 0.990758; `Nx`: 0.054464 -> 0.007920
  -> 0.004588, nx=ny=1,3,5), independently reproduced by review, which
  confirmed the gap-limit is continuous from the right so the 1e-6 gap is
  real physics, not an artefact. Gate: `dolfinx-src-simulation-pytest` 44
  passed (was 37); `dolfinx-src-treecode-pytest` 22 passed/1 xfailed (was 20
  passed); `dolfinx-src-demag-pytest` 18 passed and `dolfinx-src-import-pytest`
  19 passed/3 skipped unchanged. Review: APPROVE WITH FOLLOW-UPS -- findings
  1 (a second touching-default witness) and 2 (the guard originally missed
  the overlapping case) were acted on in `571df59f`/`899d3906`; finding 3
  (the guard over-rejects a vanishingly small ~5e-10 gap band that would
  technically work) is left as deliberate benign conservatism. No Treecode
  factory exposure, GCR or Demag2D work. See `transition-notes.org`'s "P2.2
  sim_with MacroGeometry wiring" section. [Claude Sonnet 5]
- SR1 P2.4 (correct discrete-time Zeeman energy, `ff906f11`, register D3):
  `DiscreteTimeZeeman.update()` previously rebound `self.H` to a brand-new
  `Field` on each interval crossing, bypassing `set_value()`, so the cached
  UFL energy form `self.E` (built once in `setup()` against the original `H`
  `Function`) kept assembling the discarded setup-time field forever --
  `compute_energy()` froze at `-8.042477193189932e-23` while
  `compute_field()`/`energy_density()` stayed current: 16.7% relative error at
  the first crossing, 50% at t=1ns, unbounded in general (100% wrong when
  `H(0)=0`, wrong sign when `m.H(t)` flips). The fix routes the refresh
  through `self.set_value(self.field_function(t))`, exactly as the base
  `TimeZeeman.update` does: `self.H` is written in place and `self.E` is
  re-formed against it. Measured field bit-invariance (max|difference| = 0.0
  in `compute_field()`/`average_field()` for both a constant-vector and a
  spatially-varying callable contract) proves the fix changes no field value;
  dynamics, effective field and every `.ndt` trajectory stay bit-identical,
  and only `compute_energy()` is corrected. The legacy stale value is
  preserved as an explicit divergence pin
  (`_D3_LEGACY_STALE_ENERGY = -8.042477193189932e-23`,
  `test_discrete_time_zeeman_energy_diverges_from_legacy_stale_value`) rather
  than dropped; the committed oracle fixture
  (`fixtures/timezeeman_oracle.json`) does NOT numerically discriminate the
  original defect (its correct energies are ~1e-37 J against `atol=1e-18`),
  so `test_oracle_discrete_time_zeeman_sequence_matches_legacy` passes
  unchanged. Gate: focused `dolfinx-src-timezeeman-pytest` went from 28 passed
  to 32 passed (the staleness quirk pin replaced by the D3 divergence pin plus
  4 new corrected-behaviour/invariance/switch-off/analytic tests);
  `dolfinx-src-energies-pytest` 45 passed; `dolfinx-src-simulation-pytest` 37
  passed. Independent review: APPROVE, nothing blocking (reproduced the
  multi-crossing/sign-flip tracking, the field bit-invariance, and confirmed
  `E_now/E_legacy == 2.0` is a robust structural fact, not a coincidence). The
  clean-tree aggregate `dev/bin/verify-dolfinx-m5` on the committed worktree
  exited 0 with all 32 steps green (log `/tmp/finmag-p24-m5-clean.log`). D4
  (hysteresis no-re-relax) is untouched and remains approved but unimplemented
  as SR1 P2.5. See `transition-notes.org`'s "P2.4 discrete-time Zeeman energy
  correction" section. [Claude Sonnet 5]
