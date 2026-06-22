# Finmag Project Notes

## High-Level Summary

Finmag is a finite-element micromagnetics codebase built on top of FEniCS/dolfin. The public entry point is the `Simulation` class, which orchestrates:

- mesh and function-space setup,
- magnetisation and material fields,
- effective-field construction from interactions,
- time integration,
- scheduled output and data saving.

## Strategic Milestones

The project now has four explicit transition milestones:

1. make Finmag usable under Python 3 on the current legacy `dolfin 2017.1.0`
   transition stack;
2. provide a non-container install path, preferably via `pixi`;
3. migrate from legacy `dolfin 2017.1.0` to legacy FEniCS/`dolfin 2019.1.0`;
4. later prototype and expand a separate `dolfinx` port.

Working rule for future agent effort:

- do not revive historical features or clean up old xfails unless that work
  directly advances one of those milestones.

Current model-label convention:

- the Codex app currently reports `gpt-5.5 high` for this workstream
- use `gpt-5.5 high` in commit-message attribution and new agent-authored
  signatures until the user updates this information
- older `[Codex GPT-5.4]` comments are historical labels and should not be
  treated as the current runtime model

This repository contains many research-era features that may never be needed
again. The transition should optimize for usable scientific workflows, not for
maximum historical surface area.

## Project Documents

The transition work uses three top-level project notes with distinct roles:

- `plan.org`
  - forward-looking roadmap
  - milestones, scope, sequencing, acceptance criteria, and priorities
- `transition-notes.org`
  - factual engineering log
  - environment findings, exact versions, technical constraints, verified
    baselines, and transition-specific discoveries
- `agents.md`
  - agent orientation and working conventions
  - repository summary, architecture pointers, local rules, and collaboration
    guidance

Short version:

- `plan.org` = intent
- `transition-notes.org` = evidence
- `agents.md` = operational context

Workflow witnesses:

- `dev/bin/verify-python3-m1`
  - Docker / legacy-DOLFIN aggregate witness
- `dev/bin/verify-python3-m2`
  - pixi / FEniCS-2019 aggregate witness
- `dev/bin/verify-python3-m3`
  - first explicit pytest gate for the FEniCS-2019 compatibility track
  - currently includes the historical `barmini` subset plus passing
    `sim_ode`, applied-field, exchange-static, `test_llg.py`, the full
    `test_sundials_ode.py` file, and the current anisotropy/DMI/energy
    creation layer under FEniCS-2019, plus the current Heun / varying-alpha /
    unit-length / interaction-linearity / common-energy regression slice and
    the currently compatible FK demag / thin-film / domain-wall / region-energy
    slice
  - the earlier SUNDIALS-7 crash in `test_simple_1d` is fixed by explicitly
    attaching the matching nonlinear solver on the modern wrapper path. [Codex GPT-5.4]
  - `demag_pbc_test.py` is now included in M3 as a collected skip when the
    pixi native build lacks `treecode_bem`; treat the native extension as a
    separate build task, not a broad M3 blocker. [Codex gpt-5.5 high]
- `.github/workflows/python3-m1.yml` and `.github/workflows/python3-m2.yml`
  - matching lightweight CI jobs for the two workflow witnesses
- `.github/workflows/python3-m3.yml`
  - matching lightweight CI job for the first FEniCS-2019 pytest gate

The repository describes itself as a research prototype developed roughly from 2011 to 2018.

## Repository Shape

Important top-level directories:

- `src/finmag`: main Python package
- `native`: compiled extension modules and native tests
- `examples`: user-facing examples and reproductions
- `doc`: documentation and notebooks
- `install`: installation scripts, Docker files, environment setup
- `dev`: sandbox, experiments, historical artifacts

The codebase is large. `src/finmag` alone contains 265 Python files.

## Core Runtime Architecture

### Public API

Main package exports come from:

- `src/finmag/__init__.py`
- `src/finmag/init.py`

Primary user-facing objects:

- `finmag.Simulation`
- `finmag.sim_with(...)`
- `NormalModeSimulation` and normal-mode helpers

### Main Control Flow

The usual runtime path is:

1. create a `dolfin` mesh;
2. create `Simulation(mesh, Ms, ...)`;
3. set initial magnetisation;
4. add interactions such as exchange, demag, anisotropy, Zeeman, DMI;
5. integrate with a Sundials or SciPy backend;
6. save averages, fields, VTK, restart data, or scheduled outputs.

### Key Modules

- `src/finmag/sim/sim.py`
  Main orchestration object. Manages mesh, kernel, integrator, scheduler, output writers, and convenience APIs.

- `src/finmag/field.py`
  Wrapper around `dolfin.Function`. Central abstraction for scalar/vector fields and NumPy/FEM conversions.

- `src/finmag/physics/llg.py`
  Landau-Lifshitz-Gilbert dynamics and links to effective-field evaluation.

- `src/finmag/physics/effective_field.py`
  Owns the active interactions and computes the total effective field by summing interaction contributions.

- `src/finmag/energies/*`
  Interaction modules such as `Exchange`, `Zeeman`, anisotropy, DMI, and demag.

- `src/finmag/energies/energy_base.py`
  Common field/energy assembly base class used by several interactions.

- `src/finmag/drivers/*`
  Time integration backends. `SundialsIntegrator` is the default path for the core workflows tested so far.

- `src/finmag/scheduler/*`
  Event scheduling for output, checkpointing, and stop conditions.

- `src/finmag/util/fileio.py`
  `.ndt` table output via `Tablewriter` and related IO helpers.

## Instructions for changes by agent

When you make changes, add a short comment to explain the new code. At
the end of the comment or comment line, add your model number such as
"[Codex GPT-5.4]"

## Convenience Example: `barmini`

`barmini` is an especially useful integration fixture.

Location:

- `src/finmag/example/bar.py`

What it does:

- creates a small 3x3x10 nm bar mesh;
- uses `sim_with(...)`;
- sets `Ms`, `alpha`, `A`, initial magnetisation;
- includes demag by default.

Why it matters:

- it is fast enough for repeated testing;
- it exercises `Simulation`, common energies, Sundials integration, and output paths;
- the current `pixi` milestone is stronger than the earlier SciPy-only probe:
  the verified `pixi run barmini-smoke` path now uses full default `barmini()`
  with FK demag and the native Sundials 7 backend. [Codex GPT-5.4]
- `pixi run restart-smoke` is also verified now, so the pixi path covers both
  time integration and restart/save-load on the Sundials backend. [Codex GPT-5.4]
- several existing tests already use it.

Barmini-based tests identified in the repo include:

- `src/finmag/tests/test_effective_field.py`
- `src/finmag/tests/test_writing_data.py`
- `src/finmag/drivers/tests/test_integrator_raises_exception_on_exceed_maxsteps.py`
- `src/finmag/tests/test_restart_simulation.py`
- plus many broader tests in `src/finmag/sim/*`, `src/finmag/tests/bugs/*`, and utility tests

## Native Code

The project is not pure Python. Native code is built via:

- `native/Makefile`

Compiled outputs are installed into:

- `src/finmag/native`

Modules mentioned by the native build include:

- `sundials.so`
- `llg.so`
- `llb.so`
- `fast_sum_lib.so`
- `treecode_bem.so`
- `neb.so`

The package also has runtime logic that may attempt to build native modules on import through:

- `src/finmag/native/init.py`
- `src/finmag/util/native_compiler.py`

This is operationally important when running tests from writable vs non-writable locations.

## Environment Findings

### Reference Image

Known-good container:

- `finmag/finmag:latest`

Running the agreed minimal acceptance suite inside the image's own `/finmag` tree succeeded:

- 21 tests passed
- runtime about 66 seconds

### Bind-Mount Caveat

Running the same test suite with the host checkout bind-mounted into the container failed because:

- the container attempted to rebuild native modules into the mounted tree;
- writes to generated artifacts such as native outputs and compiler logs failed with permission errors.

Conclusion:

- use the in-image checkout as the current oracle;
- for local edits, copy the checkout into a writable directory inside the container before running tests.

### Python 3 in the Image

The image also contains Python 3.5.x, but not Python 3 DOLFIN bindings.

Observed behavior:

- `python` imports `dolfin 2017.1.0`
- `python3` exists
- `python3` does not import `dolfin`

This is an important migration constraint:

- Python 3 syntax and compatibility work can proceed in-container,
- but a real Python 3 Finmag runtime will require a separate strategy for obtaining `dolfin` under Python 3.

## Minimal Acceptance Suite

Current agreed suite:

- `src/finmag/tests/test_effective_field.py`
- `src/finmag/tests/test_writing_data.py`
- `src/finmag/drivers/tests/test_integrator_raises_exception_on_exceed_maxsteps.py`
- `src/finmag/tests/test_restart_simulation.py`
- `src/finmag/tests/test_sim_ode.py`
- `src/finmag/tests/test_applied_field.py`
- `src/finmag/tests/test_exchange_static.py`
- `src/finmag/tests/test_llg.py`
- `src/finmag/util/ode/tests/test_sundials_ode.py`

This suite covers:

- package import
- `Simulation`
- `barmini`
- `LLG`
- `EffectiveField`
- `Exchange`
- `Zeeman`
- Sundials ODE integration
- restart handling
- `.ndt` output writing

## Python 3 Porting Observations

The code is strongly Python 2 era.

Common breakpoints identified in the source tree:

- implicit relative imports such as `from init import *`
- package-relative imports written as bare imports, for example in `init.py`
- `basestring`
- `xrange`
- `.iteritems()` / `.itervalues()`
- old exception syntax such as `except RuntimeError, ex`
- Python 2 `print` statements

These are widespread, but the most important early files are concentrated in the package/import layer and the core runtime path.

## Recommended Port Order

1. package/import layer
2. `Field`, helpers, file IO
3. `EffectiveField`, `LLG`, drivers, scheduler, `Simulation`
4. common energies
5. native build/import glue

Do not mix this with a `dolfin` migration until the Python 3 port is stable.

## Python 3 Runtime Progress

The Python 3 path has moved past the initial package-layer work.

Confirmed working environment:

- image: `finmag-py3-dolfin2017`
- `python3 = 3.5.4`
- `dolfin.__version__ = 2017.1.0`

The image also needs:

- `OMPI_MCA_plm=isolated`

for reliable `import dolfin` inside Docker.

The current Python 3 progress is now stronger than plain import:

- `import finmag` works
- `finmag.example.barmini()` constructs
- `sim.run_until(1e-12)` works
- the first `barmini`-based acceptance subset passes under Python 3:
  - `src/finmag/tests/test_effective_field.py`
  - `src/finmag/tests/test_writing_data.py`
  - `src/finmag/drivers/tests/test_integrator_raises_exception_on_exceed_maxsteps.py`
  - `src/finmag/tests/test_restart_simulation.py`
  - total: `7 passed`
- the full agreed minimal acceptance suite also passes under Python 3:
  - `21 passed`
  - `python3-scipy` and `python3-matplotlib` are installed in the transition
  image
- `test_sim_ode.py` intentionally keeps a lazy `matplotlib` import because
  plotting is optional and only used for `do_plot=True`
- the broader Python 3 core transition suite also passes:
  - `356 passed, 16 skipped, 6 xfailed, 13 warnings`
  - includes `field_test.py`, `field_setters_test.py`,
    `scheduler/scheduler_test.py`, `energies/exchange_test.py`,
    `energies/anisotropy_test.py`, `energies/zeeman_test.py`,
    `energies/dmi_test.py`, the util batch (`fileio`, `helpers`,
    `length_scales`, `meshes`, `mesh_templates`, `tests/test_meshes.py`,
    `pbc`, `plot_helpers`,
    `vtk_saver`, `dmi_helper`, `set_function_values`, `DMI_from_helix`),
    relax/restart/time regressions,
    `sim/hysteresis_test.py`, `sim/magnetisation_patterns_test.py`,
    full `sim_helpers_test.py`, full `sim/sim_test.py`,
    Sundials reinit and stiff/scipy ODE driver coverage,
    `drivers/tests/test_integrators.py`,
    `tests/jacobean/test_jacobean_computation.py`,
    `tests/jacobean/test_jacobean_integration.py`, and
    `tests/jacobean/test_native_llg.py`,
    `physics/tests/test_effective_field.py`,
    `physics/tests/test_equation.py`,
    `physics/tests/test_terms.py`,
    `energies/cubic_anisotropy_test.py`,
    `energies/test_energies_in_regions.py`,
    `energies/magnetostatic_field_test.py`, and
    `energies/thin_film_demag_test.py`,
    `tests/test_energy_creation_with_variable_Ms.py`,
    `tests/test_interactions_scale_linearly_with_m.py`,
    `tests/test_heun.py`,
    `tests/test_spatially_varying_alpha.py`,
    `tests/test_spatially_varying_anisotropy.py`, and
    `tests/test_unit_length.py`,
    `tests/test_anis.py`,
    `tests/test_dmi.py`, and
    `tests/test_dmi_terms.py`,
    `util/fft_test.py`,
    `energies/demag/demag_pbc_test.py`,
    `energies/demag/fk_demag_test.py`, and
    `energies/demag/fk_demag_2d_test.py`,
    `tests/demag/test_bem_computation.py`, and
    `tests/demag/test_demag_sphere.py`
- additional targeted progress after that last full rerun:
  - `energies/demag/fk_demag_2d.py` had a real 2d->3d lifting bug fixed; the
    corresponding `fk_demag_2d_test.py` is now a real passing regression test
  - `sim/sim.py` now enables extrapolation when interpolating fields onto
    regional submeshes, which unblocks
    `energies/zeeman_test.py::test_compare_stray_field_of_sphere_with_dipolar_field`
  - `tests/test_jacobian.py` now collects and passes under Python 3 after a
    Python 2 syntax cleanup
  - `sim/normal_mode_sim.py` and `util/fft.py` now support the mesh-region PSD
    path on this DOLFIN/Python 3 stack
  - `Simulation.profile()` now works under pytest because profiling uses
    `Profile.runctx(...)` with an explicit `sim` binding
  - `sim_test.py::test_setting_different_material_parameters_in_different_regions`
    is now a real initialisation test that probes the region-dependent fields
    directly, rather than a Paraview-dependent placeholder ending in
    `NotImplementedError`
  - `sim_test.py::test_regression_schedule_switch_off_field` now passes after
    fixing a typo in `Simulation.remove_interaction()` that left stale
    `Tablewriter` callbacks behind after removing the Zeeman interaction
  - `test_energies_in_regions.py::test_energies_in_separated_subdomains`
    now passes; its old self-skip was stale
  - `helpers_test.py::test_apply_vertexwise` now passes after updating
    `helpers.apply_vertexwise()` for the current DOLFIN API
  - targeted verification for this batch:
    `6 passed in 91.45s`
  - `tests/test_jacobian.py` has been added to the active Python 3 core-suite
    verifier and CI workflow
  - follow-up targeted verification:
    `2 passed in 62.50s`
    for `sim_test.py::test_compute_and_plot_power_spectral_density_in_mesh_region`
    and `sim_test.py::test_profile`
  - final substantive `sim_test.py` xfail targeted verification:
    `1 passed in 57.38s`
  - follow-up regression verification:
    `1 passed in 56.12s`
  - further non-GUI targeted verification:
    `1 passed in 64.48s` for
    `energies/test_energies_in_regions.py::test_energies_in_separated_subdomains`
  - further non-GUI targeted verification:
    `1 passed in 79.23s` for
    `util/helpers_test.py::test_apply_vertexwise`

The current Python 3 transition gate is:

- `.github/workflows/python3-core-suite.yml`
- `dev/bin/verify-python3-core-suite`

Intentional skips and xfails are explicit:

- the transition image now contains `netgen`, `tix`, `xauth`, and `xvfb`, but
  the historical `netgen 4.9.13` still aborts after some mesh-generation runs;
  tests therefore use a `netgen_is_usable()` probe instead of checking only
  whether the binary is installed
- the mesh-generation path is nevertheless usable again in the transition
  image: Finmag now accepts post-export `netgen` aborts when the mesh file is
  already present, and falls back to the Python meshconvert module when the
  `dolfin-convert` script is missing
- `dolfinh5tools` is an external package historically installed alongside
  Finmag, and it backs `Field.save_hdf5(...)`
- the Python 3 transition image now installs `dolfinh5tools` from GitHub and
  patches its Python-2-style package import so `Field.save_hdf5(...)` is green
  in the active core suite
- the Python 3 transition image now also installs `gmsh`
- `regular_polygon(...)` and `regular_polygon_extruded(...)` no longer require
  the standalone `dolfin-convert` script; they use the existing Python
  meshconvert fallback
- `util/meshes_test.py::test_build_mesh` no longer requires `mshr`; it uses
  Dolfin-native rectangle, disk, and box meshes and passes in the transition
  image
- `sim_test.py::test_m_average_is_robust_with_respect_to_mesh_discretization`
  now uses `gmsh` and the shared Python meshconvert fallback directly, so it no
  longer needs the `sh` Python package or a standalone `dolfin-convert`
- `normal_modes/eigenmodes/helpers_test.py`,
  `normal_modes/eigenmodes/eigenproblems_test.py`, and
  `normal_modes/eigenmodes/eigensolvers_test.py` now pass under Python 3 and
  are included in both the local verifier and GitHub Actions core-suite gate
- `normal_modes/deprecated/normal_modes_deprecated_test.py` passes under the
  legacy-DOLFIN Python 3 core gate, but it is not yet in the pixi/M3 gate
- `normal_modes/eigenmodes/eigensolvers.py` no longer treats `num=None` as an
  operand to `min(...)`, and the unconditional SLEPc debug dumps were removed
- `normal_modes/deprecated/normal_modes_deprecated.py` materializes
  `filter(...)` results before using `len(...)` or NumPy indexing
- latest full local run of `dev/bin/verify-python3-core-suite` succeeded:
  `397 passed, 8 skipped, 1 xfailed`
- that full run predates the latest normal-modes gate expansion; the
  normal-modes subset was verified separately
- the historical `GCR` demag solver is intentionally left unported on the
  Python 3 path
- `src/finmag/energies/demag/gcr_demag.py` remains only as a placeholder and
  is not registered in `KNOWN_SOLVERS`
- the weak-tolerance demag linearity area now has both:
  - a positive regression test showing that loose Krylov tolerances are
    detectably non-linear, and
  - the original historical `xfail`, kept intentionally so the tolerance
    expectations can be reviewed by a human later
- the current green baseline includes:
  - `sim_helpers_test.py::test_get_submesh`
  - `sim_test.py::TestSimulation::test_mark_regions`
  - `sim_test.py::TestSimulation::test_pbc2d_m_init`
  - the rewritten `tests/test_dmi_terms.py`
  - `energies/dmi_test.py::test_dmi_pbc2d`
  - `energies/test_energies_in_regions.py::test_energies_in_touching_subdomains`
  - `sim_test.py::TestSimulation::test_sim_sllg`
  - `sim_test.py::TestSimulation::test_sim_sllg_time`
  - `tests/zhangli/zhang_li_test.py`
  - `util/meshes_test.py::test_regular_polygon`
  - `util/meshes_test.py::test_regular_polygon_extruded`

## Import Frontier Reached So Far

After the recent Python 3 fixes, `import finmag` now gets through:

- explicit relative package imports
- configuration/logging startup
- optional plotting/version helper imports
- `Field` import with optional `dolfinh5tools`
- energies package import
- native module build and load
- scheduler and PBC helper imports
- example package imports
- startup version reporting

Current verified result in the Python 3 DOLFIN image:

- `import finmag` succeeds
- `print(finmag)` reports the imported module object from `src/finmag/__init__.py`

Current verified runtime result in the same image:

- `sim = finmag.example.barmini(...)` succeeds
- `sim.run_until(1e-12)` succeeds
- the printed smoke result is:
  - `constructed Simulation`
  - `advanced 1e-12`

## Important Python 3 Compatibility Decisions

To reach the import milestone without broadening scope unnecessarily, some
subsystems are now optional at import time:

- `SLLG` is imported lazily and only required when the `sllg` kernel is selected
- the SciPy integrator backend is optional and raises only if explicitly requested
- normal-mode support is optional and raises only if explicitly requested

This keeps the base package importable while preserving a clear failure mode for
unfinished or missing optional dependencies.

## SLLG Status

The `sllg` kernel is no longer just an optional-import placeholder on the
Python 3 transition path.

- `src/finmag/physics/llb/sllg.py` now imports under Python 3
- the native stochastic integrator binding accepts the Python 3 seed path
  again after coercing the seed to a plain Python `int`
- `Simulation(..., kernel="sllg")` works in the transition image
- `src/finmag/tests/zhangli/zhang_li_test.py` now collects and passes under
  Python 3
- `src/finmag/physics/llb/sllg_test.py` now contains a collected pytest smoke
  test for the native `RandomMT19937.gaussian_random_np` binding
- `src/finmag/physics/llb/llb_test.py` now passes under Python 3 without
  expected failures after restoring LLB interaction setup and regional
  save-data coverage
- `src/finmag/tests/test_solid_angle.py` and
  `src/finmag/tests/test_solid_angle_invariance.py` now pass under Python 3
- `src/finmag/tests/energy_density/test_energy_density.py` now passes under
  Python 3 with the external `nsim` comparison skipped when unavailable
- `src/finmag/tests/test_skyrmions.py` passes under Python 3
- `src/finmag/tests/test_1d_domain_wall_profile_uniaxial_anisotropy.py`
  passes under Python 3 after print cleanup
- `src/finmag/tests/cython/test_cython.py` now runs under Python 3 using
  Debian's `cython3` executable
- `src/finmag/physics/tests/neb/neb_test.py` passes under Python 3 without
  code changes
- `src/finmag/tests/zhangli/stt_nonlocal_test.py` passes under Python 3 after
  print cleanup and `llg_stt.py` integer-division fixes
- `src/finmag/util/oommf/test_mesh.py` passes under Python 3 after import and
  parse cleanup in the OOMMF utility package
- `src/finmag/util/visualization_test.py` now runs under Python 3; pure
  flight-path tests pass and GUI/rendering tests remain skipped
- stale unconditional animation/export skips in `src/finmag/sim/sim_test.py`
  and `src/finmag/util/helpers_test.py` have been replaced by explicit
  Paraview/rendering/movie dependency checks
- `src/finmag/util/helpers_test.py::test_get_hg_revision_info` now has an
  explicit historical Mercurial dependency skip instead of an unconditional
  skip; `get_hg_revision_info()` is pre-Git compatibility code and should be
  removed in a later cleanup phase, not during the current Python 3 transition
- `src/finmag/sim/sim_test.py::test_compute_eigenmode_animations` now runs
  its non-movie VTK eigenmode export path under Python 3
- `src/finmag/tests/test_sim_parallel.py` now collects under Python 3 and
  skips explicitly in serial runs with fewer than two MPI ranks
- `src/finmag/tests/comparison/test_dmdt.py` now collects under Python 3 and
  passes in the transition image now that the external `oommf` executable is
  installed
- `src/finmag/tests/nmag/exchange_1d/test_exchange_1d.py` passes under
  Python 3 against checked-in reference data
- `src/finmag/tests/nmag/anisotropy_1d/test_nmag_1d_anisotropy.py` passes
  under Python 3 against checked-in reference data
- `src/finmag/tests/nmag/spinwaves/test_spinwaves.py` passes under Python 3
  against checked-in reference data
- `src/finmag/tests/nmag/exchange_3d/test_dynamics_3D.py` passes under
  Python 3 against checked-in reference data
- `src/finmag/tests/oommf/test_anisotropy.py` and
  `src/finmag/tests/oommf/test_exchange.py` collect and pass under Python 3 in
  the transition image now that the external `oommf` executable is installed
- `src/finmag/tests/slonczewski/validation/finmag/test_finmag_validation.py`
  passes under Python 3, but is comparatively slow because it runs a 10 ns
  validation simulation
- `src/finmag/tests/slonczewski/oscillator/test_oscillator.py` now collects
  under Python 3 and skips explicitly; the expensive mesh setup import is lazy
  so collection remains cheap
- the active Python 3 core-suite gate includes the Zhang-Li file in addition
  to the already-covered `sim_test.py` SLLG cases
- the active Python 3 core-suite gate also includes `sllg_test.py` and
  `llb_test.py`
- the active Python 3 core-suite gate also includes the two solid-angle files
- the active Python 3 core-suite gate also includes `energy_density` and
  `test_skyrmions.py`
- the active Python 3 core-suite gate also includes the 1D domain-wall profile
  test
- the active Python 3 core-suite gate also includes the Cython workaround test
- the active Python 3 core-suite gate also includes the small NEB unit tests
- the active Python 3 core-suite gate also includes the Zhang-Li nonlocal STT
  test
- the active Python 3 core-suite gate also includes the pure OOMMF mesh helper
  tests
- the transition image now installs the Southampton/Fangohr OOMMF fork from
  `https://github.com/fangohr/oommf`
- OOMMF refuses to run as root, so the transition image builds and runs it as
  the `oommfbuild` user and uses that account as the default container user
- OOMMF reports `OOMMF 2.1a0` in the current image
- the OOMMF binary comparison/regression tests now run under Python 3 rather
  than skipping for a missing executable
- the OOMMF Python support path needed two Python 3 fixes:
  `src/finmag/util/oommf/ovf.py` reads mixed ASCII/binary OVF files in binary
  mode and decodes only header lines, and
  `src/finmag/util/oommf/lattice.py` avoids NumPy array truth-value checks
- targeted OOMMF verification passed for `test_dmdt.py`,
  `test_anis_oommf.py`, `test_cubic_anis_oommf.py`,
  `test_exchange_field.py`, `tests/oommf/test_anisotropy.py`, and
  `tests/oommf/test_exchange.py`
- the active Python 3 core-suite gate also includes visualization helper tests
  that do not require GUI/rendering dependencies
- the active Python 3 core-suite gate also includes the parallel simulation
  test as a tracked serial-CI skip
- the active Python 3 core-suite gate also includes the OOMMF dm/dt comparison
  as a real test in the transition image
- the active Python 3 core-suite gate also includes the Nmag 1D exchange
  reference-data comparison
- the active Python 3 core-suite gate also includes the Nmag 1D anisotropy
  reference-data comparison
- the active Python 3 core-suite gate also includes the Nmag spinwaves
  reference-data comparison
- the active Python 3 core-suite gate also includes the Nmag 3D exchange
  reference-data comparison
- the active Python 3 core-suite gate also includes the standalone OOMMF
  anisotropy/exchange comparisons as real tests in the transition image
- the active Python 3 core-suite gate also includes the Slonczewski validation
  test
- the active Python 3 core-suite gate also includes the Slonczewski oscillator
  validation as a tracked skip
- the active Python 3 core-suite gate also includes the legacy comparison
  anisotropy/demag/exchange tests; OOMMF tests run in the transition image,
  and two stale Magpar mesh comparisons are tracked
  expected failures
- the OOMMF helper stack has been ported far enough for Python 3 to run the
  external `oommf` comparison tests: coordinate `zip` iterators are
  materialised, MD5 input is encoded, OVF binary output separates bytes from
  text, `OVFStream` reads mixed ASCII/binary OVF files in binary mode, and
  `reduce` comes from `functools`
- `dev/bin/verify-python3-core-suite` completed successfully in the transition
  image with `502 passed, 14 skipped, 3 xfailed` in
  approximately 1547 seconds; this is the current broad Python 3 regression
  baseline
- targeted follow-up after that full run:
  - restored the historical weak-tolerance demag `xfail` for later manual
    review, while keeping the new explicit negative regression test
  - removed a stale deprecated normal-modes `xfail`;
    `test_plot_spatially_resolved_normal_mode_in_region` now passes after
    dropping the obsolete `use_fenicstools` keyword
  - removed the stale SciPy-sparse/Nanostrip normal-modes `xfail` in
    `eigensolvers_test.py`; a later Hermitian-specialisation fix removed the
    remaining SLEPc expected failure there as well
  - fixed `SLEPcEigensolver._solve_eigenproblem()` so configured `tol` and
    `maxit` values are actually forwarded instead of being shadowed by truthy
    default arguments
  - taught `SLEPcEigensolver` to promote Hermitian inputs from the generic
    non-Hermitian problem classes to `HEP`/`GHEP`, which removes the former
    RingGraph `N=200` SLEPc `xfail`
  - adjusted the sample SLEPc test fixture to use the stronger convergence
    budget (`tol=1e-10`, `maxit=1000`) and verified the full
    `eigensolvers_test.py` file now passes without SLEPc expected failures
  - made `test_thin_film_argument_saves_time_on_thin_film` robust against
    CI/container timing jitter by replacing the strict ordering assertion with
    a small tolerated regression margin
  - closed returned matplotlib figures in the plotting-heavy normal-mode,
    simulation, and plot-helper tests; this removed the repeated "More than
    20 figures have been opened" warnings from the core-suite run
  - removed the local `figure.autolayout=True` override in
    `plot_spatially_resolved_normal_mode()`, which removes the remaining
    matplotlib `tight_layout` compatibility warnings from the normal-mode
    profile plotting tests
  - guarded the zero-denominator path in
    `normal_modes/eigenmodes/helpers.compute_relative_error()`, which removes
    the divide-by-zero warning from the zero-eigenvalue eigensolver case
  - treated `freq ~= 0` as a zero-length animation window in the deprecated
    normal-mode animation exporter, which removes the divide-by-zero warnings
    from the non-movie eigenmode animation smoke path
  - switched the noisy relative-difference diagnostics in
    `exchange_test.py` and `tests/nmag/exchange_1d/test_exchange_1d.py` to
    masked `np.divide(..., where=...)`, which removes warnings caused only by
    exact-zero reference values
  - switched `helpers.fnormalise()` to masked division for zero vectors,
    preserving the old result without emitting the old invalid-divide warning
  - suppressed the exploratory `sqrt(A / K1)` warning inside the
    domain-wall-profile fit, while keeping the original `curve_fit` path and
    assertions unchanged
  - suppressed the SciPy sparse/Nanostrip singular-matrix warning locally in
    `eigensolvers_test.py`, because the shift-invert solver path still returns
    valid eigenpairs for that testcase
  - changed the transposed Robertson SciPy/VODE test to assert the expected
    "Excess work done" warning explicitly instead of leaking it into the suite
    warning summary
  - verified the remaining Magpar xfails fail on node-array shape mismatch
    before any field tolerance comparison; this points to reference-mesh drift
    from regenerated Netgen meshes rather than a Python 3 runtime bug
  - the existing `python3-core-suite` workflow already exercises
    `normal_modes_deprecated_test.py`, so this improvement is already covered
    by CI
  - the latest full rerun confirms the updated baseline above; the remaining
    three xfails are two Magpar mesh-drift cases and the intentionally
    preserved weak-tolerance demag historical xfail
  - the pytest core-suite warning summary is now clean; the remaining oddity is
    only the old DVODE Fortran stderr diagnostic from the pathological
    Robertson/VODE case

## Native Build Findings

The Python 3 import path now depends on a functioning native rebuild in the
snapshot image.

Important findings:

- `native/Makefile` had to become Python-version-aware instead of assuming Python 2
- the vendored Sundials wrapper needed explicit handling for `libsundials-dev 2.7.0+dfsg-2`
- the original vendored custom `nvector_serial` path was not reliable enough for SUNDIALS 2.7 at runtime
- switching the Python 3 build to the system `libsundials_nvecserial` and exposing callback-local NumPy views from raw NVector memory fixed the CVODE runtime regression
- import-time builds are more robust when they do not try to build the native unit-test binary

## Next Milestone

The next milestone is no longer package import or the first smoke step.

It is:

- start validating the agreed acceptance suite under Python 3
- most likely begin with the `barmini`-based subset

## Native Build Findings

The Python 3 DOLFIN image initially failed to build Finmag native modules
because Sundials headers were missing. Adding this package fixed that layer:

- `libsundials-dev = 2.7.0+dfsg-2`

After that, the native build proceeds and fails specifically in:

- `native/src/sundials/sundials_cvode_impl.h`

The cause is version handling in the wrapper:

- it only knows Sundials 2.4 and 2.5
- the snapshot image provides Sundials 2.7
- the wrapper therefore resolves `sundials_traits<-1>`

This in turn causes the compile-time type mismatches now observed:

- direct linear solver Jacobian callback signatures use `long` in Sundials 2.7
- `CVDlsGetLastFlag` and related APIs expect `long int *`
- Finmag currently falls back to `int` because the version trait is unresolved

Practical implication:

- the next work item is not general Python 3 cleanup
- it is a native compatibility patch for the Sundials 2.7 wrapper

## Useful Commands

Reference import check in the Python 3 image:

```bash
docker run --rm -e OMPI_MCA_plm=isolated finmag-py3-dolfin2017 \
  bash -lc 'python3 - <<\"PY\"
import dolfin
print(dolfin.__version__)
PY'
```

Current Finmag import probe:

```bash
docker run --rm -e OMPI_MCA_plm=isolated -v "$PWD:/repo:ro" finmag-py3-dolfin2017 \
  bash -lc 'rm -rf /tmp/finmag && cp -a /repo /tmp/finmag && cd /tmp/finmag && \
  PYTHONPATH=/tmp/finmag/src python3 -c "import finmag"'
```

## Test and Warning Notes

Warnings observed in the passing reference run:

- `aeon.timer` nested-measurement warnings
- Python 2 deprecation warning from use of `msg.message` in `src/finmag/drivers/sundials_integrator.py`

These are not immediate blockers, but they are useful signals when cleaning up the code.

## Current M3 Baseline

The active pixi/FEniCS-2019 verifier now completes at:

- `493 passed`
- `21 skipped`
- `5 xfailed`

using:

- `dev/bin/verify-python3-m3`
- `pixi run barmini-suite`

The newest promoted M3 tests are:

- `src/finmag/sim/sim_test.py`
- `src/finmag/physics/tests/test_equation.py`
- `src/finmag/physics/tests/test_terms.py`
- `src/finmag/energies/zeeman_test.py`
- `src/finmag/tests/demag/test_bem_computation.py`
- `src/finmag/tests/demag/test_demag_sphere.py`
- `src/finmag/tests/oommf/test_anisotropy.py`
- `src/finmag/tests/oommf/test_exchange.py`
- `src/finmag/tests/comparison/test_dmdt.py`
- `src/finmag/tests/comparison/anisotropy/test_anis_oommf.py`
- `src/finmag/tests/comparison/anisotropy/test_cubic_anis_oommf.py`
- `src/finmag/tests/comparison/exchange/test_exchange_field.py`
- `src/finmag/tests/comparison/anisotropy/test_anis_magpar.py`
- `src/finmag/tests/comparison/demag/test_demag_field.py`
- `src/finmag/tests/comparison/exchange/test_exchange_compare_magpar.py`
- `src/finmag/tests/test_sim_parallel.py`
- `src/finmag/tests/nmag/anisotropy_1d/test_nmag_1d_anisotropy.py`
- `src/finmag/tests/nmag/exchange_1d/test_exchange_1d.py`
- `src/finmag/tests/nmag/exchange_3d/test_dynamics_3D.py`
- `src/finmag/tests/nmag/spinwaves/test_spinwaves.py`
- `src/finmag/tests/slonczewski/validation/finmag/test_finmag_validation.py`
- `src/finmag/tests/slonczewski/oscillator/test_oscillator.py`
- `src/finmag/energies/demag/demag_pbc_test.py`
- `src/finmag/normal_modes/deprecated/normal_modes_deprecated_test.py`
- `src/finmag/tests/zhangli/zhang_li_test.py`
- `src/finmag/tests/zhangli/stt_nonlocal_test.py`
- `src/finmag/tests/test_skyrmions.py`
- `src/finmag/tests/test_solid_angle.py`
- `src/finmag/tests/test_solid_angle_invariance.py`
- `src/finmag/physics/tests/neb/neb_test.py`
- `src/finmag/physics/llb/sllg_test.py`
- `src/finmag/physics/llb/llb_test.py`
- `src/finmag/util/meshes_test.py`
- `src/finmag/sim/sim_helpers_test.py`
- `src/finmag/tests/test_meshes.py`
- `src/finmag/util/mesh_templates_test.py`
- `src/finmag/util/helpers_test.py`
- `src/finmag/util/length_scales_test.py`
- `src/finmag/util/pbc_test.py`
- `src/finmag/util/plot_helpers_test.py`
- `src/finmag/util/vtk_saver_test.py`
- `src/finmag/util/fileio_test.py`
- `src/finmag/util/dmi_helper_test.py`
- `src/finmag/util/test_set_function_values.py`
- `src/finmag/util/test_dmi_from_helix.py`
- `src/finmag/drivers/tests/sundials_nsteps_test.py`
- `src/finmag/drivers/tests/sundials_reinit_test.py`

Additional confidence added around the new `physics/equation.py` Python
fallback:

- direct native-vs-Python parity tests for `terms`
- direct native-vs-Python parity tests for `Equation.solve()` and pinning
- a Python-fallback `jtimes` finite-difference contract test
- checked-in native reference data in
  `src/finmag/physics/tests/equation_reference_data.json` so the fallback can
  still be checked against legacy native outputs without requiring the
  container backend
- a native-stack benchmark report showing the Python fallback is roughly
  `107x` slower on a small repeated `Equation.solve()` loop, so it remains a
  compatibility bridge rather than a performance-equivalent replacement
- `src/finmag/drivers/tests/test_scipy.py`
- `src/finmag/drivers/tests/test_integrators.py`
- `src/finmag/util/ode/tests/test_sundials_stiff_ode.py`
- `src/finmag/tests/jacobean/test_jacobean_computation.py`
- `src/finmag/tests/jacobean/test_jacobean_integration.py`
- `src/finmag/tests/jacobean/test_native_llg.py`
- `src/finmag/util/oommf/test_mesh.py`
- `src/finmag/util/visualization_test.py`
- `src/finmag/util/fft_test.py`
- `src/finmag/drivers/tests/test_relaxation.py`
- `src/finmag/drivers/tests/test_relax_two_times.py`
- `src/finmag/tests/test_time.py`
- `src/finmag/scheduler/scheduler_test.py`
- `src/finmag/tests/test_cyclic_references_in_sim.py`
- `src/finmag/tests/bugs/test_bug_ndt_file_writing.py`
- `src/finmag/tests/cython/test_cython.py`
- `src/finmag/physics/tests/test_effective_field.py`
- `src/finmag/tests/test_jacobian.py`
- `src/finmag/normal_modes/eigenmodes/helpers_test.py`
- `src/finmag/normal_modes/eigenmodes/eigenproblems_test.py`
- `src/finmag/normal_modes/eigenmodes/eigensolvers_test.py`
- `src/finmag/field_setters_test.py`
- `src/finmag/field_test.py`
- `src/finmag/energies/dmi_test.py`
- `src/finmag/energies/magnetostatic_field_test.py`
- `src/finmag/energies/zeeman_test.py`
- `src/finmag/tests/demag/test_bem_computation.py`
- `src/finmag/tests/demag/test_demag_sphere.py`
- `src/finmag/sim/magnetisation_patterns_test.py`
- `src/finmag/sim/hysteresis_test.py`
- `src/finmag/tests/oommf/test_anisotropy.py`
- `src/finmag/tests/oommf/test_exchange.py`
- `src/finmag/tests/comparison/test_dmdt.py`
- `src/finmag/tests/comparison/anisotropy/test_anis_oommf.py`
- `src/finmag/tests/comparison/anisotropy/test_cubic_anis_oommf.py`
- `src/finmag/tests/comparison/exchange/test_exchange_field.py`
- `src/finmag/tests/comparison/anisotropy/test_anis_magpar.py`
- `src/finmag/tests/comparison/demag/test_demag_field.py`
- `src/finmag/tests/comparison/exchange/test_exchange_compare_magpar.py`
- `src/finmag/tests/test_sim_parallel.py`
- `src/finmag/tests/nmag/anisotropy_1d/test_nmag_1d_anisotropy.py`
- `src/finmag/tests/nmag/exchange_1d/test_exchange_1d.py`
- `src/finmag/tests/nmag/exchange_3d/test_dynamics_3D.py`
- `src/finmag/tests/nmag/spinwaves/test_spinwaves.py`
- `src/finmag/tests/slonczewski/validation/finmag/test_finmag_validation.py`
- `src/finmag/tests/slonczewski/oscillator/test_oscillator.py`
- `src/finmag/energies/demag/demag_pbc_test.py`
- `src/finmag/normal_modes/deprecated/normal_modes_deprecated_test.py`

Current M3 boundary notes for future agents:

- full NEB tangent computations still require `finmag.native.neb`, even
  though the helper-only `src/finmag/physics/tests/neb/neb_test.py` file now
  runs in M3
- `src/finmag/energies/zeeman_test.py` now runs in M3. The sphere-vs-dipole
  test keeps the historical `140.0` A/m absolute guard; it uses
  `maxh_sphere=2.0` because investigation showed the pixi/FEniCS-2019 Netgen
  mesh at `maxh_sphere=2.5` had a slightly larger pointwise tail despite
  passing the relative field checks. Do not loosen that tolerance without
  renewed numerical review. [Codex GPT-5.4]
- `src/finmag/tests/demag/test_bem_computation.py` now runs in M3 with the
  same DOLFIN-2019 array fallback for native FK BEM construction that
  production `FKDemag` uses. The unported GCR branch is still an explicit
  skip. [Codex GPT-5.4]
- OOMMF is now installed from conda-forge in the pixi environment, so M3 runs
  the promoted OOMMF comparison files directly instead of treating them as
  external-tool skips. `src/finmag/tests/comparison/test_dmdt.py` and
  `src/finmag/tests/comparison/exchange/test_exchange_field.py` use
  `get_local()` for DOLFIN-2019 PETSc vectors. [Codex GPT-5.4]
- Do not try to install `nsim` on the pixi track. Where Nmag comparisons are
  useful, prefer checked-in reference data, as in
  `src/finmag/tests/comparison/exchange/test_exchange_field.py`. [Codex GPT-5.4]
- The latest M3 expansion promoted checked Nmag reference-data tests and
  Slonczewski validation without installing `nsim`; the focused subset passed
  as `12 passed, 2 skipped, 2 xfailed`, and the full M3 gate exited
  successfully. [Codex gpt-5.5 high]
- `src/finmag/normal_modes/deprecated/normal_modes_deprecated_test.py` now
  runs in M3. Its previous Kittel sphere failure was caused by ARPACK's
  implicit random start vector on SciPy 1.17.1; the deprecated generalized
  solver now supplies a deterministic alternating `v0` when callers do not
  pass one. [Codex gpt-5.5 high]
- `src/finmag/energies/demag/demag_pbc_test.py` is now in M3 as a collected
  skip: focused pixi runs report `2 skipped` with exit code 0 when
  `finmag.native.treecode_bem` is missing. [Codex gpt-5.5 high]
- The pixi/M3 `barmini-suite` now lists all `101` files from
  `dev/bin/verify-python3-core-suite`; remaining differences are collected
  skips/xfails, not missing files. [Codex gpt-5.5 high]

The old `instant` dependency is no longer required for the solid-angle
reference tests on this path:

- `finmag.util.solid_angle_magpar.return_csa_magpar()` now falls back to a
  NumPy implementation when `instant` is unavailable

The pixi native build now includes `llb.so` again:

- `src/finmag/physics/llb/sllg_test.py` is active in M3
- `src/finmag/tests/zhangli/zhang_li_test.py::test_zhangli_sllg` now runs on
  the pixi/FEniCS-2019 path

Recent helper-path compatibility work to know about:

- `src/finmag/util/meshes.py` now uses the DOLFIN 2019 `MeshEditor.open()`
  cell-type string API and forces Gmsh output to `msh2` for the old
  `dolfin-convert` bridge
- `src/finmag/util/meshes.py` also falls back to the Netgen Python API when
  conda-forge does not ship the historical CLI binary, and a launched Netgen
  probe timeout is treated as a real failure rather than a skip
- `src/finmag/sim/sim_savers.py` and `src/finmag/sim/sim_helpers.py` now use
  `get_local()` on PETSc-backed vectors in the exercised M3 save/helper paths
- `src/finmag/sim/sim.py` now uses a Python 3.11-safe
  `inspect.getfullargspec` fallback in scheduling
- `src/finmag/util/helpers.py` and `src/finmag/util/helpers_test.py` now use
  DOLFIN 2019-compatible PETSc-vector access, callable helper-expression
  setup, and `MeshFunction`-based cell marking on the exercised path
- `src/finmag/util/length_scales_test.py` now uses a pytest-9-compatible
  `setup_method()` hook, and `src/finmag/util/vtk_saver_test.py` now
  initializes PETSc vectors with `set_local()`
- `src/finmag/drivers/sundials_integrator.py` now treats the expected
  `mxstep` stop condition in `advance_steps()` via actual step-count deltas,
  which is more stable across SUNDIALS wrapper versions than matching only the
  older `CV_TOO_MUCH_WORK` exception text
- `src/finmag/util/ode/tests/test_sundials_stiff_ode.py` now uses
  `solve_ivp(method="BDF")` as the SciPy stiff-solver reference because the
  legacy real-valued `ode(..., "vode")` path no longer behaves like SciPy
  0.19.1 on the Robertson problem under SciPy 1.17.1
- the two transposed Robertson cases are now explicit `xfail(strict=True)`
  tests of known bad Jacobian orientation behaviour, not passing negative
  tests
- `src/finmag/tests/jacobean/test_native_llg.py` now uses `get_local()` for
  the PETSc-backed alpha vector on the exercised M3 path

Current additional M3 boundary note:

- `treecode_bem` is still missing from the pixi native build. The PBC demag
  regression file is represented in M3 as a skip until that extension exists.
  Do not add a NumPy fallback as the next step; the agreed path is to build
  and activate `finmag.native.treecode_bem` on pixi, then unskip the PBC demag
  checks once the native extension is validated. [Codex gpt-5.5 high]

## M4 / DOLFINx Probe

The first DOLFINx work is deliberately isolated from the default pixi
environment:

- `pixi.toml` has a `dolfinx` feature with `python = "3.12.*"` and
  `fenics-dolfinx`
- the `dolfinx` environment uses `no-default-feature = true`, so it does not
  inherit the FEniCS-2019/Python-3.11 M2/M3 stack
- `dev/bin/verify-dolfinx-m4` runs the current import and tiny
  mesh/function smoke probes plus `dolfinx-pytest`
- `dev/dolfinx/test_dolfinx_smoke.py` keeps the same tiny DOLFINx
  mesh/function workflow under pytest
- `.github/workflows/dolfinx-m4.yml` runs the same isolated probe in CI
- local verification produced `dolfinx 0.10.0 rank 0` and
  `dofs 9 sum 13.5`; the pytest probe passes as `2 passed`

Do not treat this as a Finmag port yet; it only establishes a separate M4
dependency/probe lane. [Codex gpt-5.5 high]

The next small M4 prototype step is also isolated under `dev/dolfinx`:

- `dev/dolfinx/prototype.py` creates vector-valued DOLFINx magnetisation
  fields and assembles exchange, constant-field Zeeman, and constant-axis
  uniaxial anisotropy energy
- `dev/dolfinx/test_prototype.py` checks constant-vector interpolation,
  component-count validation, analytical unit-square Zeeman energy, and
  `unit_length` scaling. It also covers analytical uniaxial anisotropy cases,
  axis normalisation, zero-axis validation, and exchange energy for constant
  and linear fields. Exchange scaling is documented as
  `unit_length ** (dim - 2)`. [Codex gpt-5.5 high]
- the prototype also has an explicit normalized LLG step for a constant
  effective field. Treat it as the first checked time-integration workflow, not
  as a production solver. [Codex gpt-5.5 high]
- `dev/dolfinx/relaxation_example.py` is the first checked end-to-end M4
  example with JSON output. The `dolfinx-example` task writes
  `/tmp/finmag-dolfinx-relaxation-summary.json` by default. [Codex gpt-5.5
  high]
- the example JSON has `schema_version = 1` and a hand-written validator. Keep
  this contract explicit if the output grows. [Codex gpt-5.5 high]
- `dev/dolfinx/README.md` documents supported scope and explicit non-scope for
  M4. Do not imply demag, restart/data I/O, production time integration, or
  legacy `finmag.Simulation` API compatibility until those are actually built.
  [Codex gpt-5.5 high]
- the relaxation example has small dataclasses for parameters and JSON results.
  Treat them as an API sketch, not as a stable public API. [Codex gpt-5.5 high]
- the parameter dataclass validates constants and vector sizes before DOLFINx
  form assembly. Keep invalid M4 inputs explicit. [Codex gpt-5.5 high]
- the relaxation example CLI path has pytest coverage for stdout JSON and the
  requested output file. [Codex gpt-5.5 high]
- invalid CLI controls are covered by pytest; keep wrapper-facing validation
  aligned with function-level validation. [Codex gpt-5.5 high]
- summary validation checks the nested `parameters` block as part of the M4
  JSON contract. [Codex gpt-5.5 high]
- repeated relaxation-example runs with identical controls are tested for
  deterministic JSON summaries. [Codex gpt-5.5 high]
- vector-parameter validation rejects wrong sizes and non-numeric values before
  DOLFINx form assembly. [Codex gpt-5.5 high]
- `dolfinx-pytest` now runs all tests below `dev/dolfinx`; local verification
  is `25 passed`

The reduced M4 completion criteria are represented by the current `dev/dolfinx`
lane: one end-to-end example, documented supported scope, and explicit
unsupported subsystems. Treat this as a stable base for M5, not as a production
port or legacy API replacement. [Codex gpt-5.5 high]

Keep `dev/dolfinx` as an exploration lane. It should reduce DOLFINx risk and
provide executable witnesses, but it must not become a clean-room replacement
for Finmag by accident. [Codex gpt-5.5 high]

For the actual FEniCS-2019.1-to-DOLFINx port, preserve the existing Finmag
software design wherever practical: public API, `Simulation`, energy modules,
drivers, field/data abstractions, restart/output conventions, and tests are
the baseline. Departures need concrete justification from DOLFINx semantics,
Python/MPI/runtime changes, packaging constraints, or other software-stack
changes. [Codex gpt-5.5 high]

Use the green FEniCS-2019/Python-3 tests as the behavioural contract. If the
DOLFINx port requires changed behaviour, update tests explicitly and document
why; do not let `PrototypeSimulation` become the production API unless that is
deliberately mapped back to the existing Finmag design. [Codex gpt-5.5 high]

`dev/dolfinx/porting_map.md` records the legacy surfaces to preserve, current
prototype evidence, and promotion criteria. Consult it before moving any
`dev/dolfinx` implementation into `src/finmag`. [Codex gpt-5.5 high]

The FK demag/BEM baseline in M3/pixi is compiled `finmag.native.llg`
(`compute_bem_fk` or `compute_bem_fk_from_arrays`). Do not treat the NumPy
Magpar helper as the production FK BEM implementation. PBC/treecode demag is a
separate native-code gap because it depends on `finmag.native.treecode_bem`.
[Codex gpt-5.5 high]

`pixi run native-fk-bem-smoke` is the direct smoke for that compiled FK BEM
baseline. Keep it in the M3 verifier before `barmini-suite` so native BEM
failures are isolated from broader demag regression failures.
[Codex gpt-5.5 high]

Use `dev/bin/verify-python3-native-fk-bem-smoke` for a single-command
diagnostic run of the same witness. [Codex gpt-5.5 high]

M5 has started with `dev/dolfinx/simulation.py`, a reduced
`PrototypeSimulation` wrapper over the checked M4 helpers. It is an API sketch
for the core path, not compatibility with legacy `finmag.Simulation`. [Codex
gpt-5.5 high]

`dev/dolfinx/field_adapter.py` is a narrow DOLFINx-backed compatibility probe
for initial legacy `finmag.field.Field` behaviours: scalar/vector inspection,
constants, callables, nodal values, normalisation, and volume averages. [Codex
gpt-5.5 high]

`PrototypeSimulation.state_summary()` reports mesh label, average
magnetisation, parameters, and energy terms without advancing the simulation.
[Codex gpt-5.5 high]

`PrototypeSimulation.relaxation_summary()` provides a JSON-compatible basic
output path for the first M5 core sketch. [Codex gpt-5.5 high]

The simulation wrapper summary conforms to the existing relaxation summary
validator, including `dolfinx_version` and schema version. [Codex gpt-5.5 high]

`PrototypeSimulation.write_relaxation_summary()` writes the validated summary
to JSON as the first direct file-output path for the M5 core sketch. [Codex
gpt-5.5 high]

`PrototypeSimulation.relaxation_trace()` and `write_relaxation_trace()` provide
a JSON-compatible per-step trace with absolute prototype simulation time,
average magnetisation, and energy terms. This is a reduced data-I/O contract,
not legacy NDT, VTK/XDMF, or scheduler support. [Codex gpt-5.5 high]

`PrototypeSimulation.time` and `run_until(...)` are a narrow compatibility-
shaped probe for the legacy `Simulation.run_until` idea. They use the explicit
M5 stepper and are not a production DOLFINx driver. [Codex gpt-5.5 high]

`PrototypeSimulation.write_restart_state()` and `read_restart_state()` provide
a narrow JSON restart-state round trip for the reduced unit-square wrapper.
This captures prototype time, nodal magnetisation values, and parameters only;
it is not a general Finmag restart format. [Codex gpt-5.5 high]

`dev/dolfinx/restart_example.py` is a runnable M5 restart-state witness, and
`dev/bin/verify-dolfinx-m4` now runs it through the `dolfinx-restart-example`
pixi task. [Codex gpt-5.5 high]

Zero exchange and anisotropy constants short-circuit to `0.0` before UFL form
assembly to avoid degenerate zero-form domain errors. Local DOLFINx verification
is `51 passed`. [Codex gpt-5.5 high]

Keep this out of `src/finmag` until the reduced DOLFINx API shape is clearer.
[Codex gpt-5.5 high]
