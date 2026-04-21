# Finmag Project Notes

## High-Level Summary

Finmag is a finite-element micromagnetics codebase built on top of FEniCS/dolfin. The public entry point is the `Simulation` class, which orchestrates:

- mesh and function-space setup,
- magnetisation and material fields,
- effective-field construction from interactions,
- time integration,
- scheduled output and data saving.

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
- `normal_modes/deprecated/normal_modes_deprecated_test.py` also passes under
  Python 3 and is included in both gates
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
- a small number of historical failures remain marked `xfail`, including known
  weak-tolerance demag linearity checks
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
- `src/finmag/physics/llb/llb_test.py` now collects under Python 3 and exits
  cleanly with its two existing expected xfails
- `src/finmag/tests/test_solid_angle.py` and
  `src/finmag/tests/test_solid_angle_invariance.py` now pass under Python 3
- the active Python 3 core-suite gate includes the Zhang-Li file in addition
  to the already-covered `sim_test.py` SLLG cases
- the active Python 3 core-suite gate also includes `sllg_test.py` and
  `llb_test.py`
- the active Python 3 core-suite gate also includes the two solid-angle files

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
