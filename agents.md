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

## Test and Warning Notes

Warnings observed in the passing reference run:

- `aeon.timer` nested-measurement warnings
- Python 2 deprecation warning from use of `msg.message` in `src/finmag/drivers/sundials_integrator.py`

These are not immediate blockers, but they are useful signals when cleaning up the code.
