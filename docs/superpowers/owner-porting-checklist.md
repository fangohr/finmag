# Finmag Porting Decisions — Owner Meeting Checklist

**Meeting date:** 2026-07-23

**Repository owner:** Hans Fangohr

**Others present:** Sam Holt

For each row, tick exactly one:

- **Now** — needed for the first usable serial simulator release (SR1)
- **Later** — required eventually, but can follow SR1
- **Not now** — not required for SR1; retain it in the backlog and reconsider
  later. This does not mean “never.”

The status column does not use “partial” or “mostly.” It says exactly what is
working and what is missing or broken. “Working and tested” applies only to the
scope named in that row; it does not imply that the whole subsystem is complete.

Write a short note where the scope is unclear. A permanent omission requires a
separate explicit owner decision in `acceptance-register.md`; a **Not now** tick
does not authorise deletion or removal from the parity backlog.

## 1. Core simulator and everyday physics

| Functionality | Exact current boundary | Now | Later | Not now | Notes |
|---|---|:---:|:---:|:---:|---|
| Pixi install, native build and core `import finmag` | **Working and tested.** The editable install, native build, provenance check and core import pass CI. Some optional top-level names still trigger raw `dolfin` imports; see their separate row. | [x] | [ ] | [ ] | |
| Legacy `Simulation` constructor, state and common methods | **Working and tested:** construction, `set_m`, state access, interactions, energy, `run_until` and `relax`. **Missing/different:** complete public-surface audit, callable pin masks and several convenience methods. | [x] | [ ] | [ ] | |
| Everyday top-level compatibility names | **Working:** `Simulation`, `sim_with`, `Field`, energies, versions and configuration. **Broken:** `set_logging_level` and `example` can fail through raw legacy-`dolfin` imports. | [x] | [ ] | [ ] | Include `finmag.example` and barmini in SR1. |
| Normal-mode top-level compatibility names | **Broken:** `NormalModeSimulation` and `normal_mode_simulation` can fail through raw legacy-`dolfin` imports. The implementation itself is deferred below. | [ ] | [x] | [ ] | |
| `sim_with` for common interactions and FK demag | **Working and tested:** Exchange, Zeeman, uniaxial anisotropy, DMI and FK demag arguments. | [x] | [ ] | [ ] | All examples which can me run and all tests needed now. Port all tests and skip/xfail ones we know should not work for now |
| `sim_with` for MacroGeometry demag | **Not working:** direct dense-FK MacroGeometry works, but the factory rejects its nx/ny/spacing arguments. | [x] | [ ] | [ ] | MacroGeometry is needed now. |
| `sim_with` Treecode selector | **Not working:** direct Treecode construction works, but the factory rejects a non-FK solver choice. | [ ] | [x] | [ ] | Reconsider later; it may not be needed. |
| Common mesh generators | **Working and tested:** box, sphere, cylinder, ellipsoid, disks, ring, cone, common CSG combinations and the example `.geo` subset. | [x] | [ ] | [ ] | |
| Common `Field` operations | **Working and tested:** constants, callables, arrays, set/get, ordering, averages, normalisation, point probing, spherical conversion and VTK/XDMF write. | [x] | [ ] | [ ] | |
| Remaining `Field` operations | **Not implemented:** `from_generic_vector`, `cross`, `dot`, scalar coercion, HDF5 and plotting. | [x] | [ ] | [ ] | |
| Deterministic serial LLG dynamics | **Working and tested:** composed effective fields, deterministic trajectories, indexed pins, `run_until` and relaxation. | [x] | [ ] | [ ] | General MPI stepping is a separate Later row. |
| Callable pin masks | **Not implemented:** indexed pins work, but coordinate-based Python callables are not accepted. | [x] | [ ] | [ ] | |
| Exchange with supported box assembly | **Working and tested:** constant and spatially varying A. | [x] | [ ] | [ ] | The obsolete matrix/project/direct algorithms are separate Not-now rows. |
| Static applied fields | **Working and tested:** Zeeman and DipolarField values, fields and energies. | [x] | [ ] | [ ] | |
| Time-dependent applied fields | **Working and tested:** TimeZeeman, DiscreteTimeZeeman, TimeZeemanPython scalar mode and OscillatingZeeman. **Missing/different:** vector-valued TimeZeemanPython and the stale-energy decision below. | [x] | [ ] | [ ] | |
| Uniaxial anisotropy | **Working and tested:** constant/varying coefficients and axes. | [x] | [ ] | [ ] | |
| Cubic anisotropy | **Working and tested:** constant axes and constant/varying K coefficients through both field paths. **Not implemented:** spatially varying cubic axes. | [x] | [ ] | [ ] | |
| DMI documented variants | **Working and tested:** bulk/auto, interfacial and documented 1D/2D forms, including varying D. | [x] | [ ] | [ ] | |
| DMI `D2D` variant | **Not implemented:** raises a D2D-specific error. | [ ] | [x] | [ ] | |
| FK demagnetisation | **Working and tested:** compiled FK field and energy against frozen-oracle references. | [x] | [ ] | [ ] | |
| MacroGeometry demagnetisation | **Working and tested when constructed directly:** dense-FK analytic/cross-method checks. **Missing:** `sim_with` wiring is the separate Now row above. | [x] | [ ] | [ ] | |
| Treecode demagnetisation | **Working and tested when constructed directly. Missing:** `sim_with` selector is the separate Later row above. | [ ] | [x] | [ ] | Reconsider whether this is required. |
| Spatially varying materials and regions | **Working and tested:** varying Ms, A, K, D, axis and alpha; region marking, energies and averages. **Not implemented:** region-restricted field output. | [x] | [ ] | [ ] | |
| Slonczewski STT | **Working and tested when used alone:** oracle/analytic and both-integrator checks. **Different:** simultaneous behavior with Zhang-Li is an owner choice below. | [x] | [ ] | [ ] | simultaneous not needed |
| Zhang-Li STT | **Working and tested when used alone:** oracle/analytic, domain-wall and both-integrator checks. **Different:** simultaneous behavior with Slonczewski is an owner choice below. | [x] | [ ] | [ ] | simultaneous not needed |
| Relaxation | **Working and tested.** | [x] | [ ] | [ ] | |
| Hysteresis | **Runs, but is physically wrong after the first stage:** the driver changes the applied field but does not perform a fresh relaxation, so later points are not independently equilibrated. | [x] | [ ] | [ ] | Fix and document; see Section 4. |
| Scheduler and NDT output | **Working and tested:** scheduled actions plus NDT write/read. | [x] | [ ] | [ ] | |
| VTK/XDMF field output | **Working and tested for writing. Not implemented:** general readback. | [x] | [ ] | [ ] | |
| `.npy` field snapshots | **Working and tested:** coordinate/value table format. **Different:** original files used raw backend dofs; owner choice below. | [ ] | [x] | [ ] | |
| Restart on SciPy | **Working and tested for restoring m/time. Missing evidence:** restarted trajectory is not compared with an uninterrupted control. Old raw-dof v1 files are rejected. | [x] | [ ] | [ ] | |
| Restart on Sundials | **Broken:** restart calls reset and fails on a SciPy-only `.ode` assumption. Saved driver metadata is also hard-coded to `scipy`. | [x] | [ ] | [ ] | Sundials should be default |
| SciPy integrator lifecycle | **Working and tested:** construct, advance, tolerance update, reset/reinit and SciPy restart. | [x] | [ ] | [ ] | |
| Sundials/CVODE integration | **Working and tested:** construction, trajectory advance and SciPy cross-check. **Broken:** reset and restart. | [x] | [ ] | [ ] | |
| Fast converted examples | **Working and tested:** 14 pass; 3 full-resolution cases are deliberately skipped. | [x] | [ ] | [ ] | |
| Full-resolution examples and muMAG standard problem 4 | **Not yet accepted:** no recorded all-green full lane. The current std_prob4 0.10–0.18 ns switching window is qualitative, not proof of unchanged behavior. | [x] | [ ] | [ ] | We need to make sure all the tests are ported as close to they previously were in master/pixi this will highlight any changes in behaviour or interfaces. Including standard problems |

## 2. Specialist physics and analysis

| Functionality | Exact current boundary | Now | Later | Not now | Notes |
|---|---|:---:|:---:|:---:|---|
| Thermal stochastic LLG (SLLG) | **Not implemented on DOLFINx.** | [ ] | [ ] | [x] | |
| Landau-Lifshitz-Bloch (LLB) and material laws | **Not implemented on DOLFINx.** | [ ] | [ ] | [x] | |
| Linear normal modes and eigensolvers | **Not implemented on DOLFINx;** top-level access can reach raw `dolfin` imports. | [ ] | [x] | [ ] | |
| Ringdown analysis, FFT/PSD and dispersion | **Not implemented on DOLFINx.** | [ ] | [x] | [ ] | |
| Legacy NEB variants | **Not implemented on DOLFINx.** All master NEB variants still need an algorithm/API inventory. | [ ] | [x] | [ ] | |
| GNEB as a named workflow | **No public GNEB API/workflow found on master.** Possible overlap with spherical/modified legacy NEB has not been classified. | [ ] | [x] | [ ] | To be clear one GNEB method would be good but we will do this in a later PR |
| Nonlocal `LLG_STT` spin-accumulation model | **Not implemented on DOLFINx.** This is distinct from the two working local STT terms. | [ ] | [ ] | [x] | |
| General MPI-parallel time stepping | **Not implemented.** Only ownership-sensitive Field/energy probes currently run on multiple ranks. | [ ] | [x] | [ ] | |
| `Simulation(pbc=)` function-space periodic boundaries | **Not implemented.** Treecode periodic MacroGeometry is a different, working capability. | [x] | [ ] | [ ] | |
| `Demag2D` | **Not implemented on DOLFINx.** | [ ] | [ ] | [x] | |
| GCR demag | **Not implemented.** The surviving legacy Python formulation appears incomplete and its intended scientific contract needs review. | [ ] | [ ] | [x] | |
| `FixedEnergyDW` | **Not implemented.** No master tests were found and a legacy note calls the class broken. | [ ] | [ ] | [x] | |

## 3. I/O, utilities and external tools

| Functionality | Exact current boundary | Now | Later | Not now | Notes |
|---|---|:---:|:---:|:---:|---|
| HDF5 readback | **Not implemented on DOLFINx.** | [x] | [ ] | [ ] | |
| Plotting and visualisation helpers | **Not implemented on the supported DOLFINx path.** VTK/XDMF data writing is covered separately above. | [x] | [ ] | [ ] | |
| Region/submesh field output | **Not implemented.** Region energies and averages do work. | [x] | [ ] | [ ] | |
| Magnetisation initialisers (`initialise_*`) | **Not implemented on DOLFINx.** | [x] | [ ] | [ ] | |
| `length_scales`, `mesh_info`, profiling, logging/instance/shutdown helpers | **Not implemented or still tied to legacy imports.** These names require individual inventory before any omission decision. | [x] | [ ] | [ ] | |
| `Simulation.probe_field*`, LLG `M`/`M_average` and remaining Field arithmetic | **Not implemented on the supported DOLFINx surface.** | [x] | [ ] | [ ] | |
| Specialist mesh generators | **Not implemented:** layered/internal-shell/polygon/line/embed3d and other long-tail generators. Common generators are covered above. | [ ] | [ ] | [x] | |
| Netgen binary backend | **Not implemented.** Gmsh and a limited textual `.geo` parser cover the current common subset. | [x] | [ ] | [ ] | |
| `nmesh_to_dolfin` conversion | **Not implemented;** its output is legacy dolfin-XML, which DOLFINx cannot read. | [ ] | [ ] | [x] | |
| OOMMF comparison workflow | **Live harness not ported.** Some ported tests use checked-in cross-code/reference data. | [x] | [ ] | [ ] | |
| Nmag/`nsim` comparison workflow | **Live harness not ported;** `nsim` is absent. Some checked-in Nmag reference data is exercised. | [x] | [ ] | [ ] | |
| Magpar comparison workflow | **Live harness not ported.** Saved data exists, but some historical comparisons have mesh/node-order drift. | [x] | [ ] | [ ] | |
| Legacy matrix energy assembly | **Not implemented;** box assembly is the supported algorithm. | [ ] | [ ] | [x] | |
| Legacy project energy method | **Not implemented;** box assembly is the supported algorithm. | [ ] | [ ] | [x] | |
| Legacy direct energy method | **Not implemented;** box assembly is the supported algorithm. | [ ] | [ ] | [x] | |
| Varying cubic-anisotropy axes | **Not implemented;** constant axes work. | [x] | [ ] | [ ] | |
| Legacy string `Expression` compatibility | **Not implemented as strings.** Old FEniCS accepted formula text that it compiled at runtime; DOLFINx users can supply ordinary vectorised Python callables instead. | [ ] | [ ] | [x] | Do not add string compatibility; document callable examples and clear errors. |
| Compiled `Equation`/`terms` performance backend | **Not implemented.** A slower Python fallback exists; DOLFINx LLG uses its own RHS. | [ ] | [x] | [ ] | How important is this? |
| Paraview/mencoder movie wrapper | **Not implemented.** VTK/XDMF simulation data can be written for external tools. | [ ] | [x] | [ ] | Can we use pyvista instead?|
| `batch_task` sweep tooling | **Not implemented and untested on master.** | [ ] | [ ] | [x] | |
| Mercurial revision helper | **Not ported; obsolete in the Git repository.** | [ ] | [ ] | [x] | |

## 4. Behavior decisions

Decisions discussed with Hans Fangohr and relayed on 2026-07-23:

| Topic | Decision and implementation contract |
|---|---|
| Stale `DiscreteTimeZeeman.compute_energy()` | **Fix and document.** The field and reported energy must both update; retain a regression explaining the legacy bug. |
| Hysteresis no-re-relax defect | **Fix and clearly document.** Every field stage must independently relax; retain a regression explaining why legacy output differed. |
| Default integrator | **Sundials.** Restore the legacy public default only after reset, reinitialisation, scheduling and restart pass the same lifecycle checks as SciPy. |
| Simultaneous Slonczewski and Zhang-Li STT | **Reject explicitly.** Each mode works separately; attempting to configure both must raise a clear error rather than silently applying precedence or last-call-wins. |
| Invalid pin index | **Raise `ValueError`.** Do not preserve legacy behavior that logged and silently retained stale pins. |
| Restart format | **Version 2 only.** There are no legacy restart files to support; reject v1 clearly rather than attempting unsafe raw-dof remapping. |
| Field snapshot format | **No raw-format compatibility.** Retain the coordinate/value representation; do not reproduce backend-dof arrays. |
| Spatially varying cubic K2 indexing | **Use correct physics.** Do not reproduce the legacy native indexing bug. |
| Legacy string `Expression` inputs | **No backward compatibility.** Use documented vectorised Python callables and clear errors for string inputs. |

## Meeting outcome

**SR1 must contain:**

The rows ticked **Now** are the canonical SR1 scope. In particular: deterministic
serial dynamics; callable pins; MacroGeometry-through-FK; PBC probe and serial
implementation; selected Field/I/O/convenience rows; checked reference-data
comparisons; and the legacy-test translation inventory.

**Definitely later:**

The rows ticked **Later** are the canonical later backlog, including normal
modes, ringdown/FFT, NEB/GNEB work, MPI stepping, Treecode factory exposure,
and movie tooling.

**Not required in the short term; retain for later review:**

The rows ticked **Not now** are retained, not dropped: thermal SLLG/LLB,
nonlocal STT, Demag2D/GCR/FixedEnergyDW, specialist mesh/tooling paths and
legacy energy assembly. Section 4’s per-behaviour decisions are canonical for
the meeting record; the acceptance register is canonical for approved
corrections and compatibility changes.

**P0.2 and next engineering slice:**

P0.2 baseline evidence was recorded on 2026-07-23. Next is SR1-L1: Sundials
reset/restart lifecycle (P1.1/P1.2).

[Codex GPT-5]
