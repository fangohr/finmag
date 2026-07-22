# Finmag DOLFINx Port — Acceptance Register (Task 29)

Every documented deviation from legacy behavior, and every "accept-drop"
candidate from the master-parity audit, gathered for a single user sign-off.
Each item is test-pinned and documented in `transition-notes.org` /
`dev/dolfinx/porting_map.md`. Disposition column filled in after user review.

## A. Behavioral deviations (accept, or request a fix)

| # | Deviation | Why it exists | Recommendation | Disposition |
|---|---|---|---|---|
| D1 | `DMI(dmi_type='D2D')` raises by name | `D2D` was never in the legacy class's documented `dmi_type` options | Accept (deferral) | _pending_ |
| D2 | Legacy K2 native-field typo NOT reproduced — port ships the correct per-node field | `energy.cc` has a fixed-index bug; dormant for constant K2, divergence-pinned for varying K2 | Accept (correct physics) | _pending_ |
| D3 | `DiscreteTimeZeeman.compute_energy()` stale after interval update — legacy bug PRESERVED | Bug-compatible with legacy; `compute_field()` stays correct | **Your call: keep vs fix** | _pending_ |
| D4 | `hysteresis()`/`hysteresis_loop()` don't independently re-relax after first stage — legacy defect PRESERVED | Legacy was equally broken (oracle-confirmed); `relax()` itself works | **Your call: keep vs fix** | _pending_ |
| D5 | `TimeZeemanPython` vector-valued `time_fun` raises by name | Narrow branch, untested even on legacy | Accept (deferral) | _pending_ |
| D6 | Constant anisotropy axis normalised; spatially varying axis used as-given | Restores the legacy cosine contract for constant axes; varying-axis matches legacy | Accept | _pending_ |
| D7 | `Field.from_function` interpolates on space mismatch (legacy raw-copied/errored) | Tested superset of legacy behavior | Accept | _pending_ |
| D8 | `Simulation` default integrator = scipy (legacy default = sundials) | Factory `llg_integrator` default IS sundials (legacy-faithful); only `Simulation`'s own default differs, to avoid coupling every gate to the native build | **Your call: keep scipy vs restore sundials** | _pending_ |
| D9 | STT spatially-varying J/Ms/p use coordinate ordering throughout | Legacy fed raw-dof arrays alongside coordinate-ordered m — a latent legacy inconsistency; port is self-consistent | Accept | _pending_ |

## B. Accept-drop candidates (from the master-parity audit)

| # | Capability | Why droppable | Recommendation | Disposition |
|---|---|---|---|---|
| M2 | GCR demag solver | Legacy formulation judged unsound; FK is the validated path | Drop | _pending_ |
| M3 | Compiled `Equation`/`terms` backend | Python fallback works, ~107× slower; DOLFINx LLG uses its own RHS anyway | Drop (or perf-port later) | _pending_ |
| M7 | Magpar mesh-drift comparison xfails | Fixture node-ordering drift; physics fine | Drop (as xfail) | _pending_ |
| M8 | Paraview/mencoder movie export | External renderer; VTK/XDMF files already written | Drop | _pending_ |
| M9 | Mercurial `get_hg_revision_info` | Dead pre-Git tooling | Drop (delete) | _pending_ |
| M11 | Transposed-Robertson SciPy stiff xfail | Upstream SciPy regression, not Finmag | Drop | _pending_ |
| M12 | Weak-Krylov-demag tolerance xfail | Historical tolerance expectation | Drop / review | _pending_ |
| L20 | Long-tail: varying cubic axes, string `Expression` coefficients, matrix/project/direct energy methods | DOLFINx has no `Expression`; box-assemble is the method | Drop / deferral | _pending_ |
| L21 | `batch_task` sweep tooling | Untested even on master | Drop | _pending_ |
| — | `FixedEnergyDW` | Untested on master; legacy notes call it "broken"; depends on deferred treecode | Drop | _pending_ |
| — | `Demag2D` | Deferred by name; low usage | Drop / port-later | _pending_ |

## Still-planned (NOT drops — remaining port work)

Normal modes/eigenmodes + FFT (Task 25), thermal SLLG/LLB (Task 24), OOMMF/Nmag
comparison harnesses (Task 27, data checked in), I/O long tail — HDF5 read-back,
plotting, point-probing, region field output, skyrmion helpers (Task 26), MPI
(Task 28). These are scheduled work, not accepted drops.
