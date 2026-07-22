# Finmag DOLFINx Port — Deviation & Decision Register (Task 29)

Every documented deviation from legacy behavior, and every "accept-drop"
candidate from the master-parity audit, gathered in one place. Each item is
test-pinned and documented in `transition-notes.org` /
`dev/dolfinx/porting_map.md`.

**These decisions are DEFERRED to the repository owner** (fangohr/finmag). The
porting team does not rule on them; this register exists so the owner can
decide each one at review time with full evidence. Until then every item's
behavior is what the table's "Recommendation" column describes, is test-pinned,
and does not block further porting work. "Disposition" stays _deferred (owner)_
for all rows unless the owner rules otherwise.

## A. Behavioral deviations (accept, or request a fix)

| # | Deviation | Why it exists | Recommendation | Disposition |
|---|---|---|---|---|
| D1 | `DMI(dmi_type='D2D')` raises by name | `D2D` was never in the legacy class's documented `dmi_type` options | Accept (deferral) | _deferred (owner)_ |
| D2 | Legacy K2 native-field typo NOT reproduced — port ships the correct per-node field | `energy.cc` has a fixed-index bug; dormant for constant K2, divergence-pinned for varying K2 | Accept (correct physics) | _deferred (owner)_ |
| D3 | `DiscreteTimeZeeman.compute_energy()` stale after interval update — legacy bug PRESERVED | Bug-compatible with legacy; `compute_field()` stays correct | **Your call: keep vs fix** | _deferred (owner)_ |
| D4 | `hysteresis()`/`hysteresis_loop()` don't independently re-relax after first stage — legacy defect PRESERVED | Legacy was equally broken (oracle-confirmed); `relax()` itself works | **Your call: keep vs fix** | _deferred (owner)_ |
| D5 | `TimeZeemanPython` vector-valued `time_fun` raises by name | Narrow branch, untested even on legacy | Accept (deferral) | _deferred (owner)_ |
| D6 | Constant anisotropy axis normalised; spatially varying axis used as-given | Restores the legacy cosine contract for constant axes; varying-axis matches legacy | Accept | _deferred (owner)_ |
| D7 | `Field.from_function` interpolates on space mismatch (legacy raw-copied/errored) | Tested superset of legacy behavior | Accept | _deferred (owner)_ |
| D8 | `Simulation` default integrator = scipy (legacy default = sundials) | Factory `llg_integrator` default IS sundials (legacy-faithful); only `Simulation`'s own default differs, to avoid coupling every gate to the native build | **Your call: keep scipy vs restore sundials** | _deferred (owner)_ |
| D9 | STT spatially-varying J/Ms/p use coordinate ordering throughout | Legacy fed raw-dof arrays alongside coordinate-ordered m — a latent legacy inconsistency; port is self-consistent | Accept | _deferred (owner)_ |

## B. Accept-drop candidates (from the master-parity audit)

| # | Capability | Why droppable | Recommendation | Disposition |
|---|---|---|---|---|
| M2 | GCR demag solver | Legacy formulation judged unsound; FK is the validated path | Drop | _deferred (owner)_ |
| M3 | Compiled `Equation`/`terms` backend | Python fallback works, ~107× slower; DOLFINx LLG uses its own RHS anyway | Drop (or perf-port later) | _deferred (owner)_ |
| M7 | Magpar mesh-drift comparison xfails | Fixture node-ordering drift; physics fine | Drop (as xfail) | _deferred (owner)_ |
| M8 | Paraview/mencoder movie export | External renderer; VTK/XDMF files already written | Drop | _deferred (owner)_ |
| M9 | Mercurial `get_hg_revision_info` | Dead pre-Git tooling | Drop (delete) | _deferred (owner)_ |
| M11 | Transposed-Robertson SciPy stiff xfail | Upstream SciPy regression, not Finmag | Drop | _deferred (owner)_ |
| M12 | Weak-Krylov-demag tolerance xfail | Historical tolerance expectation | Drop / review | _deferred (owner)_ |
| L20 | Long-tail: varying cubic axes, string `Expression` coefficients, matrix/project/direct energy methods | DOLFINx has no `Expression`; box-assemble is the method | Drop / deferral | _deferred (owner)_ |
| L21 | `batch_task` sweep tooling | Untested even on master | Drop | _deferred (owner)_ |
| — | `FixedEnergyDW` | Untested on master; legacy notes call it "broken"; depends on deferred treecode | Drop | _deferred (owner)_ |
| — | `Demag2D` | Deferred by name; low usage | Drop / port-later | _deferred (owner)_ |

## Still-planned (NOT drops — remaining port work)

Normal modes/eigenmodes + FFT (Task 25), thermal SLLG/LLB (Task 24), OOMMF/Nmag
comparison harnesses (Task 27, data checked in), I/O long tail — HDF5 read-back,
plotting, point-probing, region field output, skyrmion helpers (Task 26), MPI
(Task 28). These are scheduled work, not accepted drops.
