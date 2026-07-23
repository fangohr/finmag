# Finmag DOLFINx Deviation and Decision Register

**Reconciled:** 2026-07-23

**Decision authority:** repository owner

**Default until decided:** no permanent exception is accepted

This is the sole disposition ledger. `capability-status.md` says what works;
this file records where the DOLFINx behavior differs from original `master` or
where an original capability might be omitted. Plans and chronological notes
may supply evidence, but they must link back here rather than invent a second
decision list.

“Recommendation” is technical advice, not approval. Only the repository owner
may change **Disposition** from *pending owner decision*. A row may be handled
after SR1 without being dropped from the final parity target.

For the owner-meeting triage sheet, see
[`owner-porting-checklist.md`](owner-porting-checklist.md). Approved exceptions
from that meeting must then be copied into atomic rows here.

The checklist's **Not now** column means deferred beyond SR1, not permanently
dropped. Those capabilities remain in the parity backlog unless a later owner
decision explicitly accepts an exception here.

Every row below is intended to describe one independently selectable decision.
If later evidence reveals another bundled row, split it before recording an
owner disposition; never use one decision to approve unrelated capabilities.

## A. Behavioral deviations

| ID | Difference from original behavior | Evidence/reason | Recommendation | Disposition |
|---|---|---|---|---|
| D1 | `DMI(dmi_type='D2D')` raises by name | The variant exists in legacy source but was absent from its documented option list | Defer until after SR1, then port or explicitly accept omission | *pending owner decision* |
| D2 | The legacy spatially varying K2 native-field indexing bug is not reproduced | Legacy `energy.cc` used a fixed index; constant K2 hid the defect. Port follows the correct energy derivative and has a divergence pin | Keep the corrected physics | **approved 2026-07-23 (owner meeting)** |
| D3 | `DiscreteTimeZeeman.compute_energy()` remains stale after an interval update | Oracle confirms the legacy defect; field computation remains current | Fix and clearly document the legacy behavior | **approved fix 2026-07-23 (owner meeting)** |
| D4 | `hysteresis()`/`hysteresis_loop()` do not independently re-relax after the first stage | Oracle confirms the legacy workflow defect was preserved | Fix so every field stage relaxes; clearly document the correction | **approved fix 2026-07-23 (owner meeting)** |
| D5 | `TimeZeemanPython` rejects vector-valued `time_fun` by name | Narrow legacy branch lacked useful tests during the port | Defer until after SR1, then port unless owner accepts omission | *pending owner decision* |
| D6a | A constant `UniaxialAnisotropy` axis is normalised | Normalisation restores the cosine contract for a constant axis | Accept and explain in user documentation | *pending owner decision* |
| D6b | A spatially varying uniaxial-anisotropy axis is used as supplied | This follows legacy behavior; it does not describe cubic-anisotropy axes | Decide whether to require pointwise normalisation or document the supplied-axis contract | *pending owner decision* |
| D7 | `Field.from_function` interpolates across compatible space mismatches | Backward-compatible superset; avoids raw-copy failure | Accept | *pending owner decision* |
| D8 | `Simulation` and `sim_with` defaulted to SciPy rather than legacy Sundials | `81fab481` restores the public defaults to native Sundials after P1.1/P1.2 validated reset/reinitialisation, scheduling, save and restart on both backends. It has signature/default, native construct-and-advance and explicit-SciPy regressions; the core smoke records `integrator_backend: sundials` at `t=1e-12` | Keep native Sundials as the public default; retain SciPy as a fully supported explicit opt-in | **approved condition discharged and implemented 2026-07-23 (`81fab481`)** |
| D9 | Spatially varying STT inputs use coordinate ordering consistently | Legacy mixed raw-dof J/Ms/p with coordinate-ordered m/H/alpha; the port removes a latent physics inconsistency | Accept corrected ordering | *pending owner decision* |
| D10 | Out-of-range pin indices raise `ValueError` | Legacy logged an error and silently retained the previous pins, risking stale constraints | Keep fail-fast `ValueError` | **approved 2026-07-23 (owner meeting)** |
| D11 | Current port silently uses last-call-wins when Slonczewski and Zhang-Li are both configured; legacy silently preferred Slonczewski | Neither silent precedence rule is an acceptable public contract | Approved target: reject the second conflicting configuration with a clear error | **approved 2026-07-23 (owner meeting)** |
| D12 | Restart v2 is coordinate-aware and rejects legacy raw-dof v1 | Raw legacy arrays cannot be mapped safely without coordinates; the owner has no v1 archives requiring support | Support v2 only and reject v1 clearly | **approved 2026-07-23 (owner meeting)** |
| D13 | Reading `Simulation.t` no longer creates an integrator | Avoids a surprising getter side effect; backend lifecycle correctness is tracked as implementation work, not a decision | Accept side-effect-free `t` | *pending owner decision* |
| D14 | `.npy` field snapshots store coordinate/value tables rather than legacy raw backend dofs | Coordinate-aware output is stable across FEM ordering changes | Keep coordinate/value format; do not add raw-format compatibility | **approved 2026-07-23 (owner meeting)** |
| D15 | Legacy string `Expression`/`UserExpression` inputs are replaced by Python callables or rejected | DOLFINx has no drop-in equivalent of the old runtime string/UserExpression contract | Use documented vectorised Python callables; reject strings clearly | **approved 2026-07-23 (owner meeting)** |
| D16a | Restart reconstructs magnetisation/time but does not reapply or validate material and interaction metadata | This matches the practical legacy reconstruction pattern, but may not satisfy every selected restart workflow | Inventory whether any selected workflow needs full-state reconstruction | *pending owner decision* |
| D16b | Restart metadata represents varying material fields only as scalar summaries | A scalar summary is lossy and cannot reconstruct spatially varying `Ms`, alpha or interaction coefficients | Define a lossless representation if full-state reconstruction is selected; otherwise label metadata informational | *pending owner decision* |

## B. Possible full-parity exceptions

These are candidates, not approved drops. “Later” means they do not block SR1
but still belong to the master-parity backlog.

| ID | Capability | Current reason for considering an exception | Recommendation | Disposition |
|---|---|---|---|---|
| M1 | `nsim`/Nmag live reference generation | Obsolete external stack; checked-in reference data can validate selected SR1 cases | Defer live generation for SR1 and retain useful checked-in comparisons; decide permanent omission later | **SR1 deferment approved 2026-07-23; permanent omission pending** |
| M2 | GCR demag solver | The surviving legacy Python path appears incomplete and its scientific correctness is unclear; silently mapping it to FK would be misleading | Inventory intended formulation and usage, then port or accept omission | *pending owner decision* |
| M3 | Compiled `Equation`/`terms` backend | Python fallback is much slower; DOLFINx LLG uses its own RHS | Defer performance work; drop only if no supported workflow depends on it | *pending owner decision* |
| M4a | Netgen binary backend | Common geometries now use Gmsh and a text-subset loader; no selected workflow has yet proved the binary backend necessary | Run the SR1 necessity probe; defer unless it identifies a required geometry | *pending owner decision* |
| M4b | `nmesh_to_dolfin` conversion | Its legacy dolfin-XML output cannot be loaded by DOLFINx and the owner does not require live Nmag generation for SR1 | Defer; reconsider only with a concrete data-conversion workflow | *pending owner decision* |
| M5 | Nonlocal `LLG_STT` spin-accumulation model | Distinct native capability, not covered by the ported local STT terms | **Not now**; retain for later full parity unless owner accepts omission | *pending owner decision* |
| M6 | `FixedEnergyDW` | No master tests; a legacy note calls it broken. Its old dolfin-XML/treecode workflow is obsolete even though treecode itself is now ported | Drop unless a real scientific use case is identified | *pending owner decision* |
| M7 | `Demag2D` | Specialist low-usage solver path is unported | Later; decide after a physics/user inventory | *pending owner decision* |
| M8 | Magpar mesh-drift comparison xfails | Saved nodes no longer match regenerated meshes; field physics should be compared by coordinates/invariants instead | Replace useful tests; drop obsolete node-order xfails | *pending owner decision* |
| M9 | Paraview/mencoder movie export | External historical renderer; VTK/XDMF write is available | Drop the movie wrapper, retain data export | *pending owner decision* |
| M10 | Mercurial `get_hg_revision_info` | Dead pre-Git tooling | **Needs decision:** do not drop until the owner confirms the permanent omission | *pending owner decision* |
| M11a | Historical transposed-Robertson SciPy xfail (`src/finmag/util/ode/tests/test_sundials_stiff_ode.py`) | The failure was attributed to an upstream SciPy/VODE regression rather than Finmag physics | **Needs decision:** re-run only if the corresponding solver contract is selected; do not copy the xfail mechanically | *pending owner decision* |
| M11b | Historical weak-Krylov-demag tolerance xfail (`src/finmag/tests/test_interactions_scale_linearly_with_m.py`) | This is a separate numerical-tolerance artifact whose relevance under DOLFINx has not been established | Re-derive the expectation before deciding whether any test remains useful | *pending owner decision* |
| M12a | Legacy matrix energy assembly (`src/finmag/energies/energy_base.py`) | Box assembly is the supported implementation and the owner marked the old algorithm Not now | Defer; inventory callers before any permanent omission | *pending owner decision* |
| M12b | Legacy project energy method (`src/finmag/energies/energy_base.py`) | Box assembly is the supported implementation and the owner marked the old algorithm Not now | Defer; inventory callers before any permanent omission | *pending owner decision* |
| M12c | Legacy direct energy method (`src/finmag/energies/energy_base.py`) | Box assembly is the supported implementation and the owner marked the old algorithm Not now | Defer; inventory callers before any permanent omission | *pending owner decision* |
| M13 | `batch_task` sweep tooling | Untested even on master | Drop unless a current workflow needs it | *pending owner decision* |
| M14 | OOMMF live reference generation | The owner accepted checked-in OOMMF data as sufficient for selected SR1 comparisons | Defer live execution for SR1; decide later whether reproducible regeneration remains valuable | **SR1 deferment approved 2026-07-23; permanent omission pending** |
| M15 | Magpar live reference generation | The owner accepted checked-in Magpar data as sufficient for selected SR1 comparisons | Defer live execution for SR1; retain coordinate/invariant comparisons and decide permanent omission later | **SR1 deferment approved 2026-07-23; permanent omission pending** |
| M16 | Legacy Heun driver | It is outside the current SciPy/Sundials implementation surface, but no owner decision has selected it for omission | Keep as a needs-decision parity item; do not label it Not now or obsolete without owner direction | *pending owner decision* |

## C. Required later work, not drop candidates

The following remain part of full master parity unless the owner later adds a
specific decision row: thermal SLLG/LLB; normal-mode linearisation and
eigensolvers; ringdown and FFT/PSD analysis; legacy NEB; general MPI stepping;
`Simulation(pbc=)` function-space periodicity; HDF5 readback; plotting and
region/submesh field output; initialisers and mesh/utilities used by supported
legacy workflows; and the useful portions of the OOMMF/Nmag/Magpar comparison
suite.

No public GNEB API/workflow was found on original master, so it is not currently
an interface-parity item. Legacy spherical/modified NEB code may have
algorithmic overlap; classify that only during the NEB inventory.

[Codex GPT-5]
