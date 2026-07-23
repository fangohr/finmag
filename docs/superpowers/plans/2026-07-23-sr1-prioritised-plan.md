# Finmag SR1 Prioritised Implementation Plan

**Decision basis:** owner checklist reviewed with Hans Fangohr; behavioral
answers relayed 2026-07-23.

**Target:** the earliest trustworthy, deterministic serial micromagnetic
simulator with the selected everyday interface and explicit deferred backlog.

This plan supersedes historical numeric task ordering for SR1. Full-master
parity remains the eventual target. A checklist **Not now** tick means deferred,
not deleted and not an accepted permanent exception.

## Resolved interpretation of the owner checklist

- Exchange remains essential; only its obsolete matrix/project/direct assembly
  algorithms are not short-term work.
- The working serial LLG remains in SR1. Callable pin masks are Now; MPI is
  Later.
- MacroGeometry through dense FK demag is Now. Treecode factory exposure is
  Later/needs review and is not coupled to MacroGeometry work.
- `finmag.example` and barmini are Now; normal-mode exports remain Later.
- Existing checked-in OOMMF/Nmag/Magpar reference data is sufficient for SR1;
  live legacy external programs are not required.
- Netgen has no demonstrated required workflow yet. Investigate first; do not
  port the binary backend without evidence. The owner’s Now tick means the
  necessity probe is Now, not that the binary backend itself is pre-approved.
- Correct known physics defects rather than reproduce them. Sundials becomes
  the default after complete lifecycle validation. Conflicting STT modes raise.
- Restart supports coordinate-aware v2 only. String expressions and raw-dof
  snapshot compatibility are not required.

## Priority 0 — Freeze evidence before source changes

### [x] P0.1 Master/pixi parity manifest

- Inventory every master/pixi test and example relevant to owner-Now features.
- Classify each as already covered, needs a DOLFINx translation, Later, Not now,
  or obsolete external execution with reusable reference data.
- Record the exact interface, physical invariant, tolerance and validation type.
- **Non-goal:** no source or tolerance changes.
- **Gate:** reviewed file-by-file manifest; `git diff --check`.

### [x] P0.2 Full-lane baseline — evidence complete, acceptance not green

- Run the aggregate verifier and `FINMAG_EXAMPLE_FULL=1` lane unchanged.
- FULL mode runs the fourteen fast entries plus three slow entries; retain the
  wrapper's unchanged 120–3600-second per-example timeouts defined in
  `examples/test_examples_dolfinx.py` rather than introducing a new global
  timeout. Baseline wrapper timeouts are expected evidence to record, not
  physics failures; only after evidence is frozen may timeout scaling/wrapper
  adjustment be proposed as a separate reviewed slice before an all-green FULL
  acceptance claim.
- Use a disposable clean worktree, run the editable-install task there first,
  and restore the editable install to the main worktree afterwards.
- Record failures/timeouts, standard-problem observables, post-run `git status
  --short`, and diffs for tracked `examples/magnetic_grain/mz.png` and
  `examples/std_prob_3/doc_table.rst` before fixing anything.
- **Non-goal:** no opportunistic fixes in the baseline run.
- **Gate:** reproducible commands, logs and clean worktree.
- **Recorded:** aggregate exit 0/all 32 green; clean FULL exit 1 (12 passed,
  5 failed). See `master-pixi-parity-manifest.md` for logs, exact versions and
  the failure split. This records baseline evidence only; it does not accept
  wrapper timeouts as physics failures or make FULL acceptance green.

## Priority 1 — Trustworthy simulator lifecycle

### P1.1 Sundials reset and reinitialisation

- Remove the SciPy-only `.ode` assumption through the smallest backend-neutral
  lifecycle change.
- Verify existing tolerance/scheduling behavior before modifying it.
- **Gates:** new RED regression; Sundials, Simulation and aggregate gates.

### P1.2 Restart integrity on both backends

- Correct saved backend provenance.
- Compare uninterrupted and checkpoint/restart trajectories through the same
  final time on SciPy and Sundials.
- Retain v2-only rejection of legacy raw-dof archives.
- **Gates:** restart/output, both backend gates and aggregate verifier.

### P1.3 Restore Sundials as public default

- Change only public defaults and directly affected tests/docs after P1.1/P1.2
  prove equal lifecycle support.
- **Non-goal:** no removal of SciPy.
- **Gates:** signature/default tests, both lifecycle gates, examples, aggregate.

## Priority 2 — Everyday API and approved physics corrections

### P2.1 Core import boundary

- Make `finmag.example`/barmini and selected everyday conveniences import.
- Deferred normal-mode families must fail explicitly by feature name.
- **Non-goal:** no normal-mode implementation.

### P2.2 `sim_with` MacroGeometry

- Wire nx/ny/spacing arguments to the existing dense-FK MacroGeometry path.
- **Non-goals:** no Treecode selector, GCR or Demag2D work.

### P2.3 Callable pin masks

- Restore coordinate-to-dof callable pin selection without changing indexed
  pins or adding MPI stepping.

### P2.4 Correct discrete-time Zeeman energy

- Rebuild/update the energy consistently with the field after interval changes.
- Preserve an explicit regression showing the legacy stale-energy behavior.

### P2.5 Correct hysteresis stage relaxation

- Ensure every applied-field stage independently relaxes.
- Test switching/loop physics, not merely result lengths.
- Document why corrected output differs from master.

### P2.6 Reject conflicting STT modes

- Keep each mode independently working.
- Raise a clear error when configuration would enable both modes.

Each P2 slice gets a RED test, the narrow focused gate, closest legacy
invariants from P0.1, and the aggregate verifier. Do not combine these into one
public-interface commit.

## Priority 3 — Selected serial physics and Field completeness

1. Probe the exact legacy `Simulation(pbc=)` contract under `dev/dolfinx`.
2. Implement serial function-space PBC from the accepted probe; do not combine
   it with MacroGeometry or MPI.
3. Port `Field.cross`, `Field.dot` and scalar coercion.
4. Port `Field.from_generic_vector` separately with ordering and MPI-ownership
   tests.
5. Port spatially varying cubic axes with oracle and analytic witnesses; retain
   the owner-approved correct K2 physics.

## Priority 4 — Selected I/O and convenience surface

Implement as separate reviewable slices:

1. HDF5 readback/round-trip with a DOLFINx-native documented format.
2. Region/submesh field output.
3. Magnetisation initialisers, one formula family per slice.
4. `Simulation.probe_field*`.
5. LLG `M`/`M_average` compatibility.
6. `length_scales` and `mesh_info`.
7. logging/instance/shutdown helpers needed by selected scripts.
8. backend-neutral NumPy/Matplotlib helpers, followed by an optional PyVista
   adapter with headless tests. Do not reproduce the mencoder wrapper.

## Priority 5 — Mesh and external-reference validation

### P5.1 Netgen necessity probe

- Identify a selected test/geometry that Gmsh and the `.geo` subset cannot
  represent or validate.
- If none exists, leave Netgen deferred. If one exists, propose a separate
  conversion design before source edits.

### P5.2 Comparison data

- Restore OOMMF, Nmag and Magpar comparisons separately using checked-in data.
- Compare by coordinates and physical invariants, not raw node ordering.
- Do not resurrect `nsim` merely to reproduce historical execution.

### P5.3 Legacy tests

- Port tests in small feature-family clusters driven by P0.1.
- Skip/xfail only after the positive replacement test is ported, or for a named
  Later/Not-now capability with a linked acceptance-register reason.
- Never create a single mechanical whole-suite rewrite.
- These feature-family slices may run in parallel with other SR1 work, but SR1
  is not complete until every selected owner-Now legacy-test family has a
  positive DOLFINx witness or an explicitly linked owner disposition.

## Priority 6 — Full scientific acceptance and SR1 handoff

1. Run the full-resolution examples after preceding fixes land.
2. Add mesh-matched oracle or convergence/quantitative trajectory evidence for
   standard problem 4; its broad switching window remains qualitative only.
3. Execute minimal and realistic demag/dynamics examples from a clean install.
4. Publish supported API, validation limits, Later and Not-now lists.
5. Declare SR1 only when all selected Now rows are working or explicitly moved
   by a new owner prioritisation decision, P0.2 is recorded, and the P5.3
   legacy-test-family gate is satisfied.

## Deferred beyond SR1

Normal modes/eigensolvers; ringdown and FFT/PSD; legacy NEB; a separate GNEB
PR; MPI stepping; DMI D2D; possible Treecode factory exposure; compiled
Equation performance; and movie-generation conveniences.

Thermal SLLG, LLB, nonlocal STT, Demag2D, GCR, FixedEnergyDW, specialist mesh
generators, nmesh conversion, old energy assembly algorithms, batch tooling and
Mercurial helpers are **Not now**, not permanently deleted. Revisit them during
later full-parity planning.

[Codex GPT-5]
