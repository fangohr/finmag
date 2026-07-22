# Finmag DOLFINx Port — Status, Usability & Agent Handover

**Date:** 2026-07-22 · **Branch:** `dolfinx-parity` · **Maintainer contact:** Sam Holt

This is the single entry point for any agent or human taking over the port.
Read this, then the progress ledger, then the plan files — in that order.

## 1. What this project is

Finmag (legacy micromagnetics code, Python 2 + dolfin 2017) is being ported to
DOLFINx 0.10 / Python 3.12 with a hard contract: **the final repo must have the
same functionality as the original `master`, with no silent behavior changes.**
Every deviation is documented, test-pinned, and awaits explicit user acceptance.

Branch lineage (verified):

```
master (b5015c5a)      original Python 2 / dolfin-2017 code — the functionality yardstick
  └─ pixi (ba928093)   Python 3 + FEniCS 2019 transition (M1-M3) — the FROZEN LEGACY ORACLE
       └─ dolfinx-port (ce60ae3e)   Phase 1: core DOLFINx port (Tasks 1-12) — awaiting human review
            └─ dolfinx-parity      Phase 2 (Tasks 13-16) + Phase 3 (Tasks 17-31) — ACTIVE
```

## 2. How usable is it right now?

**Usable today, serial, via the legacy API** (`import finmag`; `sim_with` /
`Simulation`). Install:

```
pixi run -e dolfinx dolfinx-install-editable   # editable install (see README INSTALL section)
pixi run -e dolfinx dolfinx-native-build       # builds bem_arrays.so, sundials.so, treecode_bem
dev/bin/verify-dolfinx-m5                      # the aggregate gate (~21 sub-gates), must be green
```

Working, oracle-validated capabilities:

| Area | Status |
|---|---|
| Energies | Exchange, DMI (all documented variants), Zeeman + full time-dependent family, uniaxial + cubic anisotropy (both field paths), ThinFilmDemag, FK demag (compiled BEM), treecode/PBC demag via `MacroGeometry` |
| Dynamics | LLG (`run_until`, `relax`), STT (Slonczewski + Zhang-Li), SciPy AND native CVODE integrators (factory default = sundials, legacy-faithful; `Simulation` default = scipy, registered deviation) |
| Materials | Spatially varying Ms/A/K/D/alpha; regions (`mark_regions`, per-region energies/averages) |
| Meshes | Gmsh-bridge generators (box/sphere/cylinder/ellipsoid/nanodisk/elliptical/ring/cone/CSG combos) + legacy Netgen-CSG `.geo` files via `from_geofile` (subset parser); md5-keyed caching |
| I/O | Restart (coordinate-aware v2), NDT tables, VTK/XDMF write, scheduler |
| Examples | `examples/` converted at the minimal-changes bar (Task 30): µMAG std_prob_3/4, exchange_demag vs nmag/OOMMF reference data, macrospin, precession, etc. — gated in CI |
| Packaging | `pyproject.toml`, editable install, no `PYTHONPATH` needed |

NOT yet ported (raise by name): normal modes/eigenmodes + FFT/PSD (Task 25),
thermal SLLG/LLB (Task 24), OOMMF/Nmag/Magpar comparison harnesses (Task 27),
plotting/visualization helpers + HDF5 read-back + point probing (Task 26),
`Simulation(pbc=)` function-space PBC, MPI-parallel stepping (Task 28),
NEB, GCR demag (accept-drop candidate).

## 3. Where execution stands

- Phase 1 (Tasks 1-12): DONE, sealed by a whole-branch review; branch
  `dolfinx-port` awaits the human merge review.
- Phase 2 Tier 1 (Tasks 13-16): DONE, sealed by a critical whole-branch
  (Fable) review with a composed-physics guard test.
- Phase 3 (audited plan, Tasks 17-29 + inserted 30/31): 18, 19, 20, 21(=17),
  22, 23, 30 DONE and review-approved. **Task 31 (public array
  component-ordering correction, user-directed) is the in-flight slice** —
  implementer done/pending, Fable review pending at time of writing; see the
  ledger for its outcome.
- Remaining after 31: 24 (SLLG/LLB), 25 (normal modes/NEB/FFT — user cares;
  may be pulled ahead of 24), 26 (I/O long tail), 27 (harnesses), 28 (MPI +
  CI), 29 (accept-drop sign-off).
- **Execution is PAUSED after Task 31 + its review by user directive.** Do
  not dispatch further slices without user go-ahead.

## 4. The acceptance-pending register (user must sign off; Task 29)

All documented in `transition-notes.org` + `dev/dolfinx/porting_map.md`,
each test-pinned: (1) DMI `D2D` variant deferred; (2) legacy K2 native-field
typo not reproduced (correct field shipped; dormant for constant K2, live
divergence pin for varying K2); (3) `DiscreteTimeZeeman` stale-energy legacy
bug preserved; (4) hysteresis no-re-relax legacy defect preserved; (5)
`TimeZeemanPython` vector-`time_fun` deferred by name; (6) constant-axis
normalisation (cosine contract restored) vs varying-axis as-given; (7)
`Field.from_function` interpolation superset; (8) `Simulation` default
backend scipy vs legacy sundials (factory default IS sundials); (9) STT
spatially-varying input ordering made self-consistent (legacy fed raw-dof);
(10) Task 30 drift-table rows still open after Task 31 closes rows #3/#6
(ordering) — see the drift table. Plus the audit's accept-drop candidates
(GCR, compiled Equation backend, nsim, Mercurial helper, Paraview movie
export, 3 historical xfails, batch_task) consolidated in plan Task 29.

## 5. Working conventions (binding for any successor agent)

- **Per-slice protocol**: `docs/superpowers/plans/2026-07-06-dolfinx-core-port.md`.
  Oracle = `dev/bin/run-legacy-oracle` at pixi tip `ba928093…`; fixture schema
  `docs/superpowers/specs/legacy-oracle-fixtures.md`; formulas from C++/legacy
  source, never docstrings; deviations by-name + documented + registered;
  TDD RED-first; docs (plan checkboxes, transition-notes, porting_map) IN the
  slice commit (verify with `git show --stat`).
- **Plans**: `2026-07-21-dolfinx-full-parity.md` (Tasks 13-31) +
  `2026-07-21-master-parity-audit.md` (the two-layer gap register).
- **Progress ledger** (git-ignored, authoritative session memory):
  `.superpowers/sdd/progress.md` — read it fully before resuming; per-task
  reports/briefs live beside it.
- **Gate**: every slice adds a `dolfinx-src-*` pixi gate folded into
  `dev/bin/verify-dolfinx-m5`; the whole thing must stay green; the
  cleanliness guard forbids tests dirtying tracked files.
- **Orchestration ladder** (user-set): Sonnet default; Opus for hard slices +
  physics reviews; Fable sparingly for critical reviews (new-physics
  transcription, whole-branch, stochastic) and hard plans. Every slice:
  implement → independent review → fix round → re-review before closing.
- **User's standing requirements**: minimal changes (especially examples);
  no silent behavior changes; full master parity as end state; pause points
  respected.

## 6. Known open threads (beyond the plan tasks)

- Verify std_prob_4 FULL-resolution runtime on the next `FINMAG_EXAMPLE_FULL=1` run.
- `_owned_vertex_to_dof` distinctness guard correct-by-inspection but untested.
- Legacy-lane treecode activation (pixi-era skip) left as follow-up.
- Accumulated per-task Minors are listed in the ledger at each "deferred to
  roll-up" line — sweep them at the next whole-branch review.
- Phase 3 whole-branch Fable review still to be scheduled (after remaining
  slices or at the user's request).
