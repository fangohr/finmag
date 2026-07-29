> **ARCHIVED (2026-07-29).** Historical record of the porting process; statements reflect their writing date. Current truth: docs/README.md.

# SR1 batch-ratification decision sheet

**Date:** 2026-07-27 (SR1 S5a). **Branch:** `dolfinx-parity`.
**For:** the repository owner. **Decision authority:** owner only.

This sheet exists so SR1 can be declared with **zero rows left "pending owner
decision"** (acceptance criterion 2). It enumerates every open row in
[`acceptance-register.md`](../acceptance-register.md) — the authoritative
ledger — plus two new row candidates surfaced during SR1 execution (D32, P1).
Each entry is three lines: what the state actually is, what is recommended,
and where the evidence lives.

**Note on scope:** the executing brief named only three section-B rows
(M4a, M4b, M5), but 15 further M-rows sit at *pending owner decision* in the
register; acceptance criterion 2 forces all 20 into this sheet. If you
intended M6–M16 to stay open past SR1, that criterion is what needs relaxing.

**This sheet does not change the register.** After your line-item decisions
return, one commit (SR1 S5b) writes them into the register. Any row you
redirect to **FIX NOW** becomes its own slice and blocks SR1 declaration until
it lands.

## The lens: your four acceptance criteria (2026-07-27)

1. Everything selected for SR1 works, with a passing witness in a gated lane.
2. Physics and interface validated unchanged, or minimally and explicitly
   changed — every deviation fixed or ratified; zero pendings at declaration.
3. Clear documentation of working vs waiting-to-port (`docs/SUPPORTED.md`, S6).
4. Delayed functionality **fails correctly in kept tests**, via standard pytest
   `xfail(strict=True)`. Tests for unported functionality are kept, never
   deleted or silently skipped.

Full text: [`2026-07-27-sr1-completion-design.md`](2026-07-27-sr1-completion-design.md),
section "Owner acceptance criteria".

## Decision verbs

| Verb | Meaning | Effect on the parity backlog |
|---|---|---|
| **ACCEPT** | Ratify the current DOLFINx behaviour as the permanent contract | Row closes; nothing left to do |
| **DEFER** | SR1 disposition stands; the gap stays a named full-parity backlog item | Row closes as *decided-deferred*, not a permanent exception |
| **DROP** | Permanently omit the capability from the parity target | Row closes; capability leaves the target |
| **RESTORE** | Restore deleted master tests verbatim as retained never-ported backlog | Row closes; criterion-4 artifact reinstated |
| **FIX NOW** | Blocks SR1 declaration until a fix slice lands | New slice |

Rows marked **(owner knowledge decisive)** are ones where the recommendation
rests on "no selected workflow needs this" — only you can confirm that.

---

## A. Behavioural deviations

### D1 — `DMI(dmi_type='D2D')` raises by name
- **State:** the undocumented legacy `'D2D'` variant is refused by name with
  `NotImplementedError`; every other `dmi_type` option works.
- **Recommend: DEFER** — port the variant under full parity. It was absent from
  its own documented option list on master, so nothing selected depends on it.
- **Evidence:** `src/finmag/tests/test_dmi.py:182`
  `test_d2d_dmi_type_is_deferred_by_name`; register D1.

### D5 — `TimeZeemanPython` rejects vector-valued `time_fun` by name
- **State:** a narrow legacy branch, refused by name; the by-name deferral is
  exercised in a gated lane.
- **Recommend: DEFER** — port under full parity; scalar `time_fun` (the tested
  legacy path) works.
- **Evidence:** `src/finmag/energies/zeeman_test.py:662`
  `test_time_zeeman_python_vector_time_fun_is_deferred_by_name`; register D5.

### D6a — a constant `UniaxialAnisotropy` axis is normalised
- **State:** the port normalises a constant easy axis, restoring the cosine
  contract the energy expression assumes; master used the axis as supplied.
- **Recommend: ACCEPT** — a non-unit axis silently rescales K1 on master; this
  is the corrected reading of the documented contract. Document in
  `docs/SUPPORTED.md` (S6).
- **Evidence:** `src/finmag/energies/anisotropy.py`; register D6a.

### D6b — a spatially varying uniaxial axis is used as supplied
- **State:** the varying-axis path follows legacy behaviour exactly (no
  pointwise normalisation). The register asks you to choose between requiring
  normalisation and documenting the supplied-axis contract.
- **Recommend: ACCEPT the supplied-axis contract, documented** — it is the
  minimum-change option (criterion 2) and matches master bit-for-bit; adding
  pointwise normalisation would be a new divergence, not a fix.
- **Evidence:** `src/finmag/energies/anisotropy.py`; register D6b.

### D7 — `Field.from_function` interpolates across compatible space mismatches
- **State:** where master raised on a raw-copy mismatch, the port interpolates
  between compatible spaces (e.g. DG0 into CG1). Strict superset: every input
  master accepted still behaves identically.
- **Recommend: ACCEPT** — backward-compatible, and the register already
  recommends acceptance.
- **Evidence:** `src/finmag/field_test.py:1694`
  `test_from_function_interpolates_dg0_into_cg1`; register D7.

### D9 — spatially varying STT inputs use coordinate ordering consistently
- **State:** legacy mixed raw-dof `J`/`Ms`/`p` with coordinate-ordered
  `m`/`H`/`alpha` in the same expression — a latent physics inconsistency that
  only bit for spatially varying inputs. The port is coordinate-ordered
  throughout (`xxx` layout).
- **Recommend: ACCEPT the corrected ordering** — reproducing the legacy mixture
  would mean shipping a known wrong answer for varying STT inputs.
- **Evidence:** the coordinate-ordered inputs themselves —
  `src/finmag/physics/llg.py:601` (`p`, via
  `get_ordered_numpy_array_xxx()`), `:627` (`_J` as a `Field`), `:663-670`
  (the assembled `grad_m` term returned in `xxx` order) and `:728`
  (`_dmdt_zhangli_numpy`, which consumes them). Uniform-input agreement
  pinned against the legacy oracle in `src/finmag/tests/test_stt.py:290`
  `test_zhangli_rhs_and_gradient_match_legacy_oracle_fixture`; register D9.

### D13 — reading `Simulation.t` no longer creates an integrator
- **State:** master's `t` getter lazily constructed an integrator as a side
  effect; the port's getter is side-effect free and the integrator is still
  created lazily on first real use.
- **Recommend: ACCEPT** — a getter with a construction side effect is not a
  contract worth preserving; lazy creation is retained where it belongs.
- **Evidence:** `src/finmag/sim/sim_test.py:773`
  `test_t_is_zero_before_integration` (asserts `not sim.has_integrator()`),
  paired with `sim_test.py:858` `test_integrator_created_on_first_use`.

### D16a — restart does not reapply or validate material/interaction metadata
- **State:** restart reconstructs magnetisation and time; materials and
  interactions are rebuilt by the user's own script, matching the practical
  legacy reconstruction pattern. The register asks for a workflow inventory.
- **Recommend: ACCEPT as the SR1 restart contract, documented** — every
  selected restart workflow rebuilds the `Simulation` in script and then
  restores state; no selected workflow requires full-state reconstruction.
  *(owner knowledge decisive)*
- **Evidence:** `src/finmag/tests/test_restart_output.py:199,232,498`
  (`test_restart_roundtrip_same_simulation`,
  `test_restart_cross_instance_same_mesh_recipe`,
  `test_restarted_trajectory_matches_uninterrupted_run`); register D16a.

### D16b — restart metadata records varying material fields as scalar summaries
- **State:** the summary is lossy and cannot reconstruct spatially varying
  `Ms`/alpha/coefficients. Its status depends entirely on D16a.
- **Recommend: ACCEPT, relabelled *informational*** — follows directly from
  accepting D16a; the register's own conditional ("otherwise label metadata
  informational"). If you instead pick full-state reconstruction on D16a, this
  row becomes FIX NOW and needs a lossless representation.
- **Evidence:** `src/finmag/tests/test_restart_output.py:166`
  `test_restart_stores_coordinate_aware_format`; register D16b.

### D17 — `sim_with(nx=...)` refuses `pitch <= extent` by name (macro-geometry demag)
- **State:** SR1 disposition already approved 2026-07-23 (keep the by-name
  refusal). Open only as *permanence*: the coincident-node BEM defect (row sums
  -2/-3, ~158% cube field error) is real and unfixed, and whether master
  computed it correctly is still **unverified**.
- **Recommend: DEFER (not a permanent exception)** — the refusal stands as the
  SR1 contract; the BEM coincident-node kernel fix stays a named full-parity
  backlog item. A strictly larger pitch (`extent*(1+1e-6)`) reproduces the
  legacy reference to 0.08%.
- **Evidence:** `src/finmag/energies/demag/demag_pbc_test.py::test_field_1d`
  and `::test_field_2d` — master's assertions transcribed verbatim, marked
  `xfail(strict)` against this defect (measured ~50x / ~22x master's bounds);
  register D17.

### D19 — function-space periodic boundaries (`Simulation(pbc='1d'/'2d')`) deferred
- **State:** SR1 deferment already approved. **Blocked**, not merely unwired:
  `dolfinx_mpc` is absent and DOLFINx 0.10.0 `fem.functionspace` has no
  `constrained_domain`; a probe proved periodicity cannot be recovered by
  post-hoc value copying. Open only as *permanence*.
- **Recommend: DEFER (not a permanent exception)** — keep the by-name
  `_deferred('pbc')` guard; revisit under full parity by adding `dolfinx_mpc`
  (moderate) or hand-rolling a reduced dofmap (invasive). Note the demag
  image-lattice PBC is a separate, working feature.
- **Evidence:** `src/finmag/sim/sim_test.py:1206` `test_pbc_is_deferred`;
  `src/finmag/tests/test_energies.py::test_exchange_periodic_boundary_conditions`
  (restored but degenerate under D19 — disclosed in the register); register D19.

### D20 — `LLG.M` / `LLG.M_average` implemented as correct physics
- **State:** frozen master's `LLG.M` raised `RuntimeError` on *every* access,
  and `M_average` divided an integral by itself (`volume_Ms == volume`),
  collapsing to the dimensionless `m_average` against a docstring promising
  A/m. The port ships `M = Ms*m` and the Ms-weighted volume average. **Zero**
  consumers exist even in frozen master.
- **Recommend: ACCEPT the corrected physics** — there is no working legacy
  behaviour to preserve and no oracle or workflow regresses.
- **Evidence:** `src/finmag/tests/test_llg.py:556`
  `test_M_average_diverges_from_legacy_dimensionless_m_average` (divergence
  pin); register D20.

### D21 — `save_m_in_region` restores the documented intent with a changed call convention
- **State:** frozen legacy `save_m_in_region` raised `AttributeError` on every
  call (it wrote `tablewriter.entities`, which does not exist) — the exact
  error recorded in the shipped notebook. The port registers the intended
  per-region `<name>_m_{x,y,z}` `.ndt` column but takes a region **id**
  (requires `mark_regions`) rather than a subdomain-marking function.
- **Recommend: ACCEPT** — no working legacy behaviour existed to reproduce, and
  the id convention follows the port's region architecture. Region-restricted
  *submesh field* output stays separately deferred.
- **Evidence:** `src/finmag/tests/test_variable_params.py:884`
  `test_save_m_in_region_registers_ndt_column_with_region_average`;
  implementation `src/finmag/sim/sim.py:828`; register D21.

### D22 — on-outer-face point probes resolve to the wrong boundary vertex (~0.5 rel. error)
- **State:** a real defect in `Field.probe` / `evaluate_at_point` /
  `Simulation.probe_field` for points lying exactly on an outer mesh face;
  strictly interior points are correct. Confirmed by multiple agents. Already
  documented-deferred 2026-07-25; open only as *permanence*.
- **Recommend: DEFER (fix in a later slice)** — it is a probe-location
  artifact, not a field-physics difference, and every affected comparison test
  works around it structurally today. Keep it as a named backlog defect, not a
  permanent exception.
- **Evidence:** `src/finmag/tests/comparison/exchange/test_exchange_compare_magpar.py:240`
  `test_interior_probe_matches_nodal_value` (interior-only workaround) and
  `:213` (boundary via coordinate-keyed nodal lookup);
  `src/finmag/energies/demag/fk_demag_test.py` (centre-only); register D22.

### D23 — `set_m()` silently accepts a NaN `m_init`
- **State:** master raised `ValueError` on a NaN initial magnetisation; the
  port leaves `sim.m` containing NaN. Pinned `xfail(strict)`. Documented-deferred
  2026-07-25; open only as *permanence*.
- **Recommend: DEFER (fix in a later slice)** — restoring the guard is small
  and clearly right; it should not become a permanent exception. Consider
  **FIX NOW** if you consider a silently-NaN simulation an SR1-blocking hazard;
  it is the cheapest fix on this sheet.
- **Evidence:** `src/finmag/sim/sim_test.py::test_set_m` (`xfail(strict)` pin);
  register D23.

### D24 — teardown surface (`shutdown`, `instances_delete_all_others`, `close_logfile`) absent
- **State:** the port deliberately dropped master's cyclic-reference teardown
  machinery for a plain `instances` dict, so master's `test_clean_up` surface
  is unported. Documented-deferred 2026-07-25; open only as *permanence*.
- **Recommend: DEFER (port or accept in a later sim.py slice)** — a
  long-running many-`Simulation` script is a legitimate workflow, so keep it on
  the backlog rather than declaring the omission permanent.
- **Evidence:** master's `test_clean_up` carried verbatim under the
  `NOT PORTED` banner in `src/finmag/sim/sim_test.py`;
  `sim_test.py:1939 test_removing_logger_handlers_allows_to_create_many_simulation_objects`
  (carried, calls `sim.close_logfile()` — cited to D24 in SR1 T1); register D24.

### D25 — the port allows re-adding a previously-removed interaction
- **State:** master asserted `AssertionError` on re-adding a removed
  interaction; the ported `EffectiveField` permits it. Documented-deferred
  2026-07-25; the register asks whether to restore the legacy assertion.
- **Recommend: ACCEPT the relaxation** — the ported behaviour is strictly more
  permissive, no workflow depends on the assertion firing, and reinstating a
  bare `AssertionError` as a public contract would be a step backwards.
- **Evidence:** `src/finmag/sim/sim_test.py::test_remove_interaction2`
  (asserts the ported behaviour under a `BEHAVIOURAL CHANGE` comment);
  register D25.

### D26 — `get_field_as_dolfin_function` raises `ValueError: UFL conditions cannot be evaluated as bool`
- **State:** a **real port bug**, not a design divergence, and it is on the
  **supported** call path: only the `region=` argument is deferred
  (`src/finmag/sim/sim.py:449-458`), and the failing call is the plain
  `get_field_as_dolfin_function('m')` — the failure happens when the returned
  `Function` is point-evaluated. Documented-deferred 2026-07-25; open only as
  *permanence*.
- **Recommend: DEFER (fix in a later slice)** — the honest basis is simply
  that **no selected SR1 workflow calls it**, not that it is a deferred
  surface. **This is the closest call in section A:** if you regard
  `get_field_as_dolfin_function` as part of the supported public API, make it
  FIX NOW instead. *(owner knowledge decisive)*
- **Evidence:** `src/finmag/tests/bugs/test_bug_ndt_file_writing.py:50`
  (the plain call, then `m(point)` evaluation); implementation
  `src/finmag/sim/sim.py:449`; register D26.

### D27 — `Simulation.add()` does not register per-interaction `.ndt` columns
- **State:** master's `add()` registered `E_<name>` / `H_<name>_{x,y,z}`
  columns, so `plot_ndt_columns(columns=['E_Demag', ...])` now raises
  `KeyError`. Pinned `xfail(strict)`. Documented-deferred 2026-07-25.
- **Recommend: DEFER (restore registration in a later slice)** — a genuine
  functional gap in a real legacy workflow (per-interaction energy traces);
  plotting and table output are already section-C later work, so this belongs
  on the backlog rather than being ratified away.
- **Evidence:** `src/finmag/util/plot_helpers_test.py::test_plot_ndt_columns_and_plot_dynamics`
  (`xfail(strict)`); register D27.

### D28 — `ScipyIntegrator.reinit()` does not reset the rhs-eval counter
- **State:** the Sundials backend resets `_n_rhs_evals` on reinit; the SciPy
  backend does not. Master never tested it (legacy `ScipyIntegrator.reinit()`
  was a complete no-op). Documented-deferred 2026-07-25.
- **Recommend: DEFER (reset the counter in a later slice)** — a diagnostic
  inconsistency between backends, not a physics difference; small and worth
  fixing, but nothing selected depends on the counter.
- **Evidence:** `src/finmag/drivers/tests/test_scipy.py` (`xfail(strict)` pin
  plus a regression pinning the current counter-survives-reinit behaviour);
  register D28.

### D29 — anisotropy/Magpar comparison retains ~8% residual (tolerance `8e-2` vs master `5e-7`)
- **State:** **RESOLVED.** The S4 physics investigation returned **DRIFT**, and
  an adversarial review **UPHELD** it. The ported `UniaxialAnisotropy` field is
  the exact box-method field (agrees with an independent NumPy reimplementation
  to `3.6e-16`), and Magpar's stored `Hani` is the same box projection on
  Magpar's own mesh to file precision (`1.9e-6`). The residual is predicted by
  the two meshes' patch-centroid offsets (correlation 0.9988, slope 0.990) and
  collapses to `6e-16` under uniform `m`. The max at a *coincident* node is
  expected: all 116 coincident nodes are edge/corner feature points with ~10x
  the interior centroid offset. **No masked defect; no fix slice required.**
- **Recommend: ACCEPT as mesh-regeneration drift** — the `8e-2` tolerance
  stands as an explicit mesh-drift allowance on a probe-at-saved-coordinates
  comparison. Optional post-SR1 hardening (not required): assemble on the
  imported Magpar `.femsh` mesh to remove the drift term instead of tolerating
  it — a new test, not a repair.
- **Evidence:** [`2026-07-27-d29-verdict.md`](2026-07-27-d29-verdict.md) §6
  (recommended disposition text, quoted above in condensed form); probe
  `dev/dolfinx/probe_d29_anis_magpar.py`; test
  `src/finmag/tests/comparison/anisotropy/test_anis_magpar.py`; register D29.

### D31 — `.npy` `save_field`/`save_m` surface has partial test coverage
- **State:** **partially discharged.** Commit `f7517688` (SR1 S1a) added a
  live, passing witness for the *scheduled* path
  (`schedule('save_m', every=...)` → `Simulation.save_m` → `FieldSaver`).
  Still uncovered: **direct** (non-scheduled) `save_field`/`save_m` calls —
  `incremental=`/`overwrite=`/named-interaction saves (e.g. `'Demag'`). The
  three carried master tests remain `xfail(strict)` on an unrelated blocker:
  their `setup_class` fails on `df.BoxMesh` (legacy dolfin absent) before any
  test body runs. The surface itself is implemented and reachable — this is a
  missing witness, not a broken feature.
- **Recommend: DEFER (partially discharged)** — record the partial discharge,
  and keep "write coordinate-aware direct-call witnesses honouring D14, and
  unblock the three carried tests' `df.BoxMesh` setup" as a named backlog item.
- **Evidence:** witness `src/finmag/tests/test_restart_output.py:690`
  `test_schedule_save_m_every`; carried tests `test_save_field`, `test_save_m`,
  `test_save_field_scheduled` in `src/finmag/sim/sim_test.py`; **live
  implementation `src/finmag/sim/sim.py:639-665`** — note the register's
  `sim_savers.py` pointer is stale: that module is dead legacy code and is not
  imported by `sim.py`. Commits `f7517688`, `fc502ced`; register D31.

### D32 (NEW) — legacy `method=` names rejected by name; default changed to `box-assemble`
- **State:** an **undocumented public-interface change**, split out here from
  M12a so it is separately answerable. Master's `EnergyBase.__init__` defaulted
  to `method="box-matrix-petsc"` and accepted five interchangeable names; the
  port defaults to `"box-assemble"` and raises
  `NotImplementedError("energy method ... is not yet ported to DOLFINx; use
  'box-assemble'")` for the other four. Two consequences differ in kind:
  a script passing an explicit legacy `method=` **raises**; a script relying on
  the **old default raises nothing** and silently gets box assembly (all five
  methods agreed to `1e-13` on master, so the silent path is numerically
  harmless).
- **Recommend: ACCEPT option (a) — keep the raise-by-name.** An explicit
  failure is the criterion-2-consistent way to surface a removed option, and
  the changed default is safe because the values are equal.
  **Defensible alternative (b):** map the four legacy names onto
  `box-assemble` with a `DeprecationWarning` — justified precisely because
  master asserted all five agreed to `1e-13`, so this is not "silently mapping
  to something different". Pick (b) if you have legacy scripts passing
  `method=` that you would rather keep running.
- **Evidence:** `src/finmag/energies/energy_base.py:32-44` (supported vs
  deferred lists; `NotImplementedError` for the four deferred names,
  `ValueError` only for unknown ones; signature default `box-assemble`) vs
  `git show b5015c5a:src/finmag/energies/energy_base.py:52-55` (five methods,
  default `box-matrix-petsc`); gated deferral tests
  `src/finmag/tests/test_energies.py:291` and
  `src/finmag/tests/test_dmi.py:167`; **NEW row — not yet in the register.**

---

## B. Possible full-parity exceptions

### M1 — nsim/Nmag live reference generation
- **State:** SR1 deferment approved 2026-07-23; permanent omission pending.
  Checked-in Nmag reference data validates the selected comparisons today.
- **Recommend: DROP live generation permanently; retain the checked-in data**
  — the nsim stack is obsolete and no longer installable; regenerating it is
  not a capability the port can realistically own. *(owner knowledge decisive)*
- **Evidence:** nmag comparison tests under
  `src/finmag/tests/comparison/` (checked-in data path); register M1.

### M2 — GCR demag solver
- **State:** pending. The surviving legacy Python path appears incomplete and
  its scientific correctness is unclear; FK is the supported, validated solver.
- **Recommend: DROP** — silently mapping GCR to FK would be misleading, and
  porting an implementation whose correctness was never established is not
  justified. Reopenable if a concrete use appears. *(owner knowledge decisive)*
- **Evidence:** `src/finmag/energies/demag/` (FK is the supported path);
  register M2.

### M3 — compiled `Equation`/`terms` backend
- **State:** pending. The Python RHS fallback is much slower; the DOLFINx LLG
  uses its own RHS implementation.
- **Recommend: DEFER as a *performance* backlog item** — and link it to the
  **P1** row below: the std_prob_3 measurement is the first hard number
  suggesting the Python RHS may be a serious throughput regression. Do not drop
  a performance path while an unexplained ~17.6x slowdown is open.
- **Evidence:** `src/finmag/physics/llg.py` (Python RHS); P1 below; register M3.

### M4a — Netgen binary backend
- **State:** ratified 2026-07-25 as deferred for SR1; permanent omission
  pending. The P5.1 necessity probe found **no** selected test or geometry that
  Gmsh plus the `from_geofile` text-subset loader cannot represent or validate.
- **Recommend: DROP permanently** — the probe is the inventory the register
  asked for, and it came back empty. Reopen only if a required geometry
  appears. *(owner knowledge decisive)*
- **Evidence:** SR1 P5.1 probe (2026-07-25, no source change); register M4a.

### M4b — `nmesh_to_dolfin` conversion
- **State:** ratified 2026-07-25 as deferred for SR1; permanent omission
  pending. Its legacy dolfin-XML output cannot be loaded by DOLFINx at all.
- **Recommend: DROP permanently** — the output format is unusable by the
  supported stack and the same P5.1 probe identified no workflow needing it.
  *(owner knowledge decisive)*
- **Evidence:** SR1 P5.1 probe; register M4b.

### M5 — nonlocal `LLG_STT` spin-accumulation model
- **State:** the capability is deferred (correct), but its **only master
  witness files were deleted** in `e7041dc9` on the strength of a deferral-guard
  test that asserts `NotImplementedError` by name rather than reproducing the
  physics. The gap therefore no longer appears as a failing test anywhere —
  a direct conflict with acceptance criterion 4 ("tests for unported
  functionality are KEPT — they are the port's future worklist").
- **Recommend: RESTORE** `src/finmag/tests/zhangli/stt_nonlocal_test.py` and
  `src/finmag/tests/zhangli/zhang_li_test.py` **verbatim from `b5015c5a`** as
  retained never-ported backlog (they fail at collection in the non-gating
  inventory lane — plain pytest behaviour, exactly what criterion 4 prescribes
  for never-ported whole master files). The capability itself stays **DEFER**.
- **Evidence:** `git show b5015c5a:src/finmag/tests/zhangli/stt_nonlocal_test.py`
  (and `zhang_li_test.py`); current cover
  `src/finmag/sim/sim_test.py:1217 test_nonstandard_kernels_are_deferred[sllg|llg_stt]`;
  mapping header in `src/finmag/tests/test_stt.py:34-52`; register M5 / D30.

### M6 — `FixedEnergyDW`
- **State:** pending. No master tests exist, a legacy note calls it broken, and
  its dolfin-XML/treecode workflow is obsolete.
- **Recommend: DROP** — the register's own recommendation; there is no working
  reference behaviour to port to. *(owner knowledge decisive)*
- **Evidence:** `src/finmag/energies/dw_fixed_energy.py` (unported stub);
  register M6.

### M7 — `Demag2D`
- **State:** pending, unported. A specialist low-usage solver path.
- **Recommend: DEFER (keep on the backlog)** — unlike the dead-tooling rows,
  this is real physics. `ThinFilmDemag` and FK cover the selected workflows, so
  it is not an SR1 item, but dropping a solver on "low usage" alone is weaker
  evidence than the M4a/M6 probes provide.
- **Evidence:** register M7; `src/finmag/energies/thin_film_demag.py` (ported
  alternative).

### M8 — Magpar mesh-drift comparison xfails
- **State:** **factually done** (SR1 P5.2). Coordinate-probe / probe-at-saved-
  coordinates replacements landed and the obsolete node-order xfails are
  retired. The residual ~8% is separately recorded and now resolved as D29.
- **Recommend: ACCEPT (close as done)** — no exception is needed; the row
  describes completed work.
- **Evidence:** `src/finmag/tests/comparison/anisotropy/test_anis_magpar.py`,
  `.../demag/test_demag_field.py`,
  `.../exchange/test_exchange_compare_magpar.py`; D29 above; register M8.

### M9 — Paraview/mencoder movie export
- **State:** pending. An external historical renderer wrapper; VTK/XDMF data
  export is available and tested.
- **Recommend: DROP the movie wrapper permanently; retain data export** — the
  register's own recommendation; rendering belongs outside the simulator.
- **Evidence:** `src/finmag/tests/test_restart_output.py:609,618`
  (`test_save_vtk_writes_pvd`, `test_save_field_to_vtk_xdmf` — the retained
  export path); register M9.

### M10 — Mercurial `get_hg_revision_info`
- **State:** pending. Dead pre-Git tooling; the repository is git.
- **Recommend: DROP permanently** — no possible consumer.
- **Evidence:** register M10.

### M11a — historical transposed-Robertson SciPy xfail
- **State:** pending. The failure was attributed to an upstream SciPy/VODE
  regression, not to Finmag physics. **Correction to the register's framing:**
  the port already carries transposed-Robertson pins as strict `xfail`s for
  *both* backends, so this is not an open "should we copy it?" question — the
  behaviour is pinned and passing today.
- **Recommend: DROP** — i.e. close the row with **no tree change**: the
  existing strict-xfail pins stay exactly as they are, and no attempt is made
  to re-derive or "fix" an upstream-attributed failure as if it were a finmag
  expectation.
- **Evidence:** `src/finmag/util/ode/tests/test_sundials_stiff_ode.py:78`
  (`test_robertson_scipy_transposed_fails_with_excess_work`,
  `xfail(strict=True)`) and `:126` (the CVODE mirror); register M11a.

### M11b — historical weak-Krylov-demag tolerance xfail
- **State:** pending. A numerical-tolerance artifact whose relevance under
  DOLFINx has never been established.
- **Recommend: DEFER** — unlike M11a this concerns a *ported, supported*
  feature (Krylov demag scaling). Keep it on the backlog with the register's
  condition attached: re-derive the expectation before deciding whether any
  test remains useful. No tolerance may be copied unexamined.
- **Evidence:** `src/finmag/tests/test_interactions_scale_linearly_with_m.py`;
  register M11b.

### M12a / M12b / M12c — legacy matrix / project / direct energy-assembly *implementations*
*(The `method=` **interface** question is now its own row, D32 above. These
three rows cover only whether the implementations are ever ported back. Each ID
is separately answerable — answer them individually if you disagree on one.)*

- **State (all three):** master's `EnergyBase` exposed five interchangeable
  `method=` choices (`box-assemble`, `box-matrix-numpy`, `box-matrix-petsc`,
  `project`, `direct`) and its own test asserted every supported method agreed
  with the default to `1e-13` — they are numerical routes to the same field,
  not different physics. The port implements `box-assemble` only.
  - **M12a** = the matrix-assembly variants (`box-matrix-numpy`,
    `box-matrix-petsc`): precomputed-operator forms of box assembly.
  - **M12b** = `project`: L2 projection of the energy derivative.
  - **M12c** = `direct`: the direct/pointwise route.
- **Recommend: DROP all three permanently** — box assembly is the validated
  implementation and the alternatives were performance/implementation variants
  producing the same values to `1e-13`. **DROP-permanent implies an S6
  follow-up (no test deletion):** the runtime message says the methods are
  *"not yet ported to DOLFINx"*, which contradicts a permanent drop, so that
  string and the wording of the two gated deferral tests
  (`test_energies.py:291`/`:298`, `test_dmi.py:167`/`:172`) must be reworded
  from "deferred" to "removed". *(owner knowledge decisive)*
- **Evidence:** `src/finmag/energies/energy_base.py:32-44`
  (`_supported_methods = ("box-assemble",)`, `_deferred_methods` = the other
  four) vs `git show b5015c5a:src/finmag/energies/energy_base.py:52-55`;
  master's equivalence assertion is carried at
  `src/finmag/energies/anisotropy_test.py:117`
  `test_anisotropy_field_supported_methods` but is **currently vacuous** — it
  loops over `_supported_methods` minus the default, which is now empty, so it
  asserts nothing and cannot be cited as live evidence of equivalence;
  register M12a/b/c.

### M13 — `batch_task` sweep tooling
- **State:** pending. Untested even on master.
- **Recommend: DROP permanently** — there is no reference behaviour to validate
  a port against. *(owner knowledge decisive)*
- **Evidence:** register M13.

### M14 — OOMMF live reference generation
- **State:** SR1 deferment approved 2026-07-23; permanent omission pending. You
  accepted checked-in OOMMF data as sufficient for the selected comparisons.
- **Recommend: DROP live generation permanently; retain the checked-in data**
  — same reasoning as M1; reproducible regeneration is an external-harness
  project, not a finmag capability. *(owner knowledge decisive)*
- **Evidence:** OOMMF comparison data and tests under
  `src/finmag/tests/comparison/`; register M14.

### M15 — Magpar live reference generation
- **State:** SR1 deferment approved 2026-07-23; permanent omission pending.
  Checked-in Magpar data drives the coordinate/invariant comparisons.
- **Recommend: DROP live generation permanently; retain the coordinate/
  invariant comparisons** — and note D29 established exactly what the
  checked-in data can and cannot resolve (mesh drift is inherent to comparing
  against a saved foreign mesh). *(owner knowledge decisive)*
- **Evidence:** D29 verdict; `src/finmag/tests/comparison/*/test_*_magpar.py`;
  register M15.

### M16 — legacy Heun driver
- **State:** pending. The register explicitly forbids labelling it obsolete
  without your direction. It is `StochasticHeunIntegrator` — the **stochastic**
  integrator, i.e. the driver thermal SLLG needs.
- **Recommend: DEFER (keep on the backlog, tied to thermal SLLG)** — thermal
  SLLG/LLB remains in the full-parity target (register section C), so dropping
  its integrator now would silently pre-decide that row.
- **Evidence:** `git show b5015c5a:src/finmag/tests/test_heun.py` (imports
  `finmag.native.llg.StochasticHeunIntegrator`); master `native/src/llg/heun.{cc,h}`;
  register M16 and section C.

---

## P. New performance row candidate (surfaced during SR1, not yet in the register)

*(The other new candidate, D32, is filed with the behavioural deviations above.)*

### P1 — std_prob_3 FULL-mode runs ~17.6x slower than its own legacy-era header claims
- **State:** the FULL-mode bisection measures **~3161s per relax simulation**
  (mesh build + relax + energy + write, `lfactor=8.0`, `divisions=16`), timed
  from the killed acceptance run's own append-mode artifacts. The workload is
  iteration-bounded at exactly 10 simulations (`bisect(f, 8, 8.5, xtol=0.1)`
  makes exactly 5 calls, each running one vortex + one flower relax), so the
  full bisection is **~31610s (~8.8h)**. The script's own header comment
  describes "the legacy behaviour; ~30 min" for the *whole* bisection. The
  measured gap is therefore **~17.6x** (8.8h vs ~30 min). (The "~35x" figure
  in commit `c02809bf` compares the *doubled timeout ceiling* — 63300s — to
  the comment; that is a budget, not a slowness measurement.) It is genuinely
  ambiguous whether this is a port performance regression or a
  stale/aspirational legacy comment — nobody has run legacy master, and the
  timeout was sized to measured reality, not to the comment.
- **Recommend: DEFER — open as a register row and investigate post-SR1**
  (recommended: profile one `relax()` at these settings and attribute the cost;
  cross-ref **M3**, the compiled-RHS backend, as the leading suspect). SR1 is a
  *correctness* declaration and the example passes; but a ~9-hour example that
  legacy claimed took 30 minutes is a real handoff hazard and should not be
  declared silently. Alternative if you prefer closure: **ACCEPT** as
  documented slowness, recorded in `docs/SUPPORTED.md`.
- **Evidence:** SR1 task-3 report
  (`.superpowers/sdd/2026-07-27-sr1-completion/task-3-report.md`, "Third round:
  std_prob_3 timeout from measured rate"); commit `c02809bf` message and the
  per-entry comment it added in `examples/test_examples.py`.

---

## Summary

**44 open rows** (42 register rows + 2 new candidates, D32 and P1).
Recommended dispositions:

| Verb | Count | Rows |
|---|---|---|
| **ACCEPT** (ratify as permanent contract) | 13 | D6a, D6b, D7, D9, D13, D16a, D16b, D20, D21, D25, D29, D32, M8 |
| **DEFER** (SR1 stands; stays a named parity-backlog item) | 16 | D1, D5, D17, D19, D22, D23, D24, D26, D27, D28, D31, M3, M7, M11b, M16, P1 |
| **DROP** (permanent omission) | 14 | M1, M2, M4a, M4b, M6, M9, M10, M11a, M12a, M12b, M12c, M13, M14, M15 |
| **RESTORE** (reinstate deleted master tests as backlog) | 1 | M5 |
| **FIX NOW** (blocks SR1 declaration) | **0** | — |

Breakdown by register section: A = 23 rows (12 accept, 11 defer — includes the
new D32); B = 20 rows (1 accept, 4 defer, 14 drop, 1 restore); P = 1 row
(1 defer).

**Nothing on this sheet is recommended as SR1-blocking.** If you agree with
every recommendation, SR1 declaration proceeds after one register-update commit
plus the M5 test restoration.

**Rows most worth a second look** (where a different answer is defensible):
- **D26** — a real port bug on a *supported* call path; DEFER holds only
  because no selected SR1 workflow calls it. If
  `get_field_as_dolfin_function` is part of your supported public API, this is
  FIX NOW.
- **D32** — the `method=` interface change: raise-by-name (recommended) vs
  map-with-deprecation-warning. Both are defensible; your legacy scripts
  decide it.
- **D23** — the cheapest fix on the sheet; FIX NOW is reasonable if a
  silently-NaN simulation is unacceptable.
- **D16a/D16b** — accepting D16a as "informational metadata" decides D16b too;
  choosing full-state reconstruction turns both into fix slices.
- **P1** — the only row raising a *performance* question, and the only one
  with no prior register history.

## How to respond

Reply per-row ID with **agree** or an adjustment, e.g.:

```
D1 agree
D6b adjust -> require pointwise normalisation
D23 adjust -> FIX NOW
D26 agree
D32 adjust -> option (b), map legacy names with a deprecation warning
M2 adjust -> DEFER (I still use GCR)
M5 agree
P1 adjust -> ACCEPT, document the runtime
...
```

A bare "agree to all" is sufficient if you accept the sheet as written; rows
you do not mention are taken as agreed. Every **FIX NOW** you assign becomes
its own RED-first slice and blocks the `sr1` tag until it lands.
