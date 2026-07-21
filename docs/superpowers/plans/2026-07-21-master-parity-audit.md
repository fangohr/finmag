# Finmag Master-Parity Audit & Phase-3 Plan

**Date:** 2026-07-21   **Branch audited:** `dolfinx-parity` @ `2ecd5041` (read-only)
**Author:** static audit (no code changed; this document is the only write)

> Scope: definitive gap register for reaching FULL functional parity between the
> DOLFINx port (`dolfinx-parity` HEAD) and the original `master` Finmag, plus a
> prioritised Phase-3 task plan. Two layers: (1) HEAD vs the pixi branch's
> *verified* surface; (2) the pixi verified surface vs original `master` (never
> previously inventoried).

---

## 1. Lineage + method

| Branch | Commit | What it is | Verified surface |
|---|---|---|---|
| `master` | `b5015c5a` | ORIGINAL Python 2 / dolfin-2017 Finmag. Functionality yardstick. | Not runnable here (Py2). Static inventory only. |
| `pixi` | `ba928093` | Py3 + pixi + FEniCS-2019 transition (M1-M3). | Defined by `dev/bin/verify-python3-*` gates + CI workflows; skip/xfail recorded in `transition-notes.org`. |
| `dolfinx-parity` | `2ecd5041` (HEAD) | DOLFINx port, Phase-1 core + Tier-1 + Tasks 10-16. Serial-only. | `dev/bin/verify-dolfinx-m5` (`dolfinx-src-*` gates). Deferrals in `dev/dolfinx/porting_map.md`, pinned in `src/finmag/tests/test_deferred_surfaces_dolfinx.py`. |

**Method.** Cross-branch `git show`/`git ls-tree`/`git diff` only. Evidence sources:
`porting_map.md` "Near-Term Gaps"; `test_deferred_surfaces_dolfinx.py`;
`grep -rn NotImplementedError src/finmag`; the pixi-tip gate scripts
(`verify-python3-core-suite`, `-m2`, `-m3`, `-minimal-suite`, `barmini-suite` in
`pixi.toml`) and CI workflows (`.github/workflows/python3-m1|m2|m3|core-suite.yml`);
the DOLFINx gate `verify-dolfinx-m5` + `pixi.toml` `dolfinx-src-*` tasks; the M1-M3
skip/xfail rationales in `transition-notes.org` (M1-M3 sections byte-identical at
pixi tip and HEAD).

**Key framing fact (settles most of Layer 2).** The pixi M3 gate result recorded in
`transition-notes.org` (~L2036-2080) is **493 passed / 21 skipped / 5 xfailed** and
"covers all 101 files listed by `verify-python3-core-suite`"; the M1 dolfin-2017 core
gate is **502 passed / 14 skipped / 3 xfailed**. So the pixi transition *did* revive
essentially the whole master test-file set. The master-vs-pixi gap (Layer 2) is
therefore narrow and well-characterised — mostly external-tool skips and a handful of
deliberate drops. The large gap to close is Layer 1 (HEAD vs pixi).

**What HEAD's `verify-dolfinx-m5` actually verifies (the ported surface):** sliceable
imports, `Field`, common energies (Exchange, static+time Zeeman family, uniaxial &
cubic anisotropy, DMI), `EffectiveField`, deterministic `LLG`, SciPy driver, core
`Simulation`/`sim_with` incl. `run_until`/`relax`/`hysteresis`, FK demag (compiled
array BEM), restart v2 + NDT/VTK/XDMF **write**, scheduler, variable material
parameters (Ms/A/K1/K2/axis/D, per-node alpha), region accounting. **Serial only.**
Note: there is **no CI workflow** gating `verify-dolfinx-m5` (no `dolfinx-m5.yml`); it
runs manually. That is itself a gap (see Task 28).

---

## 2. Layer 1 register — pixi verified surface that HEAD defers/lacks

Legend — **State**: `by-name` = raises `NotImplementedError`/`ImportError` naming the
feature; `AttributeError` = surface removed outright; `raw-import` = still imports legacy
`dolfin` at module scope (raw `ModuleNotFoundError`); `absent` = no port at all.
**Oracle?** = can a legacy oracle fixture (`run-legacy-oracle`) or analytic result
validate it. **Size**: S ≤ ~1 slice, M ~2-3, L = multi-slice/native build.

| # | Capability | Legacy home (file) | State at HEAD | Oracle? | Size | Notes |
|---|---|---|---|---|---|---|
| L1 | Native Sundials/CVODE integrator (legacy DEFAULT backend) | `drivers/sundials_integrator.py`; `native/src/cvode`,`native/src/sundials` | `by-name` ImportError on `backend="sundials"` (`llg_integrator.py`) | analytic + pixi ref | L | SciPy driver is the working default; native path needs dolfin-free rebuild like `bem_arrays`. LLG `sundials_rhs`/`_jtimes`/`_psetup`/`_psolve` stubbed by name (`physics/llg.py`). |
| L2 | STT — Slonczewski & Zhang-Li | `physics/llg_stt.py`; `LLG.set_stt`/`set_zhangli` | `by-name` (`sim.py`, `llg.py`) | oracle (`zhang_li_test`,`stt_nonlocal_test`) | M | Deterministic; no thermal/native dep needed. `do_slonczewski`/`do_zhangli` flags dropped. |
| L3 | Stochastic SLLG + LLB | `physics/llb/{sllg,llb,material,exchange,anisotropy}.py`; `native/src/llb` | `by-name` `kernel="sllg"`; classes `raw-import` | oracle (`sllg_test`,`llb_test`) | L | Needs `finmag.native.llb` dolfin-free rebuild. |
| L4 | Normal modes + eigen-analysis | `normal_modes/`, `sim/normal_mode_sim.py`, `example/normal_modes/disk.py` | `absent` from core; helpers `raw-import` | analytic/eigen ref | L | eigensolvers/eigenproblems test-covered on pixi. FFT/PSD (L12) is the analysis half. |
| L5 | NEB (nudged elastic band) | `physics/neb.py`,`neb_cartesian.py`,`neb_cartesian_modified.py`; `native/src/neb` | `absent`; needs `native.neb` | oracle (`neb_test` helpers) | L | Full tangent computation needs native module (also a pixi gap — see Layer 2). |
| L6 | GCR demag / Demag2D / MacroGeometry / LU solver option | `energies/demag/{gcr_demag,fk_demag_2d}.py`,`solver_base.py` | `by-name` (`fk_demag.py`, `demag/__init__.py`, `gcr_demag.py`) | partial | M | GCR intentionally dropped even on pixi (see Layer 2 M2). Demag2D/MacroGeometry/LU are curated by-name. |
| L7 | PBC + treecode demag | `energies/demag/{fk_demag_pbc,demag_treecode,treecode_bem}.py`; `util/pbc2d.py`; `native/src/{fast_sum_lib,treecode_bem}` | `by-name`; `parallel`/PBC deferred (`sim.py`) | oracle (`demag_pbc_test`) | L | Needs `finmag.native.treecode_bem` build (also a pixi skip — Layer 2 M4). |
| L8 | Mesh tooling — Netgen/Gmsh generators | `util/meshes.py`, `util/mesh_templates.py`, `util/nmesh_to_dolfin.py` | `raw-import`/`by-name` (`meshes.py:1284/1295/1579`; `mesh_templates.py:49`) | shape/volume | L | Foundational for examples & realistic-geometry demag/validation. DOLFINx uses its own mesh; needs gmsh→dolfinx bridge. |
| L9 | HDF5 `dolfinh5tools` save/read + XDMF read-back | `field.py` `save_hdf5`/`load`; `sim_savers.py` | `by-name` (`field.py`) | round-trip | M | XDMF/VTK are write-only at HEAD. |
| L10 | Region-restricted field OUTPUT (submesh) | `sim_savers.py`, `sim_helpers.py` | `absent` (region *accounting* IS ported, Task 16) | round-trip | M | `mark_regions`/`compute_energy(dx=)`/`m_average_in_region` ported; per-region *field export* is not. |
| L11 | Plotting / visualization helpers | `util/visualization.py`,`visualization_impl.py`,`plot.py`,`plot_helpers.py`; `Field.plot_with_dolfin` | `by-name` (`field.py:458`; `visualization_impl.py:552`) | none (GUI) | M | GUI/render tests already skip on pixi; movie export needs Paraview/mencoder (Layer 2 M8). |
| L12 | FFT / PSD normal-mode analysis | `util/fft.py`, `util/dispersion.py` | `by-name` (`fft.py:334`); `dispersion` untested even on master | analytic | M | Pairs with L4. |
| L13 | Spherical-coordinate conversion | `Field.get_spherical`/`set_spherical` | `by-name` (`field.py`) | analytic | S | |
| L14 | Point probing / point-measure arithmetic | `field.py` probe API; `util/point_contacts.py` | `by-name` (`field.py`) | analytic | S | `point_contacts` untested even on master. |
| L15 | Skyrmion/vortex initialisers, `skyrmion_number`, `mesh_info`, `length_scales` on sim | `sim/magnetisation_patterns.py`, `sim/sim_helpers.py`, `sim/sim_details.py` | `AttributeError` (removed outright, Task 9) | analytic | M | `util/length_scales.py` itself is tested on pixi; only the `Simulation` convenience wrappers were dropped. Netgen-dependent subcases (L8). |
| L16 | External comparison harnesses OOMMF/Magpar/Nmag | `util/oommf/`, `util/magpar.py`, `tests/comparison/`, `tests/nmag/`, `tests/oommf/` | `absent`/`raw-import` (`magpar.py:195`) | cross-code | L | Data checked in; harness code unported. OOMMF now a conda dep on pixi; Magpar/Nmag are xfail/checked-in on pixi (Layer 2). |
| L17 | MPI multi-rank stepping (ODE state) | `physics/llg.py`, `drivers/scipy_integrator.py`, `sim.py` | serial-only guard raises when `comm.size>1` (`llg.py`) | 2-rank probe | L | Field/energy/EffectiveField already have 2-rank probes; only the ODE-state gather/scatter is serial. Cross-cutting. |
| L18 | Remaining optional energies: ThinFilmDemag, FixedEnergyDW | `energies/thin_film_demag.py`, `energies/dw_fixed_energy.py` | `raw-import` ModuleNotFoundError (documented gap) | oracle (`thin_film_demag_test`) | S | Only classes still importing legacy `dolfin` at module scope. `dw_fixed_energy` untested even on master. |
| L19 | Packaging — installable `finmag` (no `PYTHONPATH=src`) | (none on master; `setup2.py`, `distcp.py` legacy) | `absent` — no `pyproject.toml` at HEAD (Task 17 paused) | import check | M | Confirmed: no `pyproject.toml`/`setup.py` at repo root at HEAD. |
| L20 | Small deferred long-tail | various | `by-name` | targeted | S each | `DMI(dmi_type='D2D')` (`dmi.py:88`); spatially varying cubic axes `u1`/`u2` (`cubic_anisotropy.py:280`); legacy string `Expression` coefficients (`energy_base.py:225/259/288`, `field.py from_expression`); `TimeZeemanPython` scalar-envelope+vector-`time_fun` (`zeeman.py:415`); matrix/project/direct energy assembly methods (`energy_base.py:42`, `anisotropy.py:42`). |
| L21 | Batch/parameter-sweep tooling | `util/batch_task.py` | `by-name`/`absent` (`batch_task.py:165`); untested even on master | none | S | Convenience; low value. |

**Layer 1 count: 21 register lines** (several bundle multiple named surfaces). Every
row is a capability the pixi FEniCS-2019 gate exercises (passing, or skipped only for a
missing external tool) that HEAD does not yet provide.

---

## 3. Layer 2 register — master capabilities the pixi transition never revived/verified

Classification: **(a)** skipped for missing external tool; **(b)** deliberate
scope-narrowing; **(c)** genuinely never revived (interesting); **(d)** historical/dead.
Evidence line numbers reference `git show ba928093:transition-notes.org`.

| # | Capability / test | Master home | Pixi status (class + evidence) | Revival cost | Recommend |
|---|---|---|---|---|---|
| M1 | nsim / Nmag comparison toolchain | `tests/nmag/*`, `util/nmesh_to_dolfin.py` | **(a)+(b)** nsim intentionally outside pixi; rely on checked-in Nmag reference data (L2295-2298; omission list L249). Reference-data tests pass; live nsim never installed. | L (would need nsim) | **ACCEPT-DROP** — nsim is Py2/unmaintained; keep checked-in reference data as the oracle. |
| M2 | GCR demag solver | `energies/demag/gcr_demag.py`; `test_compute_scalar_potential_gcr` | **(c)** deliberately NOT ported: "surviving Python-side demag code does not retain a complete and clearly correct GCR formulation"; mapping `Demag('GCR')`→FK judged "scientifically misleading"; `"GCR"` removed from `KNOWN_SOLVERS`; raises `NotImplementedError` (L1704-1728, 2116-2117). | M (re-derive) | **ACCEPT-DROP** — FK demag is the correct/validated path; GCR was already unsound. Formalise the drop. |
| M3 | Compiled `Equation`/`terms` backend | `physics/equation.py`; `dolfin.compile_extension_module` | **(c)** removed; replaced by pure-Python fallback ~107x slower; native reference data checked in (L2124-2159). Tests pass via fallback. | M | **PORT (perf)** on DOLFINx — the *capability* is verified, only speed regressed. Low urgency; the DOLFINx port already uses its own LLG RHS. |
| M4 | treecode/PBC demag (`native.treecode_bem`) | `energies/demag/{demag_treecode,treecode_bem}.py`; `demag_pbc_test.py` | **(a)/(c)** collected-skip in M3 (module-level `skipif`); "treecode_bem remains unavailable on the pixi native build … follow-up is to build and activate … a NumPy fallback is not the next step" (L456-459, 2070-2074, 2299-2303). | L (native build) | **PORT** — same slice as Layer-1 L7. Native rebuild required. |
| M5 | Full NEB tangent (`native.neb`) | `physics/neb*.py`; `physics/tests/neb/neb_test.py` | **(a)** helper-only `neb_test` runs; "full NEB tangent computations still require `finmag.native.neb`" (L2212-2214, 2291-2294). | L (native build) | **PORT** if NEB is in-scope for research; else ACCEPT-DROP. Same as Layer-1 L5. |
| M6 | Stochastic SLLG (`native.llb`) | `physics/llb/sllg.py`; `test_zhangli_sllg` | **(a)/(b)** skips when `native.llb` absent; deterministic Zhang-Li kept "without forcing the stochastic sllg native module into the first pixi build" (L2207-2211). | L (native build) | **PORT** if thermal dynamics in-scope. Same as Layer-1 L3. |
| M7 | Magpar mesh-drift comparisons | `tests/comparison/anisotropy/test_anis_magpar.py::test_against_magpar`, `comparison/demag/test_demag_field.py::test_using_magpar` | **(c)** `xfail`: regenerated Netgen node array no longer matches saved Magpar reference node ordering/shape (L2011-2025, 2063-2069). Physics fine; fixture drift. | M | **ACCEPT-DROP** (as xfail) or regenerate fixtures. Not a functional gap. |
| M8 | Paraview/mencoder movie export | `dev/bin/{pvd2avi,export_paraview_animation}`; visualization anim tests | **(a)** movie subcases skip when Paraview/`sh`/mencoder absent; non-movie VTK path runs (L972-982). | M | **ACCEPT-DROP** — external renderer; VTK/XDMF files (write) already produced for external viewing. |
| M9 | Mercurial `get_hg_revision_info` + test | `util/helpers.py`; `helpers_test.py::test_get_hg_revision_info` | **(d)** skipped (no `hg` in image); "historical pre-Git coverage"; TODO to remove the function entirely once Py3 stable (L980-988, 1581-1584). | S | **ACCEPT-DROP** — dead pre-Git tooling; delete. |
| M10 | MPI multi-rank execution | `tests/test_sim_parallel.py` | **(a)** `skipif(mpi_size<2)`; runs when ≥2 ranks (L1507-1509, 1596-1598). Functionality present on pixi, not exercised in serial CI. | — | Covered on pixi; on DOLFINx it is a genuine gap (Layer-1 L17). |
| M11 | Transposed-Robertson SciPy stiff case | `util/ode/tests/test_sundials_stiff_ode.py` | **(c)** `xfail(strict=True)` — SciPy 1.17.1 vode/BDF regression, not a Finmag bug (L2272-2273). | S (upstream) | **ACCEPT-DROP** — upstream SciPy; document. |
| M12 | weak-Krylov-demag tolerance case | `tests/test_interactions_scale_linearly_with_m.py` | **(c)** historical `xfail` "kept intentionally … reviewed by a human later" (L1049-1052, 2020-2024). | S | **ACCEPT-DROP / review** — tolerance expectation, not correctness. |
| M13 | OOMMF cross-code comparisons | `tests/oommf/*`, `tests/comparison/*_oommf.py`, `util/oommf/` | **(a) resolved on pixi** — OOMMF added as conda-forge dep; run as real regression tests (L806-808, 1526-1529). | L (harness port) | On DOLFINx the harness is unported (Layer-1 L16); OOMMF binary itself is available. |

**Layer 2 finding:** master functionality was overwhelmingly revived by pixi. Only
**M2 (GCR)** and **M3 (compiled Equation backend)** are true "never-revived (c)" drops of
distinct capability; **M4/M5/M6** are native-build deferrals that are *also* Layer-1
work; the rest are external-tool skips (a) or dead code (d). No previously-unknown master
capability was discovered missing beyond what porting_map already tracks. `UNVERIFIED`:
exact per-test pass lists inside the 493-pass M3 run were not individually re-run (Py2/
container), but the aggregate counts and the file-level skip/xfail rationales are
authoritative from `transition-notes.org`.

---

## 4. Prioritised Phase-3 plan (Tasks 18+)

Continues the numbering from the existing `2026-07-21-dolfinx-full-parity.md` (Tasks
13-16, 18, 19 done; Task 17 packaging paused). Ordered in dependency order. Each task names the
Layer-1/Layer-2 register items it closes. Validation prefers: legacy oracle fixture
(`dev/bin/run-legacy-oracle`, ≤3 uses budget) → analytic result → pixi-tip reference run
(`barmini-suite`).

**Task 18 — Mesh tooling bridge (Netgen/Gmsh → DOLFINx).** *Foundational; unblocks
realistic-geometry examples, demag validation, skyrmion/region workflows.*
Files: `util/meshes.py`, `util/mesh_templates.py`, new gmsh→`dolfinx.mesh` reader,
`util/nmesh_to_dolfin.py`. Size: L. Validation: mesh shape/volume vs analytic + pixi
`meshes_test`/`mesh_templates_test` reference. Closes: **L8** (and unblocks L15, L16, M1
mesh path). Recommend FIRST — many downstream tasks need non-trivial meshes.

**Task 19 — Remaining optional energies.** Files: `energies/thin_film_demag.py`,
`energies/dw_fixed_energy.py`, `energies/__init__.py`. Size: S. Validation: legacy oracle
(`thin_film_demag_test`) + by-name for the truly-unported. Closes: **L18** (last
`raw-import` energy classes → curated behaviour).

**Task 20 — Native Sundials/CVODE backend (dolfin-free rebuild).** Files: `native/src/{cvode,sundials}`
rebuilt following the `bem_arrays` pattern (`-DFINMAG_NO_DOLFIN`), `drivers/sundials_integrator.py`,
`physics/llg.py` `sundials_*` hooks (already stubbed by name). Size: L. Validation:
analytic macrospin + pixi `barmini-smoke` (Sundials is the pixi default) + `sundials_ode`
tests. Closes: **L1**. Restores the legacy DEFAULT integrator; also settles the native-build
story that Task 17 packaging must wrap.

**Task 21 — Packaging (resume paused Task 17).** Recommend HERE — *after* Task 20 fixes
the native-extension build story (`bem_arrays` + `cvode`), so `pyproject.toml`'s build
hook wraps the final native set once. Files: new `pyproject.toml`, `pixi.toml`,
`native/Makefile` hook, `verify-dolfinx-m5`. Size: M. Validation: fresh editable install +
`verify-dolfinx-m5` green; keep `run-legacy-oracle` functional. Closes: **L19**.

**Task 22 — STT (Slonczewski & Zhang-Li).** Files: `physics/llg_stt.py`, `physics/llg.py`
(`set_stt`/`set_zhangli`), `sim.py`. Size: M. Validation: legacy oracle (`zhang_li_test`,
`stt_nonlocal_test`, `slonczewski/`). Deterministic — no thermal/native dep. Closes: **L2**.

**Task 23 — PBC + treecode/GCR demag native slice.** Files: `native/src/{fast_sum_lib,treecode_bem}`
(dolfin-free rebuild), `energies/demag/{fk_demag_pbc,demag_treecode,treecode_bem}.py`,
`util/pbc2d.py`, `Simulation` `pbc`/`parallel` paths. Size: L. Validation: legacy oracle
(`demag_pbc_test`) + FK cross-check. Closes: **L6 (PBC/treecode part), L7, M4**. (GCR itself
folded into the accept-drop task.)

**Task 24 — Thermal/SLLG + LLB.** Files: `physics/llb/*`, `native/src/llb` (dolfin-free
rebuild), `energies/random_thermal.py`, `sim.py` `kernel="sllg"`. Size: L. Validation:
legacy oracle (`sllg_test`, `llb_test`) — stochastic, so distributional/seeded checks.
Closes: **L3, M6**.

**Task 25 — Normal modes, NEB, FFT/PSD analysis.** Files: `normal_modes/`,
`sim/normal_mode_sim.py`, `physics/neb*.py`, `native/src/neb`, `util/fft.py`,
`util/dispersion.py`, `example/normal_modes/disk.py`. Size: L (largest). Validation:
analytic eigenfrequencies + legacy oracle (`neb_test`, `eigenproblems_test`). Closes:
**L4, L5, L12, M5**. Depends on Task 18 (meshes) and Task 20 (native build story).

**Task 26 — I/O, output & convenience surfaces.** Files: `field.py` (`save_hdf5`/HDF5,
XDMF read-back, `get_spherical`, point probing), `sim_savers.py` (region-restricted field
output), `util/visualization*.py`, `util/plot*.py`, `sim/magnetisation_patterns.py`,
`sim/sim_helpers.py`/`sim_details.py` (skyrmion initialisers, `skyrmion_number`,
`mesh_info` wrappers). Size: L. Validation: round-trip fixtures + analytic (skyrmion
number, spherical). Closes: **L9, L10, L11, L13, L14, L15**.

**Task 27 — External comparison harnesses.** Files: `util/oommf/`, `util/magpar.py`,
`tests/comparison/`, `tests/oommf/`, `tests/nmag/` re-wiring to DOLFINx. Size: L.
Validation: cross-code (OOMMF binary available via conda) + checked-in Nmag/Magpar data.
Closes: **L16, M13** (and re-homes M1/M7 as data-only comparisons).

**Task 28 — MPI multi-rank stepping.** Files: `physics/llg.py`, `drivers/scipy_integrator.py`,
`sim.py` — generalise the `xxx` serial ODE-state contract to a collective coordinate-sorted
gather/scatter (the design note for this already exists in `porting_map.md`). Size: L
(cross-cutting; do late so every ported interaction already has 2-rank probes to reuse).
Validation: 2-rank `test_sim_parallel` equivalence vs serial. Closes: **L17, M10**. Also:
add a `dolfinx-m5.yml` CI workflow gating `verify-dolfinx-m5` (currently ungated).

**Task 29 — FORMALLY-ACCEPT-DROP decision (single user-decision task).** Present these for
explicit user sign-off; on approval, replace by-name stubs with documented "accepted
exception" notes and close the parity checklist:
- **M2** GCR demag (unsound legacy formulation; FK is the correct path) — Layer-1 L6 GCR part.
- **M3** compiled `Equation` backend (Python fallback works; port only if perf demanded).
- **M9** Mercurial `get_hg_revision_info` (dead pre-Git tooling — delete).
- **M8** Paraview/mencoder movie export (external renderer).
- **M7** Magpar mesh-drift xfails; **M11** transposed-Robertson SciPy upstream xfail;
  **M12** weak-Krylov-demag tolerance xfail.
- **L20** long-tail: `DMI(dmi_type='D2D')` (not in public `dmi_type` contract), varying
  cubic axes, legacy string `Expression` coefficients, `TimeZeemanPython` vector-`time_fun`,
  matrix/project/direct energy-assembly methods.
- **L21** `batch_task` sweep tooling (untested even on master).
Size: S (docs + stub relabelling, no new physics).

**Recommended ordering rationale.** 18→19 remove foundational blockers cheaply; 20→21
settle the native build + packaging together; 22-25 are the physics feature families in
increasing native-dependency weight; 26-27 are IO/comparison breadth; 28 is the hard
cross-cutting parallel work done once everything else has 2-rank probes; 29 closes the
checklist. Task 17 (packaging) is folded in as Task 21 — deliberately *after* Task 20 so
packaging wraps a settled native-extension set rather than being redone.

---

## 5. Counts summary

- **Total register items: 34** — Layer 1: 21 (L1-L21); Layer 2: 13 (M1-M13).
- **Portable (recommend PORT): ~22** — Layer 1 L1-L19 (19 portable capability groups; L20/L21 are drop candidates) + Layer 2 M3/M4/M5/M6 that are distinct native-build ports (the remaining M-items either equal Layer-1 rows or are accept-drops).
- **Accept-drop candidates: ~11** — M1, M2, M7, M8, M9, M11, M12 + L20 (5 named sub-items) + L21, all consolidated into **Task 29**.
- **Estimated slices to full parity: ~12** — Tasks 18-29 (11 build/port tasks + 1 decision task). Weight concentrated in 5 L-sized native/physics slices (Tasks 20, 23, 24, 25, 28); the rest are S/M.
- **Biggest single risk / long-pole:** native-extension rebuilds (CVODE, llb, neb,
  treecode_bem) — each needs the dolfin-free `-DFINMAG_NO_DOLFIN` treatment already proven
  once for `bem_arrays`; and MPI multi-rank ODE state (Task 28), the only cross-cutting
  redesign. `UNVERIFIED`: native rebuild feasibility for `cvode`/`llb`/`neb`/`treecode_bem`
  is assumed-analogous to `bem_arrays` but not yet demonstrated.
