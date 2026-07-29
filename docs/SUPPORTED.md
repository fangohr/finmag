# Finmag (DOLFINx) — what is supported in Support Release 1 (SR1)

**Release:** SR1 — serial deterministic DOLFINx finmag. Declared 2026-07-28 on
branch `dolfinx-parity` (tag `sr1`).

**This is not full parity with original `master` (`b5015c5a`).** SR1 is a
scientifically testable, deterministic, **serial** micromagnetic simulator with
a substantial legacy-compatible `Simulation`/`sim_with` interface, plus an
explicit list of everything that is still unavailable. Full parity with
original `master` remains the final target.

This file is the **user-facing** distillation. The engineering documents behind
it are:

| Document | Role |
|---|---|
| [`capability-status.md`](capability-status.md) | canonical per-capability status and evidence (C01–C22) |
| [`acceptance-register.md`](acceptance-register.md) | the sole owner-decision ledger (D-rows, M-rows, P1) |
| [`archive/HANDOVER.md`](archive/HANDOVER.md) | retired entry point (historical; see [docs/README.md](README.md)) |
| [`archive/master-pixi-parity-manifest.md`](archive/master-pixi-parity-manifest.md) | file-by-file master↔port test/example map |

Register rows are cited inline below as **D<n>** / **M<n>** / **P1**; each is a
recorded owner decision, not an informal note.

---

## 1. Install and first simulation

```bash
pixi install -e dolfinx                          # solve/sync the environment
pixi run -e dolfinx dolfinx-install-editable     # editable install of src/finmag
pixi run -e dolfinx dolfinx-native-build         # bem_arrays.so, sundials.so, treecode
dev/bin/verify-dolfinx-m5                        # 33 focused gates; expect 33/33
```

```python
import finmag

sim = finmag.example.barmini()      # 3x3x10 nm Py bar, m0 = (1, 0, 1)
sim.run_until(1e-10)                # deterministic time integration
print(sim.m_average)                # volume-averaged magnetisation
print(sim.total_energy())           # J
sim.relax()                         # relax to equilibrium
```

`import finmag` must resolve inside this checkout; `pixi run -e dolfinx
dolfinx-provenance-check` asserts it.

---

## 2. Supported API

Everything in this section has an automated witness in the 33-gate verifier.
"Supported" means the surface works and is tested — it does not promise
bit-identity with original `master` where a register row records a deviation.

### 2.1 Construction

| Surface | Notes |
|---|---|
| `finmag.Simulation(mesh, Ms, unit_length=1, name="unnamed", kernel="llg", integrator_backend="sundials", pbc=None, average=False, parallel=False)` | `mesh` is a `dolfinx.mesh.Mesh`. `pbc` and `parallel=True` raise by name (D19). |
| `finmag.sim_with(mesh, Ms, m_init, alpha=0.5, unit_length=1, integrator_backend="sundials", A=None, K1=None, K1_axis=None, H_ext=None, demag_solver="FK", demag_solver_type=None, nx=None, ny=None, spacing_x=None, spacing_y=None, demag_solver_params=None, D=None, name="unnamed", pbc=None, sim_class=Simulation)` | one-call construction of the common setup |
| `finmag.example.bar()`, `barmini()`, `nanowire()` | ported; meshes built with `dolfinx.mesh.create_box` |
| `finmag.set_logging_level(level)` | dolfin-free (`finmag.util.logging_helpers`) |

**Default integrator backend is native Sundials/CVODE** (D8), matching legacy.
`integrator_backend="scipy"` is a fully supported explicit opt-in.

### 2.2 Magnetisation and fields

| Surface | Notes |
|---|---|
| `sim.m` (get/set), `sim.set_m(value, normalise=True)` | flat **component-blocked** `xxx` array in coordinate order — never raw backend dofs |
| `sim.m_field`, `sim.m_average`, `sim.dmdt`, `sim.dmdt_max` | |
| `sim.Ms`, `sim.alpha`, `sim.gamma`, `sim.do_precession`, `sim.pins` | scalars, callables, arrays and `Field`s; callable pin masks work (raw mesh units) |
| `sim.probe_field(field_type, pts, region=None)`, `sim.probe_field_along_line(field_type, pt_start, pt_end, N=100, region=None)` | `pts` is array-like with last axis of dimension 3, in **mesh coordinates** (`unit_length` is not applied); returns a `numpy.ma.masked_array`. `region=` is deferred by name. **Points exactly on an outer mesh face mis-resolve (D22)** — probe interior points |
| `sim.skyrmion_number()`, `sim.skyrmion_number_density_function()` | |

`m_init` accepting a **NaN** silently (instead of `ValueError`) is a known
divergence, **D23**.

### 2.3 `Field`

Construction/coercion: `from_callable`, `from_constant`, `from_function`
(interpolates across compatible space mismatches, **D7**), `from_field`,
`from_array`, `from_sequence`, `from_generic_vector` (`dolfinx.la.Vector` /
`PETSc.Vec`), `coerce_scalar_field`.

Inspection/ops: `average`, `normalise`, `allclose`, `is_constant`,
`as_constant`, `value_dim`, `mesh`, `coords_and_values`, `probe`/`__call__`,
`get_spherical`, `cross`, `dot`, `__mul__`/`__rmul__`/`__truediv__`,
`get_ordered_numpy_array_xxx`/`_xyz` and their setters.

I/O: `save_pvd`, `save_xdmf`, `save_hdf5`/`Field.from_hdf5` (a single
self-describing `.h5`, coordinate-keyed so it survives FEM reordering).

Not available: `Field.__add__` (deferred, no consumer); `from_expression`
(string `Expression`s do not exist in DOLFINx, **D15**); XDMF/VTK **function
readback** (a DOLFINx 0.10 absence, not a missing port).

### 2.4 Energy interactions

`from finmag.energies import ...`

| Class | Status |
|---|---|
| `Exchange` | supported (oracle-validated) |
| `UniaxialAnisotropy` | supported, incl. spatially varying `K1`/`K2`/axis — see the axis contract in §4.1 |
| `CubicAnisotropy` | supported, incl. spatially varying `u1`/`u2` axes (greenfield: legacy crashed on any varying axis) |
| `DMI`, `DMI_interfacial` | supported; `dmi_type='D2D'` raises by name (**D1**) |
| `Zeeman`, `TimeZeeman`, `DiscreteTimeZeeman`, `OscillatingZeeman`, `TimeZeemanPython`, `DipolarField` | supported; `TimeZeemanPython` rejects a *vector-valued* `time_fun` by name (**D5**) |
| `Demag` (Fredkin–Koehler) | supported, oracle-validated; the default |
| `Demag` treecode / `MacroGeometry` | supported (analytic + cross-method); tile **pitch must be strictly larger than the mesh extent** (**D17**) |
| `ThinFilmDemag` | supported |
| `Demag2D`, `FixedEnergyDW`, GCR | **not available** (M7, M6, M2) |

Composition, per-region parameters and `sim.mark_regions(...)` +
`m_average_in_region` / `region_measure` / `save_m_in_region` are supported.

**`method=` is `"box-assemble"` only.** Master's `box-matrix-numpy`,
`box-matrix-petsc`, `project` and `direct` are **removed** — not deferred —
and raise `NotImplementedError` by name (**D32**, **M12a/M12b/M12c**). See
§4.3.

### 2.5 Drivers, dynamics and scheduling

| Surface | Notes |
|---|---|
| `sim.run_until(t)`, `sim.advance_time(t)`, `sim.t`, `sim.reset_time(t)` | reading `sim.t` no longer constructs an integrator (**D13**) |
| `sim.relax(...)` | ported (legacy `sim_relax.py` bound as legacy did) |
| `sim.hysteresis(...)`, `sim.hysteresis_loop(...)` | ported; each stage re-relaxes to its own equilibrium (**D4** investigation) |
| `sim.create_integrator`, `sim.reinit_integrator`, `sim.set_tol`, `sim.driver` | both backends construct, advance, reset/reinitialise |
| `sim.set_stt(...)` (Slonczewski), `sim.set_zhangli(...)`, `sim.toggle_stt(...)` | **local** STT modes; configuring both raises `ValueError` naming both (**D11**, **D18**) |
| `sim.schedule(fn_or_name, ...)`, `unschedule`, `clear_schedule` | shortcuts: `save_ndt`, `save_averages`, `save_field`, `save_m`, `save_vtk`, `save_restart_data`, `eta`/`ETA` |
| `sim.save_averages()` / `.ndt` via `Tablewriter` | |
| `sim.save_field(...)`, `sim.save_m(...)` (`.npy`) | coordinate/value tables, **not** raw dofs (**D14**); direct-call witness is partial (**D31**) |
| `sim.save_vtk(...)`, `sim.save_field_to_vtk(...)` | VTK/XDMF **write** (readback unavailable) |
| `sim.save_restart_data(...)`, `sim.restart(...)` | coordinate-aware **v2** only; v1 raw-dof archives are rejected (**D12**). See §4.2 |
| `finmag.util.plot_helpers`: `plot_dynamics`, `plot_hysteresis_loop`, `plot_ndt_columns`, `surface_2d`, `boxplot` | dolfin-free, headless. `surface_3d` is importable but **not executable** (matplotlib 3.11 API break). `plot_ndt_columns` on per-interaction columns raises `KeyError` (**D27**) |
| `finmag.util.meshes`: Gmsh/OCC generators, templates, caching, `from_geofile` text subset, `mesh_info`, `mesh_size`, `length_scales` | Netgen **binary** backend and `nmesh_to_dolfin` are dropped (M4a/M4b) |

**Caveat — `maxh` density differs from legacy netgen 5.3 (D34):** meshes are
now generated by Gmsh, which maps `maxh` directly onto `Mesh.MeshSizeMax`; on
measured geometry the same `maxh` on the same `.geo` produces ~22% fewer nodes
than the archived netgen-5.3-era mesh (1202/4960 vs 1537/6886 nodes/tets), and
the same generator-version density drift is independently reported for
netgen 6.2 on the pixi branch ([issue #56](https://github.com/fangohr/finmag/issues/56)).
If you need netgen-5.3-comparable density, reduce `maxh` by roughly 10-20% and
confirm the result with `mesh_info()` rather than assuming `maxh` parity
across generator versions — see D34.

---

## 3. Validation classes and limits

Read tolerances as *what was actually checked*, not as a general accuracy claim.

| Class | Meaning |
|---|---|
| **oracle** | compared against the frozen legacy FEniCS-2019 finmag at `ba928093`, actually executed |
| **analytic** | compared against a closed-form physics result |
| **cross-method** | two independent implementations in this tree agree |
| **external-reference-data** | compared against checked-in Nmag/OOMMF/Magpar output (the live harnesses are **not** resurrected — M1/M14/M15) |
| **regression** | pinned against a recorded value of this port |
| **end-to-end** | a workflow completes and its own embedded check passes |

### 3.1 Per-family validation

| Family | Strongest class | Tolerance / limit |
|---|---|---|
| Exchange | oracle + external (Nmag) | Nmag exchange field `2e-14`; Magpar coordinate-probe `9e-8` at exact node coincidence |
| Uniaxial / cubic anisotropy | oracle + external (Nmag, OOMMF, Magpar) | Nmag 1D anisotropy; **Magpar anisotropy retains ~8% residual, tolerance `8e-2`** — see D29 below |
| DMI | oracle + analytic | includes unit-length invariance |
| Zeeman family | oracle + analytic | `DiscreteTimeZeeman` energy tracks the current field (**D3** fixed) |
| FK demag | **oracle** | plus the analytic uniformly-magnetised-sphere check |
| Treecode / MacroGeometry demag | analytic + cross-method | 3×-tiled 20 nm cube vs a directly-meshed 60×20×20 nm bar: 0.08% in `Hx`, 0.011% in `Hz` (legacy bounds 1%/2%) |
| Spatially varying parameters, regions | oracle + regression | |
| Local STT (Slonczewski, Zhang-Li) | oracle + analytic + cross-backend | each mode alone; **nonlocal `LLG_STT` is absent** (M5) |
| LLG dynamics, pins, composition | oracle + analytic + regression | |
| SciPy & Sundials drivers | regression + end-to-end | construct, advance, reset/reinit, schedule, save, restart on both |
| Restart | end-to-end | immediate *and* uninterrupted-vs-restarted trajectory equality |
| Point/topology utilities | analytic | but see D22 (outer-face probes) |

### 3.2 The muMAG standard problem #4 quantitative anchor

`examples/std_prob_4/test_std_prob_4.py` carries the one **quantitative**
external-reference anchor in the example lane, against the published Martinez
`m_averages_ref_martinez.txt` field-1 trajectory.

- Compared quantities: the `<m_x> = 0` crossing time and `<m_y>` at that
  crossing.
- Reference values (recomputed at runtime from the checked-in file):
  `t_ref = 0.138255 ns`, `<m_y>_ref = 0.733963`.
- **Tolerances are derived a priori** from the mesh and the reference
  trajectory *before* any simulation output was compared — no number was
  chosen to make an observed result pass. With exchange length
  `l_ex = sqrt(2A/(mu0 Ms^2)) = 5.6858 nm` and representative element diameter
  `h = 5.830 nm`, the P1 discretisation error scale is
  `eps = (1/8)(h/l_ex)^2 = 0.1314`, giving
  **`tol_t = eps / |d<m_x>/dt| = 8.089 ps`** and `tol_my = 0.1634`.
- `8.089 ps` is **4.9× tighter** than the coarse 0.10–0.18 ns switching window
  the reduced lane uses, so the anchor adds real constraint.
- A pre-committed fallback is recorded in the test: a deviation between
  8.09 ps and 12.13 ps adopts the already-derived sharp 3D constant
  `C = 3/16`; a deviation beyond 12.13 ps is a **physics finding** to report,
  not to accommodate.
- **Status of the anchor at declaration:** *pre-flight PASS only.* Against the
  10 ps-sampled partial trace this port produced on 2026-07-28
  (0 → 0.55 ns, containing the crossing) the interpolated crossing was
  `t = 0.13751 ns`, i.e. `|dt| = 0.75 ps` — **11× inside tolerance** — and
  `<m_y> = 0.7299` (40× inside). **It has not yet been exercised on a
  completed full-resolution 2 ns trajectory**; see §3.3.
- The `<m_y>` bound is honestly weak: it is ~100× looser than the deviation
  observed in pre-flight and will not detect sub-20% `<m_y>` regressions.

### 3.3 The FULL example lane — 15 of 17, two deferred by owner decision

`FINMAG_EXAMPLE_FULL=1 pytest -q examples/test_examples.py` runs all seventeen
converted examples at their real workloads (the reduced/fast lane is 14 of
them at reduced workload and is what `dev/bin/verify-dolfinx-m5` records).

At declaration, **15 of the 17 entries are witnessed green at full workload**:

- **14 entries** in the 2026-07-28 attempt-3 acceptance run (9.2 h), including
  the two entries that had failed every previous FULL attempt —
  `cubic_anisotropy/hysteresis.py` and `std_prob_3/run.py` (the complete
  10-simulation bisection). Log:
  `/home/sam/.claude/jobs/6b8f36a7/tmp/full-lane-final.log`.
- **`cubic_anisotropy/sim.py`** separately measured green in 3551 s
  (~59.2 min) on 2026-07-27 (`/tmp/cubic_anisotropy_sim_run2.log`), after the
  SR1 T1 fix that registered `save_m` as a scheduler shortcut.

The remaining **two** entries — `std_prob_4/test_std_prob_4.py` (the full 2 ns
trace) and `magnetic_grain/suess_2001.py` (full three-field physics) — are
**deferred by owner decision (2026-07-28)** to a re-run scheduled *after* the
post-SR1 performance work
([`plans/2026-07-28-post-sr1-performance.md`](plans/2026-07-28-post-sr1-performance.md)),
because those two entries are the primary beneficiaries of the planned
speedups (their budgeted timeouts are 38400 s and 21600 s respectively).

This is a **sequencing decision, not an evidence downgrade**: the deferral is
recorded here, in `capability-status.md` (C15 stays *partial
(declaration-qualified)*, not PASS), and in `archive/HANDOVER.md`. Completing
those two entries is named, scheduled follow-up work.

Supporting evidence chain for the FULL lane (all wrapper timeouts were rescaled
to *measured* rates, never to make a run pass):

| Commit | What it established |
|---|---|
| `20a0a22c` | scaled FULL-lane timeouts to measured reality; added `_clean_ignored_outputs()` (`git clean -fdX` scoped per example) so the lane stops being one-shot per checkout |
| `71dfc064` | fixed the timeouts the *first* acceptance attempt refuted (`std_prob_4` 5400 s → 38400 s from a linear extrapolation of a measured `t = 2.1e-10 s` in 1800 s) |
| `c02809bf` | derived `std_prob_3`'s budget from a hard measurement — one relax at `lfactor=8.0`, `divisions=16` takes 3161 s, and the bisection is iteration-bounded at exactly 10 simulations |

**Artifact policy outcome:** the FULL lane regenerates
`examples/std_prob_3/doc_table.rst`. The regenerated file differed from the
tracked one by a **1-character RST column shift with identical content**, so it
was restored (`git checkout`) rather than committed: a formatting-only
regeneration artifact is not a source change and must not enter the
declaration commit.

### 3.4 D29 — the ~8% anisotropy/Magpar residual

`tests/comparison/anisotropy/test_anis_magpar.py` compares the ported uniaxial
anisotropy field against checked-in Magpar output by coordinate probe. Its
tolerance is `8e-2`, loosened from master's `5e-7`, and the measured maximum
disagreement is ~5.1e-2 — while the *sibling* exchange/Magpar comparison holds
`9e-8` at exact node coincidence.

The reason the two differ is **Netgen mesh-regeneration drift**: master's
`5e-7` assumed the saved Magpar nodes and the finmag mesh were the *same* mesh.
They no longer are — the mesh is regenerated by a different Netgen than the one
that produced the reference — so the comparison is now "two different
discretisations of the same physics", and the residual is the discretisation
difference, largest where the anisotropy field varies fastest. Because the
maximum happened to land on an exactly-coincident node, this was **not** taken
on trust: an adversarial physics review was run specifically to rule out a
masked anisotropy-field defect. Verdict **DRIFT, UPHELD** — see
[`archive/specs/2026-07-27-d29-verdict.md`](archive/specs/2026-07-27-d29-verdict.md).
Owner disposition: ACCEPT as quantified mesh-regeneration drift; no fix slice
required.

Practical reading: **do not treat `8e-2` as finmag's anisotropy accuracy.** It
is the resolution limit of comparing against a foreign saved mesh. The
anisotropy field itself is oracle- and analytic-validated at machine-precision
tolerances elsewhere.

### 3.5 Performance — read this before planning a run (**P1**)

The port is **substantially slower than legacy-era expectations**. `std_prob_3`
in FULL mode measures ~3161 s per relax simulation and ~8.8 h for its full
bisection, against the script's own header comment claiming "~30 min" — a
**~17.6× gap**.

Root cause identified 2026-07-28: the port re-does `fem.form(...)` +
`assemble_vector(...)` on **every field evaluation**
(`src/finmag/energies/energy_base.py:183`), where legacy's default
`box-matrix-petsc` assembled the field operator **once** and then applied it
(`b5015c5a:src/finmag/energies/energy_base.py:214-222`); the numpy RHS versus
the compiled `Equation` backend (**M3**) compounds it.

The fix is planned, not done:
[`plans/2026-07-28-post-sr1-performance.md`](plans/2026-07-28-post-sr1-performance.md)
rebuilds the precompute as an *internal* optimisation of `box-assemble`
semantics — it does **not** resurrect the removed legacy method names (D32
stands).

---

## 4. Documented interface contracts (owner-ratified)

These are behaviours where the port deliberately differs from original
`master`, each ratified by the repository owner on 2026-07-28.

### 4.1 Anisotropy axis normalisation (**D6a**, **D6b**)

- **A constant `UniaxialAnisotropy` axis is normalised.** On master a non-unit
  constant axis silently rescaled `K1` (the energy density is
  `K1 * (1 - (m·u)^2)`, so `|u| != 1` changes the effective constant). The port
  normalises it, which restores the documented cosine contract. If you were
  relying on a non-unit axis to scale `K1`, scale `K1` explicitly.
- **A spatially varying axis (callable / `Field` / `Function`) is used exactly
  as supplied — no pointwise normalisation, no orthogonalisation.** This
  matches master bit-for-bit. If your varying axis is not unit-norm at every
  node, the effective anisotropy constant varies with it. Normalise it yourself
  if that is not what you want.

The same "used as supplied" rule governs `CubicAnisotropy`'s `u1`/`u2`
(`u3 = u1 × u2` is formed per node from the raw axes).

### 4.2 Restart contract (**D16a**, **D16b**)

`sim.save_restart_data()` / `sim.restart()` reconstruct **magnetisation and
simulation time**, coordinate-keyed (v2). They do **not** reapply or validate
material and interaction metadata: the restarting script is responsible for
rebuilding the same `Simulation` (same `Ms`, `A`, `K1`, interactions, `alpha`)
before calling `restart`. This matches the practical legacy reconstruction
pattern and is the accepted SR1 restart contract.

Consequently the metadata an archive carries is **informational only**, and for
spatially varying material fields it is a **lossy scalar summary** — it cannot
reconstruct a varying `Ms`, `alpha` or interaction coefficient. Do not read it
back as state. Legacy **v1 raw-dof** archives are rejected outright (**D12**);
there is no v1 compatibility path.

### 4.3 Energy `method=` names are removed, not deferred (**D32**, **M12a–c**)

Master's `EnergyBase.__init__` defaulted to `method="box-matrix-petsc"` and
accepted five interchangeable names. This port:

- defaults to **`"box-assemble"`**, and
- raises `NotImplementedError` by name for `box-matrix-numpy`,
  `box-matrix-petsc`, `project` and `direct`.

The four names are **permanently removed** (M12a/M12b/M12c ratified DROP), not
awaiting a port. They were implementation/performance variants that agreed with
box assembly to `1e-13` on master, so the *values* a script gets are unchanged;
only the option is gone. A script passing an explicit legacy `method=` fails
loudly — deliberately, so a removed option is surfaced rather than silently
remapped. A script relying on master's old default simply gets box assembly.

(The planned performance work reintroduces assemble-once *mechanics*
internally; it does not reintroduce the *names*.)

---

## 5. Waiting to be ported ("Later")

Part of the full-parity target. Not available in SR1; not dropped.

| Capability | Register |
|---|---|
| Thermal SLLG / LLB (and its `StochasticHeunIntegrator`) | C16, M16 |
| Normal modes, eigensolvers, ringdown, FFT/PSD | C17 |
| Legacy NEB / path methods | C18 |
| General MPI time stepping | C19 |
| `Simulation(pbc='1d'/'2d')` function-space periodicity — *blocked*: no `dolfinx_mpc` in the environment, and DOLFINx 0.10 has no `constrained_domain` | **D19** |
| Nonlocal `LLG_STT` spin-accumulation model | **M5** |
| `DMI(dmi_type='D2D')` | **D1** |
| `TimeZeemanPython` vector-valued `time_fun` | **D5** |
| `Demag2D` | **M7** |
| Treecode factory selector via `sim_with(demag_solver='Treecode')` | C07 |
| Coincident-node BEM fix (touching macro-geometry tiles) | **D17** |
| XDMF/VTK **function readback**; `Field.__add__`; region/submesh *field* output (`get_submesh`, `get_field_as_dolfin_function(region=...)`) | C04, C12 |
| ~~`Simulation.shutdown()` / `instances_delete_all_others()` / `close_logfile()`~~ **FIXED 2026-07-28 (CI T3, `f12cf995`)** | **D24** |
| ~~`get_field_as_dolfin_function` UFL-bool crash~~ **FIXED 2026-07-28 (CI T4, `a86fb294`)** | **D26** |
| Per-interaction `E_<name>` / `H_<name>_*` `.ndt` columns | **D27** |
| SciPy `reinit()` rhs-eval counter reset | **D28** |
| `set_m` NaN guard | **D23** |
| Direct-call `.npy` `save_field`/`save_m` witnesses | **D31** |
| Outer-face point-location fix | **D22** |
| OOMMF comparison suite (`tests/oommf/*`), `finmag.util.oommf` | C20 |
| `surface_3d` matplotlib-3.11 fix; a PyVista adapter | C12 |
| Skyrmion initialiser family (`initialise_skyrmions` fails at call time on a 2D/3D coordinate gap) | C21 |
| Compiled `Equation`/`terms` RHS backend | **M3**, **P1** |
| Weak-Krylov-demag tolerance expectation, re-derived | **M11b** |
| `finmag.util.helpers` as an importable module (still `import dolfin` at module scope); submodule spelling `from finmag.example.normal_modes import disk` | C02 |

## 6. Not now — dropped permanently

Owner-ratified permanent drops (2026-07-28). Each is reopenable only by a new
owner decision.

| Capability | Register | Why |
|---|---|---|
| Live `nsim`/Nmag reference generation | **M1** | obsolete stack, no longer installable; checked-in data retained |
| Live OOMMF reference generation | **M14** | external-harness project, not a finmag capability; checked-in data retained |
| Live Magpar reference generation | **M15** | same; coordinate/invariant comparisons retained |
| GCR demag solver | **M2** | correctness never established; mapping it to FK would be misleading |
| Netgen **binary** backend | **M4a** | the P5.1 necessity probe found no selected geometry needing it |
| `nmesh_to_dolfin` conversion | **M4b** | its dolfin-XML output cannot be loaded by DOLFINx |
| `FixedEnergyDW` | **M6** | no master tests, a legacy note calls it broken |
| Paraview/mencoder movie export | **M9** | rendering belongs outside the simulator; data export retained |
| Mercurial `get_hg_revision_info` | **M10** | dead pre-Git tooling |
| `batch_task` sweep tooling | **M13** | untested even on master |
| Energy `method=` names `box-matrix-numpy`/`box-matrix-petsc`/`project`/`direct` | **M12a–c**, **D32** | implementation variants agreeing to `1e-13`; see §4.3 |
| Legacy string `Expression`/`UserExpression` inputs | **D15** | no DOLFINx equivalent; use vectorised Python callables |
| Historical transposed-Robertson SciPy xfail | **M11a** | attributed to an upstream SciPy/VODE regression, not finmag physics |

## 7. How an unavailable surface fails

Two distinct, deliberate mechanisms. **A raw `ModuleNotFoundError: No module
named 'dolfin'` is a tracked port bug, never an instruction to install
anything.**

**(a) By-name `NotImplementedError` at runtime.** Every deferred `Simulation`
surface raises through one helper, naming the feature:

```python
>>> sim.snapshot()
NotImplementedError: snapshot: VTK output (use save_vtk) is not part of the
core DOLFINx Simulation port (deferred, see dev/dolfinx/porting_map.md).
```

Same for `pbc=`, `parallel=True`, `sllg`/`llg_stt` kernels,
`run_normal_modes_computation`, `get_submesh`, `render_scene`, `plot_mesh`,
region-restricted saves/probes, non-FK `demag_solver=`, touching macro-geometry
pitch, `DMI(dmi_type='D2D')`, removed energy `method=` names, `FixedEnergyDW`,
and the top-level `finmag.NormalModeSimulation` /
`finmag.normal_mode_simulation` / `example.sphere_inside_airbox` /
`example.normal_modes` attributes. `src/finmag/tests/test_deferred_surfaces.py`
is the aggregate gate that keeps these curated.

Two known holes in (a), both recorded: the *submodule* spelling
`from finmag.example.normal_modes import disk` and importing
`finmag.util.helpers` as a module still reach legacy `dolfin` directly (C02).

**(b) Strict `xfail`, for master tests carried but not ported, or for whole
master files never ported at all.** Where a master test function had no
covering port, it was transcribed **verbatim** into the port's file under a
`NOT PORTED` banner and marked `@pytest.mark.not_ported` plus
`@pytest.mark.xfail(reason="not ported: <feature> (register <row>)",
strict=True)` (register **D30**). Strict means: the day someone ports the
feature, the test **XPASSes and the gate fails**, forcing the marker off.
Nothing can quietly stay "carried" once it works. **2026-07-28 update (CI T5,
register D33):** the same mechanism now also covers whole never-ported master
files. Previously such a file's module-level `import dolfin` (or
`finmag.util.helpers`, `finmag.native.*`, `finmag.util.oommf`, etc.) failed
collection outright, so `dev/bin/inventory-dolfinx-suite` reported it under
`errors=`. Each of the 53 remaining files got that same import wrapped
`try/except ImportError` and every test that still observably fails afterward
gained the `not_ported`/strict-`xfail` pair above; a handful of tests that
turned out not to need the unavailable import at all were left unmarked
(9 recovered as live coverage — see register **D33** for the list). Master's
own pre-existing `xfail`/`skipif` markers (including environment-conditional
runtime `pytest.skip()` calls) are, as before, left untouched and continue to
govern their own tests' outcome.

**Its failure population (now zero) was the parity backlog, deliberately
kept — it was never a CI verdict.** The verdict is
`dev/bin/verify-dolfinx-m5`'s 33 focused gates. Run the sweep:

```bash
dev/bin/inventory-dolfinx-suite      # or: pixi run -e dolfinx dolfinx-src-suite-inventory
```

Tally at declaration (2026-07-27, pre-CI-nail-down) — **SUPERSEDED**, see
below:

```
INVENTORY: passed=754 failed=5 errors=55 skipped=25 xfailed=47
```

**Green baseline (2026-07-28, CI T1-T5, CONFIRMED by CI T7):** the 5 real
failures fixed, `fileio_test.py` ported, `Simulation` teardown ported
(D24), the D26 point-evaluation bug fixed, and the 53-file conversion wave
above (D33). Re-confirmed on the CI-nail-down-closure tree, logs
`/home/sam/.claude/jobs/6b8f36a7/tmp/t7-verify.log` and `t7-inventory.log`:

```
INVENTORY: passed=769 failed=0 errors=0 skipped=46 xfailed=271
```

(First full-tree run after the 53-file wave measured `errors=2` — two
`setup_module()`-time `import dolfin` failures in
`tests/nmag/exchange_3d/test_dynamics_3D.py` and
`tests/nmag/spinwaves/test_spinwaves.py`, invisible to a `--collect-only`
pass and so absent from the wave's collect-only-derived file list; converted
the same way, commit `828f45d6`. `xfailed=271` reflects that fix, and the
CI T7 re-run reproduced this exact tally with zero deltas.)

`errors=0`/`failed=0` is now the expected, enforced shape (the
`test-python.yml` weekly/on-demand CI job greps the inventory line for it —
see [`testing.md`](testing.md) for all six CI workflows, including
`test-fast.yml` (the push/PR fast gate), `test-python.yml` (the
weekly-on-default-branch plus on-demand full-suite inventory) and
`test-slow.yml` (the on-demand-only heavy FULL example lane)); every
remaining failure/skip is either a master-governed
skip/xfail or the D22 outer-face caveat. The exhaustive per-file
classification (now two mechanisms only) was in an earlier verification
report from a since-removed working session; the authoritative in-repo
record is
[`archive/HANDOVER.md`](archive/HANDOVER.md) under "Canonical test
paths, the legacy oracle lane, and the inventory lane" plus git history.

---

## 8. Reporting a problem

Two questions decide where something belongs:

1. Does the surface raise a **feature-naming** error? Then it is a *documented*
   deferral — check §5/§6 and the register row.
2. Does it raise a raw `ModuleNotFoundError: dolfin`, an `AttributeError`, or
   produce a wrong number? Then it is a **bug**: the port's contract is that
   every unavailable surface fails by feature name.

Include the register row if you can find one; `acceptance-register.md` is the
sole decision ledger and every deviation above is traceable to a row in it.

[Claude Opus 4.8]
