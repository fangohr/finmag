# Master/Pixi parity manifest (P0.1 — complete; P0.2 — baseline evidence complete)

**Frozen source references:** original `master` `b5015c5a47c244eea1476d8e718286137dca0c83`; Python-3/FEniCS-2019 oracle `pixi` `ba9280934e188d7f3800e7b9865e70a9422f7687`; DOLFINx worktree `dolfinx-parity` `f1a1344c423e74687ddf820c1fafc056a6271fe1` (commit authored 2026-07-22; baseline reviewed 2026-07-23).

This is the P0.1 inventory, not a claim that a mapped file has every legacy
assertion ported. **Mapped-current** means a named current test/gate witnesses
the stated contract; it never means test-by-test assertion equivalence.
Disposition and scope remain canonical in [the acceptance register](acceptance-register.md)
and [capability status](capability-status.md); this file records the evidence
map required before P0.2 baseline execution.

**Historical-audit namespace warning:** this manifest is the current evidence
map. Historical `master`/`pixi` audit labels, task numbers, and conclusions are
not a second status namespace and do not override the canonical documents.

## Vocabulary

The only capability states in this document are:

- **PASS** — the exact positive contract and listed current gate passed at the
  frozen DOLFINx baseline.
- **KNOWN-DEFECT** — implementation violates the positive contract and a
  regression demonstrates or pins that defect.
- **MISSING** — the positive contract is absent. A test that merely expects
  `NotImplementedError` is not positive evidence.
- **UNRUN** — a positive current gate exists but has no recorded result at this
  frozen baseline.

Appendix classifications are intentionally different: **mapped-current**,
**needs-translation**, **later**, **not-now**, **external-reference-data**,
**obsolete-review**, and **needs-decision**. “Later”, “not-now”, and
“obsolete-review” never authorise deletion: an obsolete-review item still needs
an explicit owner disposition if it is to leave the final-parity backlog.

Evidence terms follow `capability-status.md`: **oracle** means output from the
frozen pixi code, **external-reference-data** means checked OOMMF/Nmag/Magpar
or published data, **analytic** means a closed-form result, and
**cross-method/cross-backend** compares independent implementations. A
**regression** pins a named contract, **round-trip/end-to-end** exercises a
workflow, and **qualitative** shows plausibility without establishing parity.

## Owner-Now atomic capability ledger

| ID | Legacy interface and scientific/behavioural invariant | Current DOLFINx gate(s) | Evidence type | State |
|---|---|---|---|---|
| N01 | Editable install resolves `finmag` from this checkout | `dolfinx-install-editable`, `dolfinx-provenance-check` | install/provenance | PASS |
| N02 | `import finmag` succeeds without legacy `dolfin` | `dolfinx-src-import` | import regression | PASS |
| N03 | Everyday top-level exports and `finmag.example.barmini()` work | no positive current gate | import/API regression | MISSING |
| N04 | Deferred top-level families fail by their feature name, not raw `dolfin` import | `dolfinx-src-import-pytest` | error-boundary regression | KNOWN-DEFECT |
| N05 | `Simulation` constructs deterministic serial state and advances with `run_until` | `dolfinx-src-simulation-pytest`, `dolfinx-src-core-smoke` | regression | PASS |
| N06 | `Simulation.relax` converges under its documented stopping contract | `dolfinx-src-simulation-pytest` | workflow regression | PASS |
| N07 | `sim_with` wires common Exchange/Zeeman/uniaxial interactions | `dolfinx-src-simulation-pytest` | API regression | PASS |
| N08 | `sim_with` accepts selected MacroGeometry arguments and uses the FK path | no positive current gate | API/cross-method | MISSING |
| N09 | Field constants, arrays and Python callables preserve coordinate/value ordering | `dolfinx-src-field-pytest`, `dolfinx-src-ordering-pytest` | analytic/regression | PASS |
| N10 | Field arithmetic (`cross`, `dot`, coercion) is available | no positive current gate | API/analytic | MISSING |
| N11 | `Field.from_generic_vector` converts the supported vector contract | no positive current gate | conversion regression | MISSING |
| N12 | Field plotting produces the documented selected output | no positive current gate | rendering regression | MISSING |
| N13 | Exchange field/energy scaling is correct | `dolfinx-src-energies-pytest` | analytic/regression | PASS |
| N14 | Exchange static-equilibrium workflow matches its legacy contract | no positive current gate | physical equilibrium | MISSING |
| N15 | Static Zeeman field/energy scaling is correct | `dolfinx-src-energies-pytest` | analytic/regression | PASS |
| N16 | Time-Zeeman applied field updates its effective field | `dolfinx-src-timezeeman-pytest` | oracle/regression | PASS |
| N17 | `DiscreteTimeZeeman.compute_energy()` uses the active interval field | `dolfinx-src-timezeeman-pytest` | defect regression | KNOWN-DEFECT |
| N18 | Hysteresis independently re-relaxes at every field stage | `dolfinx-src-timezeeman-pytest` | defect regression | KNOWN-DEFECT |
| N19 | Uniaxial anisotropy field/energy is correct | `dolfinx-src-energies-pytest` | analytic/regression | PASS |
| N20 | Cubic anisotropy constant-coefficient field/energy is correct | `dolfinx-src-cubicanis-pytest` | analytic/oracle | PASS |
| N21 | Spatial K2 uses the correct derivative/indexing | `dolfinx-src-varparams-pytest` | analytic plus oracle-divergence regression (the correct port intentionally differs from the legacy indexing bug) | PASS |
| N22 | Spatially varying cubic axes are supported | no positive current gate | analytic | MISSING |
| N23 | DMI bulk/interfacial field conventions are correct | `dolfinx-src-dmi-pytest` | analytic/oracle | PASS |
| N24 | FK demag field/energy is correct | `dolfinx-src-demag-pytest` | analytic/oracle | PASS |
| N25 | Direct MacroGeometry/treecode periodic demag is correct | `dolfinx-src-treecode-pytest` | analytic/cross-method | PASS |
| N26 | Slonczewski torque works when configured alone | `dolfinx-src-stt-pytest` | analytic/oracle | PASS |
| N27 | Zhang-Li torque works when configured alone | `dolfinx-src-stt-pytest` | analytic/oracle | PASS |
| N28 | A conflicting Slonczewski/Zhang-Li configuration raises a clear error | `dolfinx-src-stt-pytest` | defect regression | KNOWN-DEFECT |
| N29 | SciPy advances and reinitialises at the documented time/state | `dolfinx-src-scipy-pytest` | driver regression | PASS |
| N30 | SciPy restarted trajectory agrees with uninterrupted trajectory | no positive current gate | restart equivalence | MISSING |
| N31 | Sundials advances at the documented time/state | `dolfinx-src-sundials-pytest` | driver/cross-backend | PASS |
| N32 | Sundials `reset_time` reinitialises without SciPy-only state | `dolfinx-src-sundials-pytest` | defect regression | KNOWN-DEFECT |
| N33 | Sundials restart reconstructs the saved v2 state | `dolfinx-src-restart-output-pytest` | defect regression | KNOWN-DEFECT |
| N34 | Restart records/restores the selected integrator provenance | no positive current gate | restart metadata | MISSING |
| N35 | Public `Simulation` default backend is Sundials | `dolfinx-src-simulation-pytest` | API/default regression | KNOWN-DEFECT |
| N36 | v2 restart stores coordinate/value state and restores magnetisation/time | `dolfinx-src-restart-output-pytest` | round-trip | PASS |
| N37 | v1 raw-dof restart is rejected clearly | `dolfinx-src-restart-output-pytest` | compatibility regression | PASS |
| N38 | NDT tables are written through the scheduled workflow | `dolfinx-src-restart-output-pytest` | end-to-end | PASS |
| N39 | Coordinate/value `.npy` field snapshots are written | `dolfinx-src-restart-output-pytest` | end-to-end | PASS |
| N40 | VTK/XDMF field files are written | `dolfinx-src-restart-output-pytest` | end-to-end | PASS |
| N41 | HDF5 field readback works | no positive current gate | round-trip | MISSING |
| N42 | Region/submesh field output works | no positive current gate | output regression | MISSING |
| N43 | Callable pin masks constrain the documented nodes | no positive current gate | dynamics regression | MISSING |
| N44 | `Simulation(pbc=...)` supplies periodic function-space stepping | no positive current gate | MPI/PBC regression | MISSING |
| N45 | Selected magnetisation initialisers work | no positive current gate | geometry/dynamics | MISSING |
| N46 | `Simulation.probe_field` and `probe_field_along_line` work | no positive current gate | point-evaluation regression | MISSING |
| N47 | `LLG.M` and `LLG.M_average` report full magnetisation | no positive current gate | API regression | MISSING |
| N48 | Selected simulation helper convenience methods work | no positive current gate | API regression | MISSING |
| N49 | Netgen binary backend necessity is established by a selected geometry probe | no positive current gate | evidence probe | MISSING |
| N50 | Checked Nmag reference-data comparison for a selected interaction passes | `dolfinx-src-examples-pytest` (`exchange_demag/test_exchange_demag.py`) | external-reference-data | PASS |
| N51 | Remaining selected checked-data Nmag comparisons have DOLFINx witnesses | no positive current gate | external-reference-data | MISSING |
| N52 | Fast converted examples execute their stated checks | `dolfinx-src-examples-pytest` | end-to-end | PASS |
| N53 | Full converted-example lane executes its stated checks | `FINMAG_EXAMPLE_FULL=1 pytest -q examples/test_examples_dolfinx.py` | end-to-end | KNOWN-DEFECT |
| N54 | Standard problem 4 has quantitative mesh-matched/convergence acceptance | no positive current gate | quantitative physics | MISSING |
| N55 | Common box/sphere/cylinder/ellipsoid/disk/ring/cone/CSG mesh generators preserve geometry and caching contracts | `dolfinx-src-meshes-pytest` | analytic/geometry regression | PASS |
| N56 | Spatially varying `Ms`, A, K, D, axis and alpha plus region energies/averages work | `dolfinx-src-varparams-pytest` | oracle/regression | PASS |
| N57 | An invalid pin index raises `ValueError` without retaining stale pins | `dolfinx-src-llg-pytest` | error-contract regression | PASS |
| N58 | Vectorised Python callables replace supported legacy string expressions and strings fail clearly | `dolfinx-src-field-pytest`, `dolfinx-src-varparams-pytest` | API/error-contract regression | PASS |
| N59 | Scheduler `at`, `every`, `at_end`, callable and clear/unschedule contracts work | `dolfinx-src-restart-output-pytest`, fast scheduling example | end-to-end/regression | PASS |
| N60 | `length_scales` and `mesh_info` selected convenience surfaces work | no positive current gate | analytic/API regression | MISSING |
| N61 | Selected logging, instance and shutdown helpers import and work | no positive current gate | API regression | MISSING |
| N62 | Selected checked-data OOMMF comparisons have DOLFINx witnesses | no positive current gate | external-reference-data | MISSING |
| N63 | Selected checked-data Magpar comparisons have DOLFINx witnesses | no positive current gate | external-reference-data | MISSING |
| N64 | Selected profiling helpers work without legacy imports | no positive current gate | API/performance regression | MISSING |
| N65 | `Field.probe` and `Field.__call__` evaluate a field at valid points | `dolfinx-src-io-utils-pytest` | analytic/regression | PASS |
| N66 | `Field.get_spherical`, `Simulation.skyrmion_number` and density use the documented conventions | `dolfinx-src-io-utils-pytest` | analytic/regression | PASS |
| N67 | `DipolarField` supplies its documented time-dependent field | `dolfinx-src-timezeeman-pytest` | analytic/regression | PASS |
| N68 | VTK/XDMF files can be read back into the documented Field contract | no positive current gate | round-trip | MISSING |
| N69 | Each selected legacy pytest file is ported in a feature-family slice, or explicitly linked to a Later/Not-now decision | no positive current gate | migration sequencing | MISSING |

### Post-baseline lifecycle resolution (not a rewrite of frozen P0 states)

The N30/N32/N33/N34/N35/N17 states above describe the frozen P0 baseline at
`f1a1344c`. The subsequent P1/P2 commits resolve their named lifecycle/defect
contracts; this is a post-baseline resolution record, not a rewrite of those
frozen rows.

| Frozen row | Resolution commit | Post-baseline evidence |
|---|---|---|
| N30 SciPy restart trajectory | `17f24413` | immediate and uninterrupted-vs-restarted SciPy trajectory checks pass |
| N32 Sundials reset | `3f4ed4ea` | backend-neutral factory passes `t0`; SciPy positional slots preserved; continuity regression passes |
| N33 Sundials restart | `17f24413` | immediate and uninterrupted-vs-restarted Sundials trajectory checks pass |
| N34 restart provenance | `17f24413` | archive records truthful `sim.integrator_backend`; writable `sim.driver` remains synchronized |
| N35 public Sundials default | `81fab481` | `Simulation` and `sim_with` default to native Sundials; regression witnesses native construct/advance and explicit SciPy support; core smoke reports `integrator_backend: sundials` at `t=1e-12`; final clean main aggregate is 32/32 green (`/tmp/finmag-p1-default-m5.log`) |
| N17 `DiscreteTimeZeeman` stale energy (register D3) | `ff906f11` | interval refresh now routes through `set_value()`, re-forming the cached `self.E`, so `compute_energy()` tracks the current field instead of freezing at the setup-time value; measured field bit-invariance (max\|difference\| = 0.0) proves no field value changed. Focused gate 28 -> 32 passed. The committed oracle fixture does NOT numerically discriminate the original defect (~1e-37 J vs `atol=1e-18`); the legacy stale value is now hard-pinned by an explicit divergence pin (`_D3_LEGACY_STALE_ENERGY = -8.042477193189932e-23`) rather than by the fixture. Clean aggregate `dev/bin/verify-dolfinx-m5` exit 0, all 32 steps green (`/tmp/finmag-p24-m5-clean.log`) |

### Post-baseline comparison-coverage resolution (SR1 P5.2 test conversion)

The N50/N51/N62/N63 external-reference-data rows above describe the frozen P0
baseline at `f1a1344c`. The P5.2 test-conversion phase (commits
`48e611ad..dadf35ae`) restored several checked-data comparisons as DOLFINx
witnesses; this is a post-baseline coverage record, not a rewrite of the frozen
rows. Each restored comparison keeps master's structure/tolerance where it
holds and records any unavoidable divergence in the acceptance register.

| Frozen row | P5.2 resolution | Post-baseline evidence |
|---|---|---|
| N50 checked Nmag comparison (selected interaction) | already PASS; unchanged | `exchange_demag/test_exchange_demag.py` (example lane) still witnesses the checked Nmag averages/energies/Edensity data |
| N51 remaining selected Nmag comparisons have DOLFINx witnesses | RESOLVED | Nmag exchange-field comparison `tests/comparison/exchange/test_exchange_field_nmag_dolfinx.py` (`test_against_nmag`, tol 2e-14 verbatim); Nmag 1D exchange `tests/nmag/exchange_1d/test_exchange_1d_dolfinx.py`; Nmag 1D anisotropy `tests/nmag/anisotropy_1d/test_nmag_1d_anisotropy_dolfinx.py` |
| N62 selected OOMMF comparisons have DOLFINx witnesses | PARTIAL | the `exchange_demag` example now executes its OOMMF energy-density assertion in the examples gate (`d02c5458`: exch rel err 0.0395 < 5e-2, demag 0.0319 < 4e-2). The dedicated OOMMF comparison suite (`tests/oommf/*`, `comparison/anisotropy/test_anis_oommf.py`, `test_exchange_field.py::test_against_oommf`) is still NOT ported (imports legacy `dolfin`); backlog under C20 |
| N63 selected Magpar comparisons have DOLFINx witnesses | RESOLVED | coordinate-probe / probe-at-saved-coordinates Magpar comparisons `tests/comparison/anisotropy/test_anis_magpar_dolfinx.py`, `tests/comparison/demag/test_demag_magpar_dolfinx.py`, `tests/comparison/exchange/test_exchange_compare_magpar_dolfinx.py` (M8 node-order xfails retired). Exchange holds master tol `9e-8`; anisotropy retains an ~8% residual recorded as register **D29** (physics investigation recommended); the demag periodic-tile field cases xfail(strict) against **D17** |

N49 (Netgen binary backend necessity probe) is NOT resolved by the P5.2 test
conversion: it needs a selected-geometry probe, not a test transcription, so it
remains MISSING. Recorded here as a non-resolution so the orchestrator does not
mistake it for closed.

## Converted examples lane

The current gate is `pixi run -e dolfinx dolfinx-src-examples-pytest`. It is a
subprocess workflow witness: an exit success means that script’s embedded check
passed, not unchanged master behaviour. The recorded `PASS` values below are
**reduced-mode** results only. `FINMAG_EXAMPLE_FULL=1` runs all seventeen
entries, not merely the final three: it enables the final three and expands the
workload/save/plot paths of several fast entries. The clean-worktree result is
recorded below: 12 passed and 5 failed; it is baseline evidence, not acceptance.

| Entry | Reduced-mode evidence/result | FULL-mode qualification/result | Classification |
|---|---|---|---|
| `demag/test_field.py` | analytic; PASS (recorded) | PASS in clean FULL run | analytic |
| `demag/test_energy.py` | analytic; PASS (recorded) | PASS in clean FULL run | analytic |
| `macrospin/test_macrospin.py` | analytic; PASS (recorded) | adds damping cases and save/plot path; PASS in clean FULL run | analytic |
| `macrospin/test_macrospin_alpha_rtol.py` | analytic; PASS (recorded) | expands from 3 to 50 alpha values and save/plot path; PASS in clean FULL run | analytic |
| `varying_alpha/run.py` | profile regression; PASS (recorded) | PASS in clean FULL run | regression |
| `scheduling/sim_with_scheduling.py` | workflow completion; PASS (recorded) | PASS in clean FULL run | qualitative |
| `edge_damping/damping.py` | profile/geometry regression; PASS (recorded) | expanded save/plot path; PASS in clean FULL run | regression |
| `exchange_1D/1d_run.py` | physical evolution sanity; PASS (recorded) | PASS in clean FULL run | qualitative |
| `spatially-varying-anisotropy/run.py` | physical sanity; PASS (recorded) | PASS in clean FULL run | qualitative |
| `precession/run.py` | physical precession sanity; PASS (recorded) | expanded save/plot path; PASS in clean FULL run | qualitative |
| `time-dependent-applied-field/test_appfield.py` | analytic result; PASS (recorded) | same witness; PASS in clean FULL run | quantitative analytic |
| `cubic_anisotropy/hysteresis.py` | loop/switching execution; PASS (recorded) | 240-second wrapper timeout; harness evidence, not a physics failure; does not itself test stage relaxation | qualitative |
| `std_prob_3/run.py` | embedded published comparison; PASS (recorded) | 240-second wrapper timeout in full-resolution/bisection workload; harness evidence, not a physics failure | quantitative comparison |
| `exchange_demag/test_exchange_demag.py` | checked Nmag data; PASS (recorded) | same witness; PASS in clean FULL run | external-reference-data |
| `std_prob_4/test_std_prob_4.py` | not enabled | 1800-second wrapper timeout; harness evidence, not a physics failure | qualitative (not parity) |
| `cubic_anisotropy/sim.py` | not enabled | immediate `KeyError`: missing scheduler keyword `save_m` | qualitative |
| `magnetic_grain/suess_2001.py` | not enabled | immediate `ModuleNotFoundError: dolfin` via `finmag.util.helpers` | qualitative |

## Appendix A — pytest-discoverable original-master map

The original-master list is exactly the 101 source files enumerated by
`dev/bin/verify-python3-core-suite` at `pixi` plus these eight legacy examples:
`demag/test_{field,energy}.py`, `exchange_demag/test_exchange_demag.py`,
`macrospin/test_macrospin{,_alpha_rtol}.py`, `nmag_example_2/test_nmag_example_2.py`,
`std_prob_4/test_std_prob_4.py`, and `time-dependent-applied-field/test_appfield.py`.
Each item below gives its current counterpart/gate or the missing contract.
The two `retired-test_anis_nmag.py` and `retired-test_cubic_anis_nmag.py`
files are not pytest-discoverable and therefore are outside the 109 count; they
remain historical external-reference-data items requiring later classification
and an owner decision, not implicit deletions.

| Original-master file | Class | Current counterpart or missing contract |
|---|---|---|
| `src/finmag/tests/test_effective_field.py` | mapped-current | `dolfinx-src-effectivefield-pytest` |
| `src/finmag/tests/test_writing_data.py` | needs-translation | NDT/VTK/XDMF gate exists: `dolfinx-src-restart-output-pytest`. HDF5 field round-trip is now covered under `dolfinx-src-field-pytest` (`field_test.py::test_save_hdf5` transcribed in `test_field_dolfinx.py`, plus the new `test_field_hdf5_dolfinx.py`; SR1 P4-hdf5, N41); VTK/XDMF function readback remains missing |
| `src/finmag/field_test.py` | needs-translation | `dolfinx-src-field-pytest` covers selected basic contracts; full legacy field file is not assertion-equivalent |
| `src/finmag/field_setters_test.py` | needs-translation | `dolfinx-src-field-pytest`; generic-vector contract remains missing |
| `src/finmag/util/fileio_test.py` | needs-translation | `dolfinx-src-restart-output-pytest`. The legacy `dolfinh5tools` timeseries HDF5 contract is not reproduced, but the coordinate-aware HDF5 field round-trip now has a gate under `dolfinx-src-field-pytest` (`field_test.py::test_save_hdf5` transcribed in `test_field_dolfinx.py`, plus `test_field_hdf5_dolfinx.py`; SR1 P4-hdf5, N41) |
| `src/finmag/util/helpers_test.py` | needs-decision | selected helper convenience contract N48 has no positive gate; Mercurial assertions remain M10 |
| `src/finmag/util/length_scales_test.py` | needs-translation | Owner-Now convenience surface; no current DOLFINx gate |
| `src/finmag/util/meshes_test.py` | needs-translation | `dolfinx-src-meshes-pytest` covers selected generators, not the bundled legacy file |
| `src/finmag/util/mesh_templates_test.py` | needs-translation | `dolfinx-src-meshes-pytest` covers selected templates, not the bundled legacy file |
| `src/finmag/util/pbc_test.py` | needs-translation | Owner-Now `Simulation(pbc=...)` N44 has no positive gate |
| `src/finmag/util/fft_test.py` | later | normal-mode/FFT analysis is C17 |
| `src/finmag/util/plot_helpers_test.py` | needs-translation | selected plotting is N12; no current gate |
| `src/finmag/util/vtk_saver_test.py` | mapped-current | `dolfinx-src-restart-output-pytest` |
| `src/finmag/util/dmi_helper_test.py` | mapped-current | `dolfinx-src-dmi-pytest` |
| `src/finmag/util/oommf/test_mesh.py` | external-reference-data | checked-in comparisons retained; live harness is C20 |
| `src/finmag/util/visualization_test.py` | needs-translation | selected plotting N12; renderer/movie decisions later |
| `src/finmag/tests/comparison/test_dmdt.py` | external-reference-data | checked-in/oracle fixtures; DOLFINx `dolfinx-src-llg-pytest` is not assertion-equivalent |
| `src/finmag/tests/comparison/anisotropy/test_anis_magpar.py` | external-reference-data | Magpar fixtures retained; known mesh drift requires review |
| `src/finmag/tests/comparison/anisotropy/test_anis_oommf.py` | external-reference-data | checked OOMMF data; `dolfinx-src-energies-pytest` covers local physics |
| `src/finmag/tests/comparison/anisotropy/test_cubic_anis_oommf.py` | external-reference-data | checked OOMMF data; `dolfinx-src-cubicanis-pytest` |
| `src/finmag/tests/comparison/demag/test_demag_field.py` | external-reference-data | checked Magpar data; `dolfinx-src-demag-pytest` |
| `src/finmag/tests/comparison/exchange/test_exchange_compare_magpar.py` | external-reference-data | checked Magpar data; `dolfinx-src-energies-pytest` |
| `src/finmag/tests/comparison/exchange/test_exchange_field.py` | external-reference-data | checked Nmag data; `dolfinx-src-energies-pytest`. Nmag half (`test_against_nmag`, tol 2e-14) transcribed under DOLFINx in `tests/comparison/exchange/test_exchange_field_nmag_dolfinx.py` (P5.2). The OOMMF half (`test_against_oommf`, tol 8e-2) is NOT covered under DOLFINx and is intentionally not transcribed (noted, not silently dropped) -- likely gated on porting `finmag.util.oommf`; parity backlog (C20) |
| `src/finmag/tests/oommf/test_anisotropy.py` | external-reference-data | OOMMF data/workflow deferred C20 |
| `src/finmag/tests/oommf/test_exchange.py` | external-reference-data | OOMMF data/workflow deferred C20; the file still `import dolfin`s and is not run under DOLFINx. The OOMMF exchange comparison remains uncovered (parity backlog, likely gated on porting `finmag.util.oommf`) |
| `src/finmag/tests/nmag/anisotropy_1d/test_nmag_1d_anisotropy.py` | external-reference-data | checked Nmag data; `dolfinx-src-energies-pytest` |
| `src/finmag/tests/nmag/exchange_1d/test_exchange_1d.py` | external-reference-data | checked Nmag data; `dolfinx-src-energies-pytest` |
| `src/finmag/tests/nmag/exchange_3d/test_dynamics_3D.py` | external-reference-data | checked Nmag data; serial dynamics `dolfinx-src-llg-pytest` |
| `src/finmag/tests/nmag/spinwaves/test_spinwaves.py` | later | normal-mode/spinwave workflow C17 |
| `src/finmag/tests/slonczewski/validation/finmag/test_finmag_validation.py` | needs-translation | `dolfinx-src-stt-pytest`; full legacy validation trajectory not mapped |
| `src/finmag/tests/slonczewski/oscillator/test_oscillator.py` | needs-translation | `dolfinx-src-stt-pytest`; oscillator workflow not mapped |
| `src/finmag/normal_modes/eigenmodes/helpers_test.py` | later | normal modes C17 |
| `src/finmag/normal_modes/eigenmodes/eigenproblems_test.py` | later | normal modes C17 |
| `src/finmag/normal_modes/eigenmodes/eigensolvers_test.py` | later | normal modes C17 |
| `src/finmag/normal_modes/deprecated/normal_modes_deprecated_test.py` | later | normal modes C17 |
| `src/finmag/util/test_set_function_values.py` | needs-translation | Field conversion N11 |
| `src/finmag/util/test_dmi_from_helix.py` | mapped-current | `dolfinx-src-dmi-pytest` |
| `src/finmag/drivers/tests/test_integrator_raises_exception_on_exceed_maxsteps.py` | needs-translation | drivers are tested, but this exact legacy maxsteps exception contract is not mapped |
| `src/finmag/scheduler/scheduler_test.py` | mapped-current | `dolfinx-src-restart-output-pytest` |
| `src/finmag/tests/test_restart_simulation.py` | needs-translation | `dolfinx-src-restart-output-pytest`; backend trajectory/provenance and full-state reconstruction remain missing |
| `src/finmag/tests/test_meshes.py` | mapped-current | `dolfinx-src-meshes-pytest` |
| `src/finmag/tests/test_sim_ode.py` | mapped-current | `dolfinx-src-simulation-pytest`, `dolfinx-src-llg-pytest` |
| `src/finmag/tests/test_applied_field.py` | mapped-current | `dolfinx-src-energies-pytest` |
| `src/finmag/tests/test_anis.py` | mapped-current | `dolfinx-src-energies-pytest` |
| `src/finmag/tests/test_dmi.py` | mapped-current | `dolfinx-src-dmi-pytest` |
| `src/finmag/tests/test_dmi_terms.py` | mapped-current | `dolfinx-src-dmi-pytest` |
| `src/finmag/tests/test_energy_creation_with_variable_Ms.py` | mapped-current | `dolfinx-src-varparams-pytest` |
| `src/finmag/tests/energy_density/test_energy_density.py` | needs-translation | energy-density output contract not separately current |
| `src/finmag/tests/test_interactions_scale_linearly_with_m.py` | needs-translation | selected energy checks exist; this bundled linearity contract is not mapped |
| `src/finmag/tests/test_heun.py` | needs-decision | legacy Heun driver has no owner disposition; see M16 |
| `src/finmag/tests/test_spatially_varying_alpha.py` | mapped-current | `dolfinx-src-varparams-pytest` |
| `src/finmag/tests/test_spatially_varying_anisotropy.py` | mapped-current | `dolfinx-src-varparams-pytest` |
| `src/finmag/tests/test_unit_length.py` | mapped-current | `dolfinx-src-energies-pytest`, `dolfinx-src-simulation-pytest` |
| `src/finmag/tests/test_1d_domain_wall_profile_uniaxial_anisotropy.py` | needs-translation | selected uniaxial energy checks exist; profile equilibrium N14 is missing |
| `src/finmag/energies/exchange_test.py` | mapped-current | `dolfinx-src-energies-pytest` |
| `src/finmag/energies/anisotropy_test.py` | mapped-current | `dolfinx-src-energies-pytest` |
| `src/finmag/energies/cubic_anisotropy_test.py` | needs-translation | `dolfinx-src-cubicanis-pytest`; varying axes remain missing |
| `src/finmag/energies/demag/demag_pbc_test.py` | mapped-current | `dolfinx-src-treecode-pytest` |
| `src/finmag/energies/demag/fk_demag_test.py` | mapped-current | `dolfinx-src-demag-pytest` |
| `src/finmag/energies/demag/fk_demag_2d_test.py` | not-now | Demag2D is explicitly deferred |
| `src/finmag/energies/test_energies_in_regions.py` | mapped-current | `dolfinx-src-varparams-pytest` |
| `src/finmag/energies/magnetostatic_field_test.py` | needs-translation | FK demag is current; exact legacy helper contract unmapped |
| `src/finmag/energies/thin_film_demag_test.py` | mapped-current | `dolfinx-src-energies-pytest` |
| `src/finmag/energies/zeeman_test.py` | mapped-current | `dolfinx-src-energies-pytest` |
| `src/finmag/energies/dmi_test.py` | mapped-current | `dolfinx-src-dmi-pytest` |
| `src/finmag/tests/test_exchange_static.py` | needs-translation | exchange field/energy N13 passes; static equilibrium N14 is missing |
| `src/finmag/tests/test_llg.py` | needs-translation | selected NumPy LLG checks exist; the bundled legacy test is not assertion-equivalent |
| `src/finmag/util/ode/tests/test_sundials_ode.py` | needs-translation | `dolfinx-src-sundials-pytest`; generic ODE API not mapped |
| `src/finmag/drivers/tests/test_relaxation.py` | mapped-current | `dolfinx-src-simulation-pytest` |
| `src/finmag/drivers/tests/test_relax_two_times.py` | needs-translation | repeated-relax/hysteresis lifecycle correction required |
| `src/finmag/tests/test_cyclic_references_in_sim.py` | needs-translation | simulation lifecycle is current; exact cycle contract unmapped |
| `src/finmag/physics/tests/test_effective_field.py` | mapped-current | `dolfinx-src-effectivefield-pytest` |
| `src/finmag/physics/tests/test_equation.py` | later | compiled Equation backend decision M3 |
| `src/finmag/physics/tests/test_terms.py` | later | compiled terms backend decision M3 |
| `src/finmag/tests/bugs/test_bug_ndt_file_writing.py` | mapped-current | `dolfinx-src-restart-output-pytest` |
| `src/finmag/tests/test_time.py` | needs-translation | backend-specific time getter/reset lifecycle is not fully mapped |
| `src/finmag/tests/demag/test_bem_computation.py` | needs-translation | native BEM gate covers selected array contract, not this bundled legacy file |
| `src/finmag/tests/demag/test_demag_sphere.py` | mapped-current | `dolfinx-src-demag-pytest` |
| `src/finmag/tests/zhangli/zhang_li_test.py` | mapped-current | `dolfinx-src-stt-pytest` |
| `src/finmag/tests/zhangli/stt_nonlocal_test.py` | not-now | nonlocal `LLG_STT`; see M5 |
| `src/finmag/physics/llb/sllg_test.py` | not-now | thermal SLLG; retained for later parity |
| `src/finmag/physics/llb/llb_test.py` | not-now | LLB; retained for later parity |
| `src/finmag/physics/tests/neb/neb_test.py` | later | NEB C18 |
| `src/finmag/tests/test_jacobian.py` | needs-translation | Sundials J-times is current, legacy Jacobian test not mapped |
| `src/finmag/tests/test_solid_angle.py` | needs-translation | no separate DOLFINx solid-angle gate |
| `src/finmag/tests/test_solid_angle_invariance.py` | needs-translation | no separate DOLFINx solid-angle gate |
| `src/finmag/tests/test_skyrmions.py` | mapped-current | `dolfinx-src-io-utils-pytest` |
| `src/finmag/tests/cython/test_cython.py` | obsolete-review | legacy Cython compilation fixture; no user capability |
| `src/finmag/tests/test_sim_parallel.py` | later | serial SR1; general MPI stepping C19 |
| `src/finmag/sim/hysteresis_test.py` | needs-translation | Transcribed in `src/finmag/tests/test_hysteresis_dolfinx.py` (P5.2, `775e6e85`). Per-stage relaxation is NOT defective: corrected under register **D4** (SR1 P2.5) -- `hysteresis()` re-relaxes each stage to its own equilibrium and the on-axis oracle stall is a genuine degenerate Stoner-Wohlfarth saddle, not a skipped relaxation |
| `src/finmag/sim/magnetisation_patterns_test.py` | needs-translation | Owner-Now initialisers N45 have no positive gate |
| `src/finmag/sim/sim_helpers_test.py` | needs-translation | `dolfinx-src-io-utils-pytest`; simulation probing and region-output helpers remain missing |
| `src/finmag/sim/sim_test.py` | needs-translation | current simulation gate covers selected core, not bundled legacy surface |
| `src/finmag/drivers/tests/sundials_nsteps_test.py` | needs-translation | `dolfinx-src-sundials-pytest`; complete Simulation reset/restart lifecycle remains missing |
| `src/finmag/drivers/tests/sundials_reinit_test.py` | needs-translation | `dolfinx-src-sundials-pytest`; complete Simulation reset/restart lifecycle remains missing |
| `src/finmag/drivers/tests/test_integrators.py` | needs-translation | backend gates cover selected operations; full legacy integrator matrix is not mapped |
| `src/finmag/drivers/tests/test_scipy.py` | mapped-current | `dolfinx-src-scipy-pytest` |
| `src/finmag/tests/jacobean/test_jacobean_computation.py` | needs-translation | J-times witness exists; full legacy Jacobian contract unmapped |
| `src/finmag/tests/jacobean/test_jacobean_integration.py` | needs-translation | J-times witness exists; full legacy Jacobian contract unmapped |
| `src/finmag/tests/jacobean/test_native_llg.py` | needs-translation | NumPy LLG is current; native legacy kernel assertion unmapped |
| `src/finmag/util/ode/tests/test_sundials_stiff_ode.py` | needs-decision | historical SciPy upstream xfail remains undecided M11a |
| `examples/demag/test_field.py` | mapped-current | fast example lane, analytic |
| `examples/demag/test_energy.py` | mapped-current | fast example lane, analytic |
| `examples/exchange_demag/test_exchange_demag.py` | mapped-current | fast example lane, checked Nmag reference data |
| `examples/macrospin/test_macrospin.py` | mapped-current | fast example lane, analytic |
| `examples/macrospin/test_macrospin_alpha_rtol.py` | mapped-current | fast example lane, analytic |
| `examples/nmag_example_2/test_nmag_example_2.py` | external-reference-data | Nmag live generator not selected; retain data only |
| `examples/std_prob_4/test_std_prob_4.py` | needs-translation | FULL wrapper timeout; qualitative gate is not parity acceptance |
| `examples/time-dependent-applied-field/test_appfield.py` | mapped-current | fast example lane, quantitative analytic frequency/deviation check |

## Appendix B — all original-master example Python files

This is the complete 49-file `git ls-tree -r master examples` Python inventory.
It is broader than the pytest-discoverable appendix: utility scripts and
historical native-binding demonstrations are classified too.

| Original-master example | Class | Current counterpart or missing contract |
|---|---|---|
| `examples/boost_python/demo1_helloworld/demo1.py` | obsolete-review | standalone Boost.Python demonstration, not Finmag user API |
| `examples/boost_python/demo2_classdef/demo2.py` | obsolete-review | standalone Boost.Python demonstration |
| `examples/boost_python/demo3_numpy/demo3.py` | obsolete-review | standalone Boost.Python demonstration |
| `examples/boost_python/demo3_numpy/demo3_exceptions.py` | obsolete-review | standalone Boost.Python demonstration |
| `examples/boost_python/demo4_dolfin/demo4.py` | obsolete-review | legacy dolfin binding demonstration |
| `examples/boost_python/demo4_dolfin/demo4_swig_objects.py` | obsolete-review | legacy dolfin/SWIG demonstration |
| `examples/boost_python/petsc/demo5.py` | obsolete-review | standalone PETSc binding demonstration |
| `examples/cubic_anisotropy/hysteresis.py` | mapped-current | fast converted lane; qualitative loop execution only |
| `examples/cubic_anisotropy/nmag_original.py` | external-reference-data | legacy Nmag generator; retain data, not live toolchain |
| `examples/cubic_anisotropy/sim.py` | needs-translation | FULL fails immediately: missing scheduler keyword `save_m` |
| `examples/demag/run_nmag.py` | external-reference-data | legacy Nmag generator; retain data |
| `examples/demag/short_nmag_comp_tests2.py` | external-reference-data | Nmag comparison data retained; no DOLFINx counterpart |
| `examples/demag/test_energy.py` | mapped-current | fast converted lane, analytic |
| `examples/demag/test_field.py` | mapped-current | fast converted lane, analytic |
| `examples/dispersion_curves/dispersion.py` | later | FFT/dispersion/normal-mode analysis C17 |
| `examples/edge_damping/damping.py` | mapped-current | fast converted lane, profile/geometry regression |
| `examples/exchange_1D/1d_angles.py` | needs-translation | analysis/plot companion; no current gate |
| `examples/exchange_1D/1d_anim.py` | later | animation/movie workflow deferred |
| `examples/exchange_1D/1d_dynamics.py` | needs-translation | dynamics workflow not separately current |
| `examples/exchange_1D/1d_plot.py` | needs-translation | plotting N12 is missing |
| `examples/exchange_1D/1d_run.py` | mapped-current | fast converted lane, qualitative evolution |
| `examples/exchange_1D/run_visual.py` | later | visualisation/movie workflow deferred |
| `examples/exchange_demag/run_nmag.py` | external-reference-data | legacy Nmag generator; retain data |
| `examples/exchange_demag/simple_1D_finmag.py` | needs-translation | legacy workflow has no converted gate |
| `examples/exchange_demag/simple_1D_nmag.py` | external-reference-data | legacy Nmag generator |
| `examples/exchange_demag/test_exchange_demag.py` | mapped-current | fast converted lane, checked Nmag data |
| `examples/exchange_demag/timings/run_finmag.py` | later | performance/timing experiment, no SR1 contract |
| `examples/exchange_demag/timings/run_nmag.py` | external-reference-data | legacy Nmag timing generator |
| `examples/exchange_demag/ztest_exchange_demag_new.py` | obsolete-review | superseded experimental test script |
| `examples/llb/macrospin.py` | later | thermal LLB C16 |
| `examples/macrospin/test_macrospin.py` | mapped-current | fast converted lane, analytic |
| `examples/macrospin/test_macrospin_alpha_rtol.py` | mapped-current | fast converted lane, analytic |
| `examples/magnetic_grain/suess_2001.py` | needs-translation | FULL fails immediately through raw `dolfin` import; execution/plot smoke only |
| `examples/nmag_example_2/__init__.py` | external-reference-data | Nmag package scaffold, no live port |
| `examples/nmag_example_2/callgraph.py` | obsolete-review | historical callgraph utility |
| `examples/nmag_example_2/run_finmag.py` | needs-translation | legacy comparison workflow not converted |
| `examples/nmag_example_2/run_nmag.py` | external-reference-data | legacy Nmag generator |
| `examples/nmag_example_2/test_nmag_example_2.py` | external-reference-data | legacy Nmag comparison retained as data only |
| `examples/precession/run.py` | mapped-current | fast converted lane, qualitative precession |
| `examples/scheduling/sim_with_notification.py` | needs-translation | notification integration not selected |
| `examples/scheduling/sim_with_progressbar.py` | needs-translation | progress-bar integration not selected |
| `examples/scheduling/sim_with_scheduling.py` | mapped-current | fast converted lane, scheduling workflow |
| `examples/spatially-varying-anisotropy/run.py` | mapped-current | fast converted lane, qualitative dynamics |
| `examples/std_prob_3/run.py` | mapped-current | fast converted lane, embedded published comparison; not a master oracle/parity proof |
| `examples/std_prob_3/table_for_doc.py` | needs-translation | documentation post-processing has no gate |
| `examples/std_prob_4/plot_averages.py` | needs-translation | plotting companion; no current gate |
| `examples/std_prob_4/test_std_prob_4.py` | needs-translation | FULL wrapper timeout; no quantitative parity acceptance |
| `examples/time-dependent-applied-field/test_appfield.py` | mapped-current | fast converted lane, quantitative analytic |
| `examples/varying_alpha/run.py` | mapped-current | fast converted lane, profile regression |

## P0.2 baseline commands and execution record — complete evidence, not acceptance

The present worktree has intentional dirty documentation. Therefore these
commands can record a **current dirty-doc baseline**, but cannot be claimed as
a clean reproducible source baseline. Capture stdout/stderr, exit code, package
versions, exact `HEAD`, and `git status --short` before source edits. The full
example lane expands the workload from the fast fourteen entries to all
seventeen entries (the same fourteen plus three slow entries); its wrapper
keeps the per-example timeouts defined in `examples/test_examples_dolfinx.py`.
The unchanged 120–3600-second per-example timeouts defined in
`examples/test_examples_dolfinx.py` are intentionally expected to record
wrapper/harness timeouts where they occur; such timeouts are evidence to freeze,
not physics failures. Only after P0.2 evidence is recorded may timeout scaling
or a wrapper adjustment be proposed as a separate reviewed slice before a
meaningful all-green FULL acceptance run.
It can overwrite tracked `examples/magnetic_grain/mz.png` and
`examples/std_prob_3/doc_table.rst`; capture post-run `git status --short` and
`git diff -- examples/magnetic_grain/mz.png examples/std_prob_3/doc_table.rst`.
Do not run it in this working tree if preserving local changes matters. For a
future reproducible rerun, create a disposable clean worktree at the frozen
commit, run `pixi run -e dolfinx dolfinx-install-editable` there before the
lane, and restore the editable install to the main worktree afterwards.

```sh
git status --short
git rev-parse HEAD
pixi run -e dolfinx dolfinx-versions
dev/bin/verify-dolfinx-m5
FINMAG_EXAMPLE_FULL=1 pytest -q examples/test_examples_dolfinx.py
```

The aggregate command contains the supported install/native-build/provenance
sequence and all focused current gates. The final command is deliberately
separate because the aggregate verifier only records the fast lane. No legacy
oracle command is part of P0.2; those are per-slice evidence after a specific
RED contract exists.

Recorded baseline, without source edits: the dirty-doc main worktree at
`f1a1344c423e74687ddf820c1fafc056a6271fe1` ran
`dev/bin/verify-dolfinx-m5` with exit 0 and all 32 steps green (fast examples:
14 passed, 3 skipped); log `/tmp/finmag-p0-baseline-m5.log`. Versions were
Python 3.12.13, DOLFINx 0.10.0, NumPy 2.4.6, SciPy 1.18.0, mpi4py 4.1.2 with
OpenMPI 5.0.10, and PETSc 3.25.2.

The clean detached worktree `/tmp/finmag-p0-full-f1a1344c` at the same commit
proved import provenance from its own `src/` through `PYTHONPATH=temp/src`, then
ran `FINMAG_EXAMPLE_FULL=1 pytest -q examples/test_examples_dolfinx.py`:
exit 1, 12 passed and 5 failed in 2557.64 seconds (42:37); log
`/tmp/finmag-p0-baseline-full-examples-clean.log`. Its git status and diff were
empty after the run; `examples/magnetic_grain/mz.png` and
`examples/std_prob_3/doc_table.rst` were unchanged. The main editable import
still resolved to `/home/sam/repos/finmag/src/finmag/__init__.py`, so no
editable-install restoration was required. The three timeouts are wrapper
evidence, not physics failures. This baseline is complete; it is not an
all-green FULL acceptance run.

| Run date/time | Worktree/commit | Dirty status captured | Aggregate result | Full lane result | Logs/artifacts | Orchestrator |
|---|---|---|---|---|---|---|
| 2026-07-23 baseline | main dirty-doc + clean detached `/tmp/finmag-p0-full-f1a1344c` at `f1a1344c` | clean FULL worktree status/diff empty | exit 0; 32 green; fast 14 passed, 3 skipped | exit 1; 12 passed, 5 failed: 3 wrapper timeouts, `save_m` KeyError, raw-`dolfin` import | `/tmp/finmag-p0-baseline-m5.log`; `/tmp/finmag-p0-baseline-full-examples-clean.log`; tracked artifacts unchanged | recorded |

[Codex GPT-5.6]
