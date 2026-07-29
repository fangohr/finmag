# Post-SR1 Performance Implementation Plan

(SR1 = Support Release 1, the first supported subset of the DOLFINx port;
see [`../SUPPORTED.md`](../SUPPORTED.md). This plan starts after its
declaration.)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DOLFINx port usable in wall-clock terms — close the
measured ~17.6× gap to legacy (register P1) by eliminating per-evaluation
re-assembly and tuning the solve path, WITHOUT changing physics, public API,
or any ratified disposition (D32's method-name rejection stands).

**Architecture:** Evidence-gated pipeline. T1 builds a committed benchmark +
profiling harness and produces the cost table that RANKS everything after
it; T2 takes the zero-risk wins; T3/T4 build the flagship precomputed-matrix
fast path for the linear energy terms (assemble the field operator once,
`H = (g·m)/nodal_volume` per evaluation — exactly legacy's
`box-matrix-petsc` mechanism, but as an INTERNAL optimization of the
existing `box-assemble` semantics, so the dropped legacy method-name API is
not resurrected); T5/T6 are probe-gated RHS and demag tuning; T7
re-measures, recalibrates the FULL-lane timeouts DOWNWARD, and closes P1
with numbers. T8 (native compiled RHS) is a contingency, decided only if
T7 misses targets.

**Tech Stack:** pixi env `dolfinx`, DOLFINx 0.10 / FFCx / PETSc, cProfile +
`time.perf_counter` harness, existing 33-gate verifier + oracle fixtures as
the physics guard.

**Execution timing:** starts AFTER the SR1 declaration commit (the current
SR1 pipeline must finish first; the machine must be free of the acceptance
run). Exception: T1's harness may be authored (not run) earlier.

## Global Constraints

- Branch `dolfinx-parity` (or a successor the owner names). NEVER change
  physics semantics: every optimization must prove field equivalence against
  the existing assembly path (rel. difference ≤ 1e-12 on asymmetric
  spatially-varying test fields) AND leave every oracle-fixture trajectory
  gate bit-green. The 33-gate verifier passes after every task.
- No public API change: `method="box-assemble"` remains the only accepted
  name (register D32, ratified 2026-07-28). Fast paths are internal.
  No new runtime dependencies without owner sign-off.
- No tolerance loosening anywhere, including solver tolerances, unless the
  legacy value is strictly matched and documented (T6's explicit job).
- Every speedup claim is backed by the committed harness run before AND
  after, on the same machine, recorded in the task report and the final
  P1 register update. No "felt faster".
- Cache-invalidation rule: any precomputed operator must be invalidated by
  every mutation path that can change its coefficients (`A`, `Ms`, `K1`,
  axis, mesh) — each fast-path task enumerates those paths and tests at
  least one invalidation explicitly.
- Commit style as SR1: one reviewed commit per task,
  `Co-Authored-By: <model> <noreply@anthropic.com>`.

---

### Task 1: Benchmark + profiling harness (the evidence gate)

**Files:**
- Create: `dev/benchmarks/bench_field_calls.py` (micro: per-call
  `compute_field` cost for Exchange, UniaxialAnisotropy, DMI, FKDemag on
  three meshes — barmini-size, 10×10×10 box, std_prob_4's bar mesh;
  N=100 calls, report mean/min µs per call)
- Create: `dev/benchmarks/bench_relax_segment.py` (macro: barmini
  `run_until(50e-12)` and a std_prob_4-mesh 100-step segment under
  cProfile; dumps top-30 cumulative functions + a category rollup:
  assembly / demag-KSP / BEM-matvec / numpy-RHS / ordering-conversions /
  integrator / scheduler)
- Create: `dev/benchmarks/README.md` (how to run; results table template)
- Commit baseline results table into the README.

**Interfaces:**
- Produces: `CATEGORY_TABLE` (the measured % split) — T2–T6 each cite it;
  the plan's priority order below is provisional until this table exists,
  and the orchestrator re-orders T5 vs T6 by measured share.

- [ ] **Step 1:** Author both scripts. Category rollup maps profile frames:
  `fem.form|assemble_vector|ffcx` → assembly; `KSP|PETSc.*solve` →
  demag-KSP; `bem|matvec` → BEM; `llg.py::_dmdt|solve_for` → RHS;
  `ordered_numpy|blocked|interleaved` → ordering; `cvode|sundials|scipy` →
  integrator.
- [ ] **Step 2:** Environment sanity block in the harness output: `nproc`,
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` (unset = report), PETSc options
  in effect. Flag oversubscription (threads × MPI > cores).
- [ ] **Step 3:** Run both on the free machine; commit scripts + baseline
  table: `Benchmark harness + baseline cost table (perf T1)`.
- [ ] **Step 4:** In the report, state the measured priority order for
  T3–T6 and predicted ceiling per item (Amdahl: category share → max
  speedup if reduced to ~0).

### Task 2: Zero-risk quick wins

**Files:**
- Modify: `src/finmag/energies/energy_base.py` — cache `fem.form(...)`
  objects ONCE in `setup()` (dE_dm form, energy form, nodal-energy form);
  `_assembled_vector` gains a pre-wrapped-form variant. No behaviour
  change: same assembly, minus per-call `fem.form` wrapping.
- Modify: `src/finmag/energies/demag/fk_demag.py` — Krylov warm starts if
  absent: keep previous `phi1`/`phi2` solutions as initial guesses
  (`ksp.setInitialGuessNonzero(True)` + not zeroing the vectors). Verify
  legacy did the same or better; if the port already warm-starts, record
  and skip.
- Test: equivalence tests beside existing gates (energies gate + demag
  gate); micro-benchmark before/after in the report.

- [ ] **Step 1:** RED-first is inverted here (no behaviour change): write
  the equivalence test FIRST (asymmetric varying-`Ms` field; assert new
  path == old path ≤1e-12 and demag field unchanged ≤1e-12 after 3
  successive solves with different m), prove it passes on the OLD code,
  then apply the change and prove it still passes.
- [ ] **Step 2:** Run `dolfinx-src-energies-pytest`, `dolfinx-src-demag-pytest`,
  `dolfinx-src-treecode-pytest` + micro-bench; expect measurable per-call
  reduction (record %).
- [ ] **Step 3:** Commit: `Cache forms + demag warm starts (perf T2)`.

### Task 3: Precomputed-matrix fast path — Exchange (flagship)

**Files:**
- Modify: `src/finmag/energies/exchange.py` + `energy_base.py`: at
  `setup()`, assemble the bilinear operator `g` once
  (`fem.petsc.assemble_matrix` of the derivative form's bilinear part);
  per call: `H_raw = (g @ m_petsc) / nodal_volume_S3`. Internal switch:
  subclasses declare `_linear_in_m = True`; base class uses the matrix
  path when true, per-call assembly otherwise. Public `method` name and
  all docstrings/semantics unchanged (document the internal fast path in
  the class docstring with a D32 note).
- Cache invalidation: enumerate mutation paths (`set_value`-class updates
  of `A`, `Ms` field changes, `mark_regions`/varying-parameter setters) —
  invalidate `g` on each; test one explicitly (change `A`, assert field
  matches fresh-assembly result).
- Test: equivalence vs per-call assembly on (a) constant A/Ms, (b)
  spatially varying A and Ms, (c) after invalidation; plus the existing
  exchange oracle fixtures must stay green untouched.

- [ ] **Step 1:** Equivalence tests first (pass against a
  forced-per-call-assembly reference mode kept for testing).
- [ ] **Step 2:** Implement; run `dolfinx-src-energies-pytest` + comparison
  gate + micro-bench (expect order-of-magnitude per-call improvement on
  exchange).
- [ ] **Step 3:** Macro-bench: barmini relax segment before/after; record.
- [ ] **Step 4:** Commit: `Exchange precomputed-operator fast path (perf T3)`.

### Task 4: Fast path — UniaxialAnisotropy and DMI

Same pattern as T3 (`H_anis = c(m·a)a` and the DMI form are linear in m;
CubicAnisotropy is NOT — it keeps per-call assembly and says so in its
docstring). Same equivalence/invalidation/test discipline; cubic gate +
dmi gate + varparams gate green. Commit:
`Anisotropy/DMI precomputed-operator fast paths (perf T4)`.

### Task 5: RHS and ordering-conversion cost (probe-gated)

ONLY if T1's table shows `numpy-RHS` + `ordering` ≥ 10% after T3/T4.
Preallocate buffers in `llg.py`'s dm/dt path; eliminate per-call
blocked↔interleaved conversions by keeping a persistent view where the
Task-31 ordering contract allows; no semantics change (same tests guard).
Commit: `RHS buffer reuse and ordering hygiene (perf T5)`.

### Task 6: Demag solver tuning (probe-gated)

ONLY if demag-KSP dominates after T2. Match legacy's demag solver
tolerances EXACTLY (read them from `b5015c5a` fk_demag; document any
current mismatch as the finding it is), evaluate KSP/PC choice on the
benchmark meshes. Any tolerance change must be to the legacy value, not
looser. Commit: `Demag KSP parity tuning (perf T6)`.

### Task 7: Re-measure, recalibrate, close P1

- Re-run the full benchmark harness; update the README table (before/after
  per task).
- Re-run the three heavy FULL-lane entries once each (std_prob_3,
  std_prob_4, magnetic_grain), record new wall times, and LOWER the
  wrapper timeouts to new-measured×2 (same evidence-comment discipline as
  SR1 T3).
- Update register P1: cause (per-eval assembly + numpy RHS) → resolution
  (measured end-to-end factor); update HANDOVER + SUPPORTED.md perf notes.
- **Targets (aspirational, set properly from T1's table):** std_prob_4
  full trace ≤ 1 h; std_prob_3 bisection ≤ 1.5 h; whole FULL lane ≤ 3 h.
  If met → P1 closed. If missed → T8 decision goes to the owner with the
  numbers.
- Commit: `Performance recalibration + P1 closure (perf T7)`.

### Task 8 (contingency, owner decision): native compiled RHS

Rebuild of legacy's compiled `Equation` (C++/nanobind or cython) for the
dm/dt hot loop. Substantially larger effort and a new build surface — only
proposed to the owner if T7 misses targets, with T1/T7 numbers attached.

---

## Priority rationale (pending T1's measured table)

1. **T3 (exchange matrix path)** — highest confidence × impact: the
  per-eval `fem.form`+assemble is structural waste on the hottest term;
  legacy's own design proves the fix.
2. **T2 (form caching + warm starts)** — near-zero risk, compounding.
3. **T4 (anisotropy/DMI)** — same mechanism, more terms.
4. **T5/T6** — real but unquantified; the probe ranks them.
5. **T8** — last resort; most effort, new build complexity.

## Self-review notes

- The plan never touches ratified dispositions: D32's name rejection is
  preserved (internal `_linear_in_m` switch, no method-name API); no test
  tolerances loosened; cubic anisotropy explicitly excluded from the
  linear fast path.
- Physics safety = equivalence tests + untouched oracle gates + 33-gate
  verifier per task; cache invalidation gets explicit enumeration + test.
- Every claim is harness-measured; the harness itself is committed and
  reproducible.
