# Performance

Audience: owner and contributor. What is slow today, why, and what the plan
is. For what is supported at all (independent of speed), see
[`SUPPORTED.md`](SUPPORTED.md); the underlying decision record is register
row **P1** in [`acceptance-register.md`](acceptance-register.md).

## The measured slowdown

The `std_prob_3` example, run at full workload, measures roughly 3161
seconds per relax simulation and about 8.8 hours for its full ten-simulation
bisection — against the script's own header comment estimating "~30
minutes" for the whole bisection. That is a **~17.6x gap**. It is real
measured evidence: nobody has re-run legacy master on the same hardware to
confirm the comment was ever accurate rather than aspirational, but the
timeout budgets used in CI were sized to the measured rate, not to the
comment.

## Root cause

Identified 2026-07-28: the port re-does form compilation and vector assembly
on **every single field evaluation** — `fem.form(...)` followed by
`assemble_vector(...)` inside `_compute_field_raw`
(`src/finmag/energies/energy_base.py:183`). Legacy's default assembly method
(`box-matrix-petsc`) instead assembled the field operator **once**, at
setup, and merely *applied* it on each subsequent evaluation
(`energy_base.py:214-222` in the original `master` at commit `b5015c5a`).

A second, compounding factor: the port's LLG right-hand side is plain numpy,
where legacy used a compiled `Equation`/`terms` backend (tracked separately
as register **M3**). Re-assembling from scratch on every call, in an
uncompiled RHS, is the combination that produces the ~17.6x gap.

## Measured attribution (2026-10-08)

The root cause above is real, but profiling showed it is **not** the dominant
cost. Measured serially at 20787 dofs (a 12x12x40 bar with Exchange,
UniaxialAnisotropy and Zeeman; no demag), one LLG right-hand-side evaluation
cost 96.8 ms:

| Component | Cost | Share |
|---|---|---|
| `UniaxialAnisotropy` assembly | 60.1 ms | 62% |
| vertex->dof permutation rebuilds (6 per evaluation) | 33.1 ms | 34% |
| `Exchange` assembly | 3.4 ms | 3.5% |
| per-evaluation `fem.form(...)` | ~0.4 ms | 0.4% |

So the re-assembly named above costs 0.4%, and its cost is flat
(~110-130 us per call) rather than growing with the mesh. Two causes the fix
plan does not list account for almost everything:

1. **An uncached coordinate permutation.** `finmag.field._owned_vertex_to_dof`
   built and queried a scipy `cKDTree` over every dof coordinate on each call,
   for a permutation that is a pure function of the function space. It is
   reached from every interaction's `compute_field`, from `EffectiveField`'s
   write-back and from `LLG`'s `m`/`dm/dt` conversions -- six rebuilds per
   right-hand-side evaluation. Caching it per function space took the
   evaluation from 96.8 ms to **65.0 ms**. (The plan has this as task T5,
   probe-gated "ONLY if ... >= 10%"; it was 34% before anything else changed.)

2. **A quartic term that is almost always zero.** `UniaxialAnisotropy` always
   put `- K2*(axis.m)**4` into the energy integrand, including for its default
   `K2 = 0`. With CG1 axis and m that is degree 8, so FFCx raises the
   quadrature degree for the whole form: assembling the full integrand cost
   60.1 ms against 8.5 ms for the `K1` term alone, a measured **7.0x**, and the
   ratio is the same whether `K2` is zero or not -- the cost is the term's
   presence, not its value.

   Leaving the term out when `K2` is the constant `0` took the evaluation from
   65.0 ms to **12.5 ms**. The optimisation is scoped to that constant case --
   the default, and the one that matters: a spatially varying `K2` keeps the
   term even where it evaluates to zero. Testing the declared constant rather
   than the assembled values also keeps the decision rank-invariant by
   construction, where a value-based test would have to agree across ranks to
   avoid building different forms.

A precomputed-matrix prototype (legacy's `box-matrix-petsc` mechanism)
reproduces the current field to 5e-16..6e-15 -- Exchange 16.8x faster,
UniaxialAnisotropy 133x -- but **is invalid when `K2 != 0`** (relative
difference 0.78), because the quartic term makes `dE_dm` cubic in m. The fix
plan's task T4 applies "the same pattern as T3" to `UniaxialAnisotropy` without
noting this; any implementation must gate the fast path on `K2`.

Together the two changes took one right-hand-side evaluation from **96.8 ms to
12.5 ms (7.8x)** at 20787 dofs, and from 2.50 ms to 0.87 ms (2.9x) at 528 dofs,
with no change to any computed value.

Those are single-run figures. Repeating the post-change measurement four times
on the same machine spanned 12.36-14.10 ms (about 13% run-to-run variance), so
read the factor as roughly **7-8x** rather than exactly 7.8x. The conclusion is
insensitive to that spread; precise before/after numbers are the job of the
committed harness the plan's task T1 defines, which does not exist yet.

Those figures exclude demag, which is measured separately below. The
end-to-end `std_prob_3` factor is smaller than any of them and is not yet
re-measured -- the plan's task T7 remains the place where the FULL-lane budgets
and register P1 are closed with end-to-end numbers.

## FK demag is the dominant term (2026-10-09)

Measured once the native build was fixed (see
[`INSTALL.md`](INSTALL.md)), which had blocked demag measurement locally.
Same configuration plus FK demag, serial:

| Mesh | dofs | RHS without demag | with demag | demag share |
|---|---|---|---|---|
| 3x3x10 | 528 | 0.85 ms | 1.92 ms | 56% |
| 6x6x20 | 3087 | 2.06 ms | 4.71 ms | 56% |
| 12x12x40 | 20787 | 12.85 ms | 32.53 ms | 61% |

Inside demag's 19.68 ms at 20787 dofs (2210 boundary nodes):

| Component | Cost | Share of demag |
|---|---|---|
| Krylov solves + lifting (two solves) | 13.61 ms | 69% |
| dense BEM matvec | 2.95 ms | 15% |
| gradient assembly | 1.55 ms | 8% |
| rebuilding a zero-valued `fem.form` per evaluation | 0.43 ms | 2% |
| ordering conversion | 0.06 ms | <1% |

**This caps what the energy-term work can achieve.** A precomputed-matrix fast
path for Exchange and UniaxialAnisotropy addresses ~38% of a realistic
right-hand side, so even reducing those terms to nothing is worth roughly
1.6x end-to-end, not the ~10x the demag-free numbers suggest.

### The zero right-hand side is no longer assembled

The phi_2 Laplace solve has no source term -- its entire right-hand side comes
from `apply_lifting`/`set_bc` writing in the Dirichlet boundary contribution --
yet the port rebuilt and assembled a zero-valued form for it on every
evaluation. At 6929 S1 dofs that measured 270 us to compile the form plus
723 us to assemble the zeros, and it also allocated and freed a ghosted PETSc
vector each time. The vector is now created once at setup and zeroed per solve:

| Mesh | demag per evaluation, before | after |
|---|---|---|
| 3x3x10 | 855 us | **334 us** (2.6x) |
| 6x6x20 | 2608 us | **1975 us** (1.3x) |
| 12x12x40 | 19385 us | **16786 us** (1.15x) |

The gain is largest on small meshes because the removed cost was mostly flat.

Reusing a *ghosted* vector is the one way this could go wrong: clearing only
its owned entries leaves the previous solve's lifting contribution in the ghost
region for the following `ghostUpdate(ADD, REVERSE)` to accumulate. Measured
with owned-only zeroing, that corrupts the demag field by **54%** under two
ranks while a serial run stays clean. FK demag had no multi-rank coverage at
all, so a repeated-solve statelessness check was added to
`energies_dolfinx_mpi_probe.py` (gate `dolfinx-src-energies-mpi`); it was
confirmed to fail with the wrong zeroing and pass with the right one.

### What is left in demag

- **The Krylov solves never warm-start.** `_ksp()` sets no
  `setInitialGuessNonzero`, so both solves begin from a zero guess on every
  evaluation even though consecutive right-hand-side calls change `m` only
  slightly and the previous `phi_1`/`phi_2` would be an excellent guess. This
  is the largest remaining item at 42% of the whole right-hand side, **but it
  is the first change that would not be value-exact**: a solve converged to
  `rtol=1e-6` lands within ~1e-6 of exact whatever its starting point, so
  results shift six orders above the 1e-12 bound the rest of this work holds
  to. It therefore needs an owner decision -- tighten the tolerance so the
  warm-started result is strictly more accurate than today's, or grant
  solver-level changes a different bound.
- **The gradient assembly** (1.55 ms) is linear in `phi`, so the same
  precomputed-operator treatment applies as for the energy terms.
- **The dense BEM matvec** (2.95 ms) is inherent to the dense Fredkin-Koehler
  operator; `TreecodeBEM` already exists as the approximate alternative, so
  this is a user choice rather than an optimisation.

## The fix plan

Detailed task-by-task in
[`plans/2026-07-28-post-sr1-performance.md`](plans/2026-07-28-post-sr1-performance.md).
In summary: rebuild "assemble once, apply per evaluation" as an **internal**
optimisation of the existing `box-assemble` semantics — recovering legacy's
mechanism without resurrecting the legacy method names, which were
deliberately removed by owner decision (register **D32**; `box-matrix-numpy`,
`box-matrix-petsc`, `project` and `direct` still raise `NotImplementedError`
by name after this work). The plan also profiles and, if targets are missed,
considers a compiled RHS backend as a follow-on (the M3 item above).

Every optimisation in that plan must reproduce the existing field values to
a relative difference of at most `1e-12` and leave every oracle-validated
trajectory test green — this is a performance change, not a physics change.

Two example entries were deliberately deferred rather than run to
completion at declaration, specifically because they are the primary
beneficiaries of this planned work: `std_prob_4` (the full 2 ns trace,
budgeted at 38400 s) and `magnetic_grain` (full three-field physics,
budgeted at 21600 s). Re-running both to completion is scheduled after the
fix lands.

## Runner ceiling versus real runtime

GitHub-hosted runners hard-cap every job at 360 minutes (6 hours), which
`test-slow.yml` declares via `timeout-minutes: 360`. The heavy example
lane's actual current runtime is roughly **10-13 hours** at full workload —
well past that ceiling. Until the fix above lands, the full lane cannot
complete on a hosted runner at all; it has to be run locally (see
[`testing.md`](testing.md) for the command), and `test-slow.yml` documents
the tier rather than guaranteeing a finished run.

## Serial time-integration limitation

Time integration in this port is **serial only** — there is no parallel time
-stepping path yet (register **C19**, tracked as future work, not part of
this performance fix). Some individual operations run under MPI (several
focused gates include MPI probes), but advancing simulation time itself
does not currently distribute across ranks. This is a separate limitation
from the assembly-cost problem above: fixing the re-assembly cost will make
each serial step faster, but will not by itself make time stepping
parallel.

## What "fast enough" looks like

The plan's own exit criterion (its Task 7) is to re-measure after the fix,
recalibrate the FULL-lane timeout budgets **downward** from today's measured
values, and close register row P1 with the new numbers — not with a
subjective "feels faster". Until that re-measurement lands, budget example
runs at today's measured rates, not the legacy-era comments in their
headers.
