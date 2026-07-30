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
