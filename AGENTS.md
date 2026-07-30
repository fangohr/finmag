# Agent conventions for this repository

Audience: agentic/automated work on `finmag`, branch `dolfinx-parity` (the
DOLFINx master-parity port). For human-contributor setup and workflow, see
[`CONTRIBUTING.md`](CONTRIBUTING.md); for the documentation index, see
[`docs/README.md`](docs/README.md). The historical agent progress log that
used to live in this file (Python-3/pixi/DOLFINx-M1..M4 narrative) is now
[`docs/archive/agents-progress-log.md`](docs/archive/agents-progress-log.md).

## Branch-assert guard

Before doing any work, confirm the branch and commit you were actually
dispatched onto:

```sh
git branch --show-current
git rev-parse HEAD
```

If either does not match what the task explicitly names, stop and report
"blocked" rather than proceeding on the wrong checkout. Do not silently
`checkout`/`switch` to "fix" a mismatch unless told to.

## The `sr1` tag is immovable

`sr1` (`e228e5f3`) marks "Support Release 1" — the point where the owner
declared the first supported subset of the DOLFINx port, documented in
[`docs/SUPPORTED.md`](docs/SUPPORTED.md). Never move, delete, or re-point
this tag, and never force-push over the commit it names. Work after `sr1`
continues on `dolfinx-parity` toward full parity with original `master`
(tag `python2`, commit `b5015c5a`); see
[`dolfinx-transition.md`](dolfinx-transition.md) for the full tag ladder.

## `dev/benchmarks/` is a stopped, uncommitted probe

`dev/benchmarks/` is untracked and holds a **stopped PERF-T1 partial** from
before an owner stop-order issued 2026-07-28. Do not resume the performance
work it represents and do not commit the directory. The performance plan
(`docs/plans/2026-07-28-post-sr1-performance.md`, register `P1`) is planned,
not started; it begins only on an explicit owner instruction.

## Safe execution protocol

- Use the frozen Python-3/FEniCS-2019 oracle at commit `ba928093` through
  `dev/bin/run-legacy-oracle` (e.g. `dev/bin/run-legacy-oracle -- pixi run
  --locked barmini-suite`) for cross-checks against the pre-port legacy
  behaviour. Prefer analytic physics where it is stronger, and label
  cross-method checks honestly. There is no in-tree legacy test lane.
- Probe uncertain DOLFINx mechanics under `dev/dolfinx` only, for bounded
  mechanics probes and reference generation; do not create a duplicate
  `finmag` package under `dev`. After a probe and its legacy contract are
  validated, port the capability by editing the existing `src/finmag` module
  directly, with a focused in-place test diff — do not copy prototype
  modules into `src` or add a dual-backend facade.
- For every slice: state the API and scientific invariant, obtain RED
  evidence, make the smallest source change, run the focused and aggregate
  gates (`dev/bin/verify-dolfinx-m5`), update the capability/decision
  documents, and obtain an independent review.
- Preserve public names and defaults unless forced or owner-approved. Every
  unavailable current API must fail by feature name (see the strict-xfail
  convention in `CONTRIBUTING.md`), not through an incidental import or
  unrelated attribute error.
- Do not combine unrelated cleanup with a parity fix. Do not dispatch new
  implementation work until the user chooses the next slice.
- Before running the legacy oracle, inspect `git worktree list` for stale
  disposable oracle worktrees under `/tmp/finmag-legacy-oracle.*`; they are
  not project state — remove them only after confirming no needed probe
  output remains in them.
- Behavioural deviations from `master` always get an acceptance-register row
  before merge (`docs/acceptance-register.md`); see
  `CONTRIBUTING.md` for the full rule.

## Model-usage note

- Do not assume a fixed workstream-wide model label; tools frequently report
  an inaccurate or stale app/model name, and different sessions may run
  different underlying agents/models entirely.
- Sign new agent-authored comments and commit-message attribution with your
  own actual current identity as you understand it, not a value carried
  over from a previous session or another tool.
- If genuinely unsure of the current model, ask the user rather than
  guessing or reusing an old label. Older labels found in comments (e.g.
  `[Codex GPT-5.4]`, `[Codex gpt-5.5 high]`) are historical and should not be
  copied forward by new sessions.
- Escalate model choice to the task at hand (harder review/plan work
  generally warrants a stronger model) rather than defaulting uniformly.

## Commit-trailer convention

End agent-authored commits with a co-author trailer naming your actual
current model identity, matching this repository's existing history, e.g.:

```
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
```

Add a `Claude-Session: <url>` trailer too when a session URL is available.
Do not carry a trailer's model name forward from an earlier commit if your
own identity differs.
