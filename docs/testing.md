# Testing

Audience: a contributor running or extending the test suite on branch
`dolfinx-parity`. For install/setup, see [`INSTALL.md`](INSTALL.md); for
where a deviation gets recorded, see
[`acceptance-register.md`](acceptance-register.md).

## The six CI workflows

All six live in `.github/workflows/`. Each installs the `dolfinx` pixi
environment via `prefix-dev/setup-pixi`. Every focused `dolfinx-src-*` gate
and the aggregated `verify-dolfinx-m5` witness run against the *installed*
package (via `dolfinx-install-editable`), not against `src/` on the
`PYTHONPATH` — see [`INSTALL.md`](INSTALL.md) for why.

| Workflow | Trigger | What it runs |
|---|---|---|
| `test-fast.yml` | every push and pull request | `dev/bin/verify-dolfinx-m5` — the 33-gate fast witness (editable install, native build, provenance check, every focused port-gate test incl. MPI probes, the comparison suite, a core Sundials smoke, and the fast examples). ~13 min. This is the everyday green gate. |
| `test-prototypes.yml` | every push and pull request | `dev/bin/verify-dolfinx-m4` — the frozen early-DOLFINx prototype gate (`dev/dolfinx/`), kept as a cheap historical guard. ~1.5 min. |
| `test-python.yml` | weekly (Mondays 03:00 UTC) + manual dispatch | the full `src/finmag` + `examples` suite inventory (`dev/bin/inventory-dolfinx-suite`), gated on the summary line reading `failed=0 errors=0`. |
| `test-slow.yml` | manual dispatch only | the heavy example lane with `FINMAG_EXAMPLE_FULL=1` — all seventeen examples at full workload. |
| `docker-image.yml` | manual dispatch only | pulls and smoke-runs the historical `finmag/finmag` Docker Hub images (python2 era). Dormant relic, kept for manual/historical use. |
| `docker-smoke.yml` | manual dispatch only | builds the pre-pixi Docker image and runs `pytest` inside it (formerly `workflow.yml`). Dormant relic, superseded by the five workflows above. |

**Inert-until-merge-to-main caveat.** GitHub only fires `schedule` and
`workflow_dispatch` triggers from a repository's **default branch**, which
for this repository is `main`, not the `dolfinx-parity` branch this work
happens on. `test-python.yml` (weekly cron + dispatch) and `test-slow.yml`
(dispatch-only) are therefore **inert while sitting on `dolfinx-parity`** —
they will not fire, on schedule or by hand, until this branch is merged into
`main`. Run their local equivalents instead (below) in the meantime.
`test-fast.yml` and `test-prototypes.yml` are unaffected: `push`/`pull_request`
triggers fire on any branch.

`test-slow.yml` carries a `timeout-minutes: 360` — GitHub-hosted runners cap
every job at 360 minutes (6 hours) regardless of what a workflow requests.
The full example lane currently needs roughly 10-13 hours (see
[`performance.md`](performance.md)), so this workflow documents the tier
rather than guaranteeing completion until the performance work lands; run
the full lane locally when you need it to actually finish.

## The milestone ladder (M1-M5)

Milestones mark the port's progress and are still referenced in commit
messages and the register. In plain words, each proved a bigger slice was
real before the next was attempted:

- **M1** — Python 3 viability: builds the old python2/dolfin-2017 Docker
  snapshot and runs the port's most basic checks inside it.
- **M2** — the pixi/conda-forge FEniCS-2019 environment works: import
  finmag, build the `barmini` example, take a time-integration step, and
  save/restart, all against the current source tree.
- **M3** — the legacy test suite ran once on that environment (since
  retired as an in-tree lane; it now only runs at the frozen oracle commit,
  see below).
- **M4** — DOLFINx itself is viable: a frozen early gate over the
  `dev/dolfinx/` prototypes (environment versions, import, a smoke test, the
  prototype pytest suite, and two runnable examples). This is
  `test-prototypes.yml` today.
- **M5** — the whole port is proven: the 33-gate aggregate — editable
  install, native build, provenance check, every focused source test gate
  including MPI probes, the comparison suite, a core Sundials smoke, and the
  fast examples. This is `test-fast.yml` today, run via
  `dev/bin/verify-dolfinx-m5`.

## The full-suite sweep and current tally

`dev/bin/inventory-dolfinx-suite` (equivalently
`pixi run -e dolfinx dolfinx-src-suite-inventory`) collects and runs
**every** test under `src/finmag` and `examples`, including tests for
capabilities that have not been ported yet. It takes roughly 30 minutes and
prints one summary line starting `INVENTORY:`.

**This is not a pass/fail verdict** — the 33-gate `verify-dolfinx-m5` is the
verdict. The sweep exists to keep the remaining porting work visible instead
of silently dropping untested master files. `test-python.yml` is the CI
wrapper around this sweep; its gate is that the summary line reads
`failed=0 errors=0`, not that every test passes.

Current tally:

```
INVENTORY: passed=769 failed=0 errors=0 skipped=46 xfailed=271
```

What each number means:

- **passed (769)** — tests that ran and confirmed the ported behaviour.
- **failed (0)** — tests that ran and did not confirm expected behaviour.
  The sweep is gated on this staying zero.
- **errors (0)** — tests that could not even be collected or set up (for
  example, an import failure). Also gated at zero.
- **skipped (46)** — tests master itself marks conditional (an unavailable
  display, an optional external tool, etc.); master's own skip decision is
  left untouched and governs the outcome.
- **xfailed (271)** — tests for a capability that genuinely has not been
  ported yet, expected to fail and marked as such (see "strict xfail"
  below). This is the visible parity backlog, not a defect count.

## The frozen legacy oracle lane

There is no more in-tree legacy (python2/dolfin-2017) test lane. Instead,
`dev/bin/run-legacy-oracle` runs one command against an immutable, frozen
Python-3/FEniCS-2019 commit (`ba9280934e188d7f3800e7b9865e70a9422f7687`),
which carries its own copy of every file it needs:

```bash
dev/bin/run-legacy-oracle -- pixi run --locked barmini-suite
```

This is what "oracle-validated" means elsewhere in these docs: a result was
actually computed at that frozen commit and compared, not just claimed. The
oracle checkout predates this repository's `pyproject.toml` packaging
change and does not need it — the two are unaffected by each other.

## How strict xfail governs unported tests

Where a master test exercises a capability that has not been ported yet, the
test function is carried over **verbatim** under a `NOT PORTED` banner
rather than deleted or silently skipped, and marked:

```python
@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: <feature> (register <row>)", strict=True)
```

`strict=True` means an *unexpected pass* fails the gate. So the day someone
actually ports the feature, the test flips from xfailed to passing and CI
turns red until the now-stale marker is removed — nothing can quietly stay
"carried" forever once the underlying work is done. Master's own
pre-existing `xfail`/`skipif` markers, where a test already had one, are
left untouched and continue to govern that test's outcome instead of gaining
a second marker.

The `not_ported` marker is declared in `pytest.ini` as a selection label, not
a skip: `-m "not not_ported"` can filter these out of a gate that wants only
currently-supported coverage, but the default full-suite sweep above runs
everything unfiltered.

## Local run commands

Verified against `pixi.toml`, `.github/workflows/`, and `dev/bin/`:

```bash
# Fast gate (test-fast.yml equivalent, ~13 min)
dev/bin/verify-dolfinx-m5

# Frozen prototype gate (test-prototypes.yml equivalent, ~1.5 min)
dev/bin/verify-dolfinx-m4

# Full-suite inventory (test-python.yml equivalent, ~30 min)
dev/bin/inventory-dolfinx-suite
# equivalently:
pixi run -e dolfinx dolfinx-src-suite-inventory

# Heavy example lane at full workload (test-slow.yml equivalent, ~10-13 h)
pixi run -e dolfinx dolfinx-install-editable
pixi run -e dolfinx dolfinx-native-build
pixi run -e dolfinx dolfinx-provenance-check
FINMAG_EXAMPLE_FULL=1 pixi run -e dolfinx dolfinx-src-examples-pytest

# Frozen legacy oracle, any command
dev/bin/run-legacy-oracle -- pixi run --locked barmini-suite
```

Any single focused gate can also be run directly — see the `dolfinx-src-*`
tasks in `pixi.toml` (for example `pixi run -e dolfinx
dolfinx-src-energies-pytest`).
