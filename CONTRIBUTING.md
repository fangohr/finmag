# Contributing to finmag (DOLFINx port)

Audience: a human contributor working on `src/finmag` on branch
`dolfinx-parity`. For repo conventions aimed at agentic/automated work, see
[`AGENTS.md`](AGENTS.md).

## Environment setup

See [`docs/INSTALL.md`](docs/INSTALL.md) for the pixi install, editable
install, native build and the one-command verifier
(`dev/bin/verify-dolfinx-m5`, expect 33/33). Do that first; everything below
assumes a working `dolfinx` pixi environment.

## Test lanes — when to run which

Full detail, including the inventory lane and the current pass/fail tally, is
in `docs/testing.md` (arrives with the restructure). The short version:

- **Fast gate** — `dev/bin/verify-dolfinx-m5` (33 focused port gates, ~13
  min). Run this before every push/PR; it is `test-fast.yml`, the everyday
  green gate.
- **Full sweep** — `dev/bin/inventory-dolfinx-suite` (the whole `src/finmag`
  tree, ~30 min). This is the **non-gating parity backlog**, not a verdict —
  see `docs/SUPPORTED.md` §7. Its CI equivalent, `test-python.yml`, runs
  weekly and on demand and must come back `failed=0 errors=0`.
- **Heavy examples** — `FINMAG_EXAMPLE_FULL=1 pixi run -e dolfinx
  dolfinx-src-examples-pytest`, run on demand (`test-slow.yml` in CI,
  long-running by design; GitHub scheduling only fires from the default
  branch, so this workflow is inert on `dolfinx-parity` until merged — run
  the local command instead).
- **Legacy oracle** — `dev/bin/run-legacy-oracle` compares against the frozen
  Python-3/FEniCS-2019 commit; there is no in-tree legacy test lane.

## How to add a test

- **Canonical master-path convention.** A ported test lives at its original
  `master` (`b5015c5a`) file path, not beside it as a `*_dolfinx.py` sibling.
  This makes `git diff b5015c5a..HEAD -- <path>` the review diff for "is this
  port faithful to master?" — always port or add a test in place at the
  master path, never create a new sibling file.
- **Strict-xfail rule for unported things.** If a master test exercises a
  capability that has not been ported yet, carry the test function verbatim
  under a `NOT PORTED` banner, mark it `@pytest.mark.not_ported`, and add
  `@pytest.mark.xfail(reason="not ported: <feature> (register <row>)",
  strict=True)`. `strict=True` means an unexpected pass fails the gate, so
  finishing the port forces the marker's removal — it cannot go stale
  silently. Do not delete or skip a master test just because its feature
  isn't ported; carry it and mark it.
- **No-tolerance-loosening rule.** Never widen a test tolerance (numerical or
  solver) to make it pass. If the DOLFINx port's value legitimately differs
  from the legacy value, match the legacy value exactly and document why, or
  — if it cannot be matched — record the divergence as a register row (see
  below) rather than loosening the check.

## Recording a behavioural deviation

**Every behavioural deviation from the python2 `master` gets an
acceptance-register row *before* merge.** The register at
[`docs/superpowers/acceptance-register.md`](docs/superpowers/acceptance-register.md)
is the sole owner-decision ledger (D-rows for divergences, M-rows for missing
capabilities, P-rows for performance); only the repository owner finalises a
row's disposition, and a recommendation without one is not an approval to
merge. Read `docs/decisions.md` (arrives with the restructure) for the
deeper "what we chose during the port, and why" narrative behind the
accepted rows, and
[`docs/superpowers/capability-status.md`](docs/superpowers/capability-status.md)
for the current per-capability status matrix these decisions feed.

## More documentation

- [`docs/README.md`](docs/README.md) — the documentation index and reading
  paths.
- [`docs/SUPPORTED.md`](docs/SUPPORTED.md) — what is supported today, and to
  what tolerance.
- [`dolfinx-transition.md`](dolfinx-transition.md) — the port story: tags,
  milestones, workflows.
