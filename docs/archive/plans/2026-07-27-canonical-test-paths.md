> **ARCHIVED (2026-07-29).** Historical record of the porting process; statements reflect their writing date. Current truth: docs/README.md.

# Canonical Test Paths (retire the `*_dolfinx.py` sibling convention) Implementation Plan

**STATUS: COMPLETE (2026-07-27).** All 8 tasks done; every step box below is
ticked. Range `e10c5893` (this plan) .. `62aae519` (Task 6), nine commits, plus
the Task-8 documentation commit. Post-move `dev/bin/verify-dolfinx-m5` is 33/33
green with every per-gate count identical to the pre-move measured baselines;
the inventory lane reported `INVENTORY: passed=752 failed=34 errors=59
skipped=25 xfailed=12` on 2026-07-27. Owner decision recorded as acceptance
register **D30**; one new parity-debt row **D31** opened. Authoritative
narrative: `docs/superpowers/HANDOVER.md`, "Canonical test paths, the legacy
oracle lane, and the inventory lane".

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every ported DOLFINx test onto its original master (`b5015c5a`)
file path so `git diff b5015c5a..HEAD -- <path>` shows the review diff
directly; run the ENTIRE suite (ported + unported) in a new inventory lane so
unported tests fail visibly; retire the in-tree legacy FEniCS-2019 test lane in
favour of the frozen oracle commit `ba9280934e188d7f3800e7b9865e70a9422f7687`.

**Architecture:** Owner decision 2026-07-27 (supersedes the sibling convention
ratified 2026-07-25). Three mechanisms: (1) per-file `git rm` master +
`git mv` port onto the master path, with any master functions the port does not
cover appended verbatim under a `NOT PORTED` banner and marked
`@pytest.mark.not_ported` so focused gates stay green while the full lane
reports them as failures; (2) a non-gating `dev/bin/inventory-dolfinx-suite`
script that runs `pytest src/finmag` with `--continue-on-collection-errors` and
tallies pass/fail/collection-error as the parity inventory; (3) deletion of the
in-tree legacy suite task (`barmini-suite`) and its `verify-python3-*` callers —
legacy execution becomes `dev/bin/run-legacy-oracle -- pixi run --locked
barmini-suite` at the frozen commit (verified: `ba928093` carries its own
`barmini-suite` task at its pixi.toml line 69 and its own copies of every test
file).

**Tech Stack:** git, pixi (env `dolfinx`), pytest, bash. Aggregate gate:
`dev/bin/verify-dolfinx-m5` (33 steps, all green before AND after).

## Global Constraints

- Branch: `dolfinx-parity`. Master reference commit: `b5015c5a`. Oracle commit:
  `ba9280934e188d7f3800e7b9865e70a9422f7687`.
- NEVER modify `src/finmag` implementation code in this plan. Test files,
  `pixi.toml`, `dev/bin`, and docs only.
- No coverage may silently vanish: a master test function may only disappear
  from the tree if the port file's traceability header accounts for it as
  covered. Uncovered functions are carried under the `NOT PORTED` banner
  (bucket A/partial files) or the whole master file stays in the tree
  (bucket B ancestors not fully covered, and all never-ported master tests).
- Focused `dolfinx-src-*-pytest` gates must be green after every task.
  Deselect carried not-ported tests with `-m "not not_ported"`; do NOT use
  xfail/skip for them (the owner wants them REPORTED as failures in the full
  lane, not hidden).
- Every moved file keeps its traceability header/docstring. Tolerances,
  assertions and xfail pins must not change in this plan — this is a re-layout,
  not a test rewrite.
- Each task ends in its own commit. Commit messages follow the repo style:
  one-line summary + body, `Co-Authored-By:` trailer for the executing model.
- The in-file traceability header is AUTHORITATIVE over this plan's table. If a
  header contradicts a target path or coverage claim below, follow the header
  and record the discrepancy in the task report.

---

## Canonical move table

Buckets per the P5.2 conversion phase. "→ path" = target after the move.
Verify each against the file's own header before moving.

### Bucket A — port takes the master path

| Port (current) | Target (master path) | Partial? |
|---|---|---|
| `src/finmag/tests/test_thin_film_demag_dolfinx.py` | `src/finmag/energies/thin_film_demag_test.py` | no |
| `src/finmag/tests/test_llg_dolfinx.py` | `src/finmag/tests/test_llg.py` | no |
| `src/finmag/tests/test_cubic_anisotropy_dolfinx.py` | `src/finmag/energies/cubic_anisotropy_test.py` | no |
| `src/finmag/tests/test_field_dolfinx.py` | `src/finmag/field_test.py` | verify header |
| `src/finmag/tests/test_simulation_dolfinx.py` | `src/finmag/sim/sim_test.py` | YES — carry uncovered `sim_test.py` fns (e.g. `test_clean_up`/shutdown surface, D24) |
| `src/finmag/tests/test_hysteresis_dolfinx.py` | `src/finmag/sim/hysteresis_test.py` | no |
| `src/finmag/tests/test_magnetisation_patterns_dolfinx.py` | `src/finmag/sim/magnetisation_patterns_test.py` | no |
| `src/finmag/tests/test_plot_helpers_dolfinx.py` | `src/finmag/util/plot_helpers_test.py` | verify header (ndt-columns xfail pin D27 stays) |
| `src/finmag/tests/test_util_helpers_dolfinx.py` | `src/finmag/util/helpers_test.py` | YES — 16/29 transcribed; carry the 13 uncovered fns |
| `src/finmag/tests/test_timezeeman_dolfinx.py` | `src/finmag/energies/zeeman_test.py` | YES — 9/14 transcribed; carry only fns not covered here NOR in the energies aggregate (cross-check both headers) |
| `src/finmag/tests/test_fk_demag_dolfinx.py` | `src/finmag/energies/demag/fk_demag_test.py` | verify header |
| `src/finmag/tests/test_treecode_pbc_demag_dolfinx.py` | `src/finmag/energies/demag/demag_pbc_test.py` | verify header (D17 strict xfails stay) |
| `src/finmag/tests/comparison/anisotropy/test_anis_magpar_dolfinx.py` | `src/finmag/tests/comparison/anisotropy/test_anis_magpar.py` | no (D29 note stays) |
| `src/finmag/tests/comparison/demag/test_demag_magpar_dolfinx.py` | `src/finmag/tests/comparison/demag/test_demag_field.py` | verify header |
| `src/finmag/tests/comparison/exchange/test_exchange_compare_magpar_dolfinx.py` | `src/finmag/tests/comparison/exchange/test_exchange_compare_magpar.py` | no |
| `src/finmag/tests/comparison/exchange/test_exchange_field_nmag_dolfinx.py` | `src/finmag/tests/comparison/exchange/test_exchange_field.py` | YES — master's `test_against_oommf` is NOT ported (N62); carry it |
| `src/finmag/tests/nmag/anisotropy_1d/test_nmag_1d_anisotropy_dolfinx.py` | `src/finmag/tests/nmag/anisotropy_1d/test_nmag_1d_anisotropy.py` | no |
| `src/finmag/tests/nmag/exchange_1d/test_exchange_1d_dolfinx.py` | `src/finmag/tests/nmag/exchange_1d/test_exchange_1d.py` | no |

Special case (split ancestor): `src/finmag/tests/comparison/demag/test_demag_field_dolfinx.py`
is a transcription of `fk_demag_test.py` sphere-analytic tests placed in the
comparison tree (its ancestor path is taken by `test_fk_demag_dolfinx.py`).
Rename it in place to
`src/finmag/tests/comparison/demag/test_demag_sphere_analytic.py`; its header
already explains the split.

### Bucket B — aggregate ports: rename in place (drop `_dolfinx`), delete ONLY fully-covered ancestors

| Port (current) | Target name | Ancestors (delete iff header accounts EVERY fn as covered; else ancestor STAYS in tree) |
|---|---|---|
| `src/finmag/tests/test_energies_dolfinx.py` | `src/finmag/tests/test_energies.py` | `energies/exchange_test.py`, `energies/anisotropy_test.py`, `energies/zeeman_test.py` (statics only — timezeeman task owns the file), `energies/magnetostatic_field_test.py` (flagged unwired/untested — likely STAYS), `energies/test_energies_in_regions.py` |
| `src/finmag/tests/test_effective_field_dolfinx.py` | `src/finmag/tests/test_effective_field.py` | `tests/test_effective_field.py` + second ancestor per header |
| `src/finmag/tests/test_dmi_dolfinx.py` | `src/finmag/tests/test_dmi.py` | `energies/dmi_test.py`, `tests/test_dmi.py`, `tests/test_dmi_terms.py` |
| `src/finmag/tests/test_fk_demag_dolfinx.py` | (bucket A above) | also accounts `energies/demag/fk_demag_2d_test.py` (2D is deferred D-row — likely STAYS) |
| `src/finmag/tests/test_meshes_dolfinx.py` | `src/finmag/tests/test_meshes.py` | `util/meshes_test.py`, `util/mesh_templates_test.py`, `tests/test_meshes.py` (12 deferred helpers uncovered — those ancestors likely STAY) |
| `src/finmag/tests/test_restart_output_dolfinx.py` | `src/finmag/tests/test_restart_output.py` | 4 ancestors per header (`test_restart_simulation.py`, `test_writing_data.py`, …) — header notes all four "still physically present"; delete the fully-covered ones |
| `src/finmag/tests/test_scipy_driver_dolfinx.py` | `src/finmag/drivers/tests/test_scipy.py` | `drivers/tests/test_scipy.py` + per header |
| `src/finmag/tests/test_sundials_driver_dolfinx.py` | `src/finmag/tests/test_sundials_driver.py` | `drivers/tests/sundials_nsteps_test.py`, `drivers/tests/sundials_reinit_test.py`, `util/ode/tests/test_sundials_ode.py`, `util/ode/tests/test_sundials_stiff_ode.py` |
| `src/finmag/tests/test_stt_dolfinx.py` | `src/finmag/tests/test_stt.py` | 4 STT ancestors per header table (`zhangli/zhang_li_test.py`, …) |
| `src/finmag/tests/test_variable_params_dolfinx.py` | `src/finmag/tests/test_variable_params.py` | `test_spatially_varying_alpha.py`, `test_spatially_varying_anisotropy.py`, `test_energy_creation_with_variable_Ms.py`, `energies/test_energies_in_regions.py` (shared with energies — delete once, only if covered) |
| `src/finmag/tests/test_io_utils_dolfinx.py` | `src/finmag/tests/test_io_utils.py` | mixed; `tests/test_skyrmions.py` is a stub per header — delete iff accounted |

### Bucket C — genuinely new: rename only (drop `_dolfinx`)

`test_magpar_io_dolfinx.py` → `test_magpar_io.py` (same dir);
`test_deferred_surfaces_dolfinx.py` → `test_deferred_surfaces.py`;
`test_dw_fixed_energy_dolfinx.py` → `test_dw_fixed_energy.py`;
`test_example_dolfinx.py` → `test_example.py`;
`test_field_hdf5_dolfinx.py` → `test_field_hdf5.py`;
`test_native_bem_arrays_dolfinx.py` → `test_native_bem_arrays.py`;
`test_ordering_contract_dolfinx.py` → `test_ordering_contract.py`;
`examples/test_examples_dolfinx.py` → `examples/test_examples.py`.

Never-ported master test files (≈95, e.g. `physics/llb/sllg_test.py`,
`physics/tests/neb/neb_test.py`, `tests/test_skyrmions.py` family, oommf
suite): UNTOUCHED. They stay at their paths and fail in the inventory lane —
that is the visible backlog, by owner intent.

---

### Task 1: `not_ported` marker + full-suite inventory lane (baseline BEFORE any moves)

**Files:**
- Create: `pytest.ini` (repo root) — only if no `[tool.pytest.ini_options]` /
  `[pytest]` config exists yet (check `pyproject.toml`, `setup.cfg`, `tox.ini`
  first; if one exists, add the `markers` key there instead)
- Create: `dev/bin/inventory-dolfinx-suite`
- Modify: `pixi.toml` (new task in `[feature.dolfinx.tasks]`)

**Interfaces:**
- Produces: pytest marker `not_ported` (later tasks mark carried master tests
  with it); pixi task `dolfinx-src-suite-inventory`; script
  `dev/bin/inventory-dolfinx-suite` printing a final line
  `INVENTORY: passed=<n> failed=<n> errors=<n> skipped=<n> xfailed=<n>` and
  exiting 0 on pytest exit codes 0/1 (test failures are DATA here), non-zero on
  pytest internal/usage errors (codes ≥2).

- [x] **Step 1: Register the marker.** Find existing pytest config
  (`grep -rn "pytest" pyproject.toml setup.cfg tox.ini pytest.ini 2>/dev/null`).
  Add:

```ini
[pytest]
markers =
    not_ported: master test carried verbatim; functionality not yet ported to DOLFINx — expected to fail in the inventory lane
```

  (or the `pyproject.toml` equivalent). Verify placement does not change pytest
  rootdir for the existing gates: run
  `pixi run -e dolfinx dolfinx-src-field-pytest` and confirm the same pass
  count as before (42 passed).

- [x] **Step 2: Write `dev/bin/inventory-dolfinx-suite`** (mode 755), matching
  the style of the existing `dev/bin/verify-*` scripts:

```bash
#!/usr/bin/env bash
# Non-gating parity inventory: run EVERY test under src/finmag in the DOLFINx
# environment, including unported master tests. Failures and collection errors
# are the parity backlog, not a CI verdict. Owner decision 2026-07-27.
set -uo pipefail
repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse --show-toplevel)"
cd "${repo_root}"
log="${TMPDIR:-/tmp}/finmag-suite-inventory.$(date +%Y%m%d-%H%M%S).log"
pixi run -e dolfinx dolfinx-src-suite-inventory 2>&1 | tee "${log}"
status=${PIPESTATUS[0]}
if (( status > 1 )); then
    echo "ERROR: pytest infrastructure failure (exit ${status}); inventory not valid." >&2
    exit "${status}"
fi
summary="$(grep -E '^(=+ .*(passed|failed|error).* =+)$' "${log}" | tail -1)"
echo "INVENTORY: ${summary:-no summary found} (log ${log})"
```

- [x] **Step 3: Add the pixi task** next to the other `dolfinx-src-*` tasks,
  same env-var style as its neighbours:

```toml
# Full-suite parity inventory: collect and run EVERYTHING, including unported
# master tests. NON-GATING — failures here are the parity backlog inventory
# (owner decision 2026-07-27). Gated lanes stay the dolfinx-src-*-pytest tasks.
dolfinx-src-suite-inventory = "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q src/finmag examples --continue-on-collection-errors -m \"not not_ported or not_ported\" -p no:cacheprovider"
```

  Note: `-m "not not_ported or not_ported"` is a no-op selector — write it
  WITHOUT any `-m` filter instead (the inventory runs everything); the snippet
  above shows the temptation to avoid. Final task line:

```toml
dolfinx-src-suite-inventory = "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q src/finmag examples --continue-on-collection-errors -p no:cacheprovider"
```

- [x] **Step 4: Run the baseline inventory:** `dev/bin/inventory-dolfinx-suite`.
  Expected: script exits 0; MANY failures/collection errors (unported master
  files import `dolfin`). Record the INVENTORY line — this is the pre-move
  baseline. If pytest aborts with a duplicate-basename collection error
  (`import file mismatch`), fix by adding the missing `__init__.py` to the
  offending test directory (several test dirs already have them) and note which.
- [x] **Step 5: Confirm gates unaffected:** run
  `pixi run -e dolfinx dolfinx-src-import-pytest` (expect 28 passed as before).
- [x] **Step 6: Commit** `pixi.toml`, the config file, the script:
  `git commit -m "Add non-gating full-suite parity inventory lane"`.

### Task 2: Move the comparison + nmag group onto master paths

**Files:**
- Modify: `pixi.toml` (`dolfinx-src-comparison-pytest` file list)
- Rename/delete per bucket-A table rows: `test_anis_magpar`,
  `test_demag_magpar`→`test_demag_field.py`, `test_demag_field`→
  `test_demag_sphere_analytic.py`, `test_exchange_compare_magpar`,
  `test_exchange_field_nmag`→`test_exchange_field.py` (carry
  `test_against_oommf`), `test_nmag_1d_anisotropy`, `test_exchange_1d`,
  and bucket-C `test_magpar_io`.

**Interfaces:**
- Consumes: `not_ported` marker (Task 1).
- Produces: comparison tests at master paths; gate
  `dolfinx-src-comparison-pytest` green with `-m "not not_ported"`.

- [x] **Step 1:** For each file: read its traceability header; then
  `git rm <master original>` (where the port takes its path) and
  `git mv <port> <target>`. Example:

```bash
git rm src/finmag/tests/comparison/anisotropy/test_anis_magpar.py
git mv src/finmag/tests/comparison/anisotropy/test_anis_magpar_dolfinx.py \
       src/finmag/tests/comparison/anisotropy/test_anis_magpar.py
```

- [x] **Step 2: Carry the OOMMF gap.** Into the moved
  `test_exchange_field.py`, append master's `test_against_oommf` verbatim from
  `git show b5015c5a:src/finmag/tests/comparison/exchange/test_exchange_field.py`
  under:

```python
# ===== NOT PORTED (carried verbatim from master b5015c5a; expected to fail) =====
# finmag.util.oommf is not ported (manifest N62, capability C20). Runs — and
# fails — in the inventory lane so the gap stays visible. [owner 2026-07-27]
```

  with `@pytest.mark.not_ported` on the function. If the master module-level
  imports it needs (`from finmag.util.oommf import ...`) fail under dolfinx,
  guard them:

```python
try:
    from finmag.util.oommf import oommf_uniform_exchange, mesh_spec  # not ported
except ImportError:
    oommf_uniform_exchange = mesh_spec = None  # not_ported tests fail visibly below
```

- [x] **Step 3: Fix `__file__`-relative paths.** Files that MOVED directory
  (none in this group — all moves are within-directory renames except verify)
  must have fixture/reference-data paths re-checked:
  `grep -n "__file__\|MODULE_DIR\|dirname" <each moved file>`.
- [x] **Step 4: Repoint the gate.** In `pixi.toml`, update
  `dolfinx-src-comparison-pytest`'s file list to the new paths and append
  `-m "not not_ported"`.
- [x] **Step 5: Run** `pixi run -e dolfinx dolfinx-src-comparison-pytest`.
  Expected: 20 passed (same as before the move; the carried oommf test is
  deselected).
- [x] **Step 6:** `grep -rn "_dolfinx" pixi.toml dev/bin src/finmag/tests/comparison src/finmag/tests/nmag`
  — no stale references to the moved names remain (docs references are handled
  in Task 8).
- [x] **Step 7: Commit:**
  `git commit -m "Move comparison/nmag tests onto master paths (canonical-paths owner decision)"`.

### Task 3: Move the energies/demag group onto master paths

**Files:**
- Modify: `pixi.toml` gates: `dolfinx-src-energies-pytest`,
  `dolfinx-src-timezeeman-pytest`, `dolfinx-src-cubicanis-pytest`,
  `dolfinx-src-dmi-pytest`, `dolfinx-src-demag-pytest`,
  `dolfinx-src-treecode-pytest`, `dolfinx-src-effectivefield-pytest`,
  `dolfinx-src-varparams-pytest`
- Move per table: `thin_film_demag`, `timezeeman`→`zeeman_test.py`,
  `cubic_anisotropy`, `fk_demag`, `treecode`→`demag_pbc_test.py`; rename
  bucket-B `energies`, `dmi`, `effective_field`, `variable_params`; delete
  fully-covered ancestors per header, KEEP not-fully-covered ancestors
  (`magnetostatic_field_test.py` expected to stay; `fk_demag_2d_test.py`
  expected to stay).

**Interfaces:**
- Consumes: `not_ported` marker.
- Produces: energies/demag tests at canonical paths; all eight gates green.

- [x] **Step 1:** Bucket-A moves (`git rm` master, `git mv` port) for
  thin_film_demag, cubic_anisotropy, fk_demag, treecode→demag_pbc, timezeeman→
  zeeman_test. These moves CHANGE DIRECTORY (`src/finmag/tests/` →
  `src/finmag/energies[/demag]/`): after each move run
  `grep -n "__file__\|dirname\|fixtures" <file>` and repair any relative
  fixture path (oracle fixtures live under `src/finmag/tests/fixtures/` — use a
  path anchored on the new location).
- [x] **Step 2:** For `zeeman_test.py` (partial 9/14): cross-check the
  timezeeman header AND the energies header; append only master fns covered by
  NEITHER, verbatim under the `NOT PORTED` banner with `@pytest.mark.not_ported`
  and guarded imports (same pattern as Task 2 Step 2).
- [x] **Step 3:** For `sim_test.py`-style partials in this group: none expected;
  verify via headers.
- [x] **Step 4:** Bucket-B renames: `test_energies_dolfinx.py`→
  `test_energies.py`, `test_dmi_dolfinx.py`→`test_dmi.py` (replaces master
  `tests/test_dmi.py` iff fully covered — else name it `test_dmi_port.py` and
  keep the master; follow the header), `test_effective_field_dolfinx.py`→
  `test_effective_field.py` (replaces master iff covered),
  `test_variable_params_dolfinx.py`→`test_variable_params.py`. For every
  ancestor the header accounts as FULLY covered: `git rm` it. For every
  ancestor with uncovered fns: leave it in the tree untouched and list it in
  the commit message as retained backlog.
- [x] **Step 5:** Update all eight gate file lists in `pixi.toml`; append
  `-m "not not_ported"` to any gate whose files now carry banner tests.
- [x] **Step 6:** Run each of the eight gates; expected pass counts unchanged
  from their last recorded values (energies 45, timezeeman 34, cubicanis 30,
  demag 18, treecode 22 passed/1 xfailed, varparams 35, effectivefield and dmi
  per their last green runs — read the exact counts from the most recent
  `verify-dolfinx-m5` log or re-derive by running before the move).
- [x] **Step 7: Commit:**
  `git commit -m "Move energies/demag tests onto master paths"`.

### Task 4: Move the core-simulation group onto master paths

**Files:**
- Modify: `pixi.toml` gates: `dolfinx-src-field-pytest`,
  `dolfinx-src-llg-pytest`, `dolfinx-src-simulation-pytest`,
  `dolfinx-src-scipy-pytest`, `dolfinx-src-sundials-pytest`,
  `dolfinx-src-stt-pytest`, `dolfinx-src-restart-output-pytest`,
  `dolfinx-src-meshes-pytest`, `dolfinx-src-io-utils-pytest`,
  `dolfinx-src-import-pytest` (dolfinx variant), `dolfinx-src-timezeeman-pytest`
  (hysteresis file if listed there)
- Moves per table: `field`→`field_test.py`, `llg`→`tests/test_llg.py`,
  `simulation`→`sim/sim_test.py` (carry uncovered fns), `hysteresis`→
  `sim/hysteresis_test.py`, `magnetisation_patterns`→
  `sim/magnetisation_patterns_test.py`, `plot_helpers`→
  `util/plot_helpers_test.py`, `util_helpers`→`util/helpers_test.py` (carry 13
  uncovered fns); bucket-B renames: `stt`, `scipy_driver`→
  `drivers/tests/test_scipy.py`, `sundials_driver`, `restart_output`, `meshes`,
  `io_utils` — ancestors deleted iff fully covered, kept otherwise.

**Interfaces:**
- Consumes: `not_ported` marker.
- Produces: core tests at canonical paths; all listed gates green.

- [x] **Step 1:** Bucket-A moves with directory changes (`tests/` → `sim/`,
  `util/`, package root): after each, repair `__file__`-relative fixture paths
  (`src/finmag/tests/fixtures/…` oracles are heavily used by llg/simulation
  files) and re-check any `sys.path` or package-relative import.
- [x] **Step 2:** `sim_test.py` partial: from the header's accounting, append
  master fns not covered anywhere (shutdown/instance surface — D24;
  `from finmag.example.normal_modes import disk` import-based tests) verbatim
  under the `NOT PORTED` banner + `@pytest.mark.not_ported` + guarded imports.
- [x] **Step 3:** `helpers_test.py` partial: append the 13 master fns the
  header lists as not transcribed (module-scope `import dolfin` in
  `finmag.util.helpers` — guard the import as in Task 2 Step 2 so the file
  still collects under dolfinx).
- [x] **Step 4:** Bucket-B renames + ancestor deletion/retention per header
  (same procedure as Task 3 Step 4). `test_meshes` retains ancestors covering
  the 12 deferred mesh helpers if the header marks them uncovered.
- [x] **Step 5:** Update every affected gate list; add `-m "not not_ported"`
  where banner tests were introduced. Also update the two references to
  `src/finmag/tests/test_example_dolfinx.py` ONLY IF Task 5 has not run yet —
  otherwise skip (Task 5 owns that rename).
- [x] **Step 6:** Run each affected gate; pass counts must match their last
  recorded green values (field 42, llg 23, simulation 61, meshes 53, import 28,
  varparams 35, stt 27, etc. — read exact numbers from the latest
  `verify-dolfinx-m5` log before starting).
- [x] **Step 7: Commit:**
  `git commit -m "Move core simulation/driver/util tests onto master paths"`.

### Task 5: Bucket-C renames (drop the `_dolfinx` suffix)

**Files:**
- Rename: the 8 bucket-C files listed in the move table (7 under
  `src/finmag/tests/**` + `examples/test_examples_dolfinx.py`).
- Modify: `pixi.toml` — `dolfinx-src-import-pytest` (both the dolfinx AND the
  legacy default-env `src-import-pytest` at line ~292 reference
  `test_example_dolfinx.py`), `dolfinx-src-deferred-pytest`,
  `dolfinx-src-native-bem-pytest`, `dolfinx-src-ordering-pytest`,
  `dolfinx-src-examples-pytest`, `dolfinx-src-io-utils-pytest`,
  `dolfinx-src-comparison-pytest` (magpar_io if not already done in Task 2).

- [x] **Step 1:** `git mv` each of the 8 files to its suffix-less name. No
  content changes except: if a file's own docstring names its old filename,
  update that one string.
- [x] **Step 2:** `grep -rn "_dolfinx.py" pixi.toml dev/ src/ examples/` —
  update every remaining reference; expect ZERO hits afterwards except
  historical docs (docs/ is Task 8).
- [x] **Step 3:** Run the affected gates:
  `dolfinx-src-import-pytest` (28), `dolfinx-src-deferred-pytest`,
  `dolfinx-src-native-bem-pytest`, `dolfinx-src-ordering-pytest`,
  `dolfinx-src-examples-pytest` (14 passed/3 skipped), plus legacy-env
  `pixi run src-import-pytest` if the legacy environment is installed locally
  (if the env is not materialised, note it and rely on the dolfinx variant).
- [x] **Step 4: Commit:**
  `git commit -m "Drop _dolfinx suffix from genuinely-new test files"`.

### Task 6: Retire the in-tree legacy test lane

**Files:**
- Modify: `pixi.toml` — delete the `barmini-suite` task (line ~311).
- Delete: `dev/bin/verify-python3-m3`, `dev/bin/verify-python3-barmini-suite`,
  and any other `dev/bin/verify-python3-*` script that invokes a task listing
  in-tree test files (`grep -l "suite" dev/bin/verify-python3-*`). KEEP the
  smoke-style scripts (`verify-python3-import`, `-barmini-smoke`,
  `-restart-smoke`, `-native-fk-bem-smoke`, `-m1`, `-m2`) — they exercise
  current `src` without referencing test files; verify each with
  `grep -n "_test\|test_" <script>` before keeping.

**Interfaces:**
- Produces: legacy suite execution contract =
  `dev/bin/run-legacy-oracle -- pixi run --locked barmini-suite`
  (runs at the frozen oracle commit, which has its own task + test files).

- [x] **Step 1:** Verify the oracle path once:
  `dev/bin/run-legacy-oracle -- git show HEAD:pixi.toml | grep -c barmini-suite`
  expected ≥1 (do NOT run the full legacy suite — minutes-long and needs the
  legacy env; existence check suffices).
- [x] **Step 2:** Delete the `barmini-suite` task and the identified scripts;
  `grep -rn "barmini-suite" pixi.toml dev/bin` → only `run-legacy-oracle`
  documentation remains (add a one-line comment in `pixi.toml` where the task
  was: `# Legacy FEniCS-2019 suite: run at the frozen oracle commit via
  dev/bin/run-legacy-oracle -- pixi run --locked barmini-suite [owner 2026-07-27]`).
- [x] **Step 3:** Confirm `dev/bin/verify-dolfinx-m5` does not reference any
  deleted script/task: `grep -n "verify-python3\|barmini-suite" dev/bin/verify-dolfinx-m5`
  → no hits.
- [x] **Step 4: Commit:**
  `git commit -m "Retire in-tree legacy suite lane; legacy runs at frozen oracle commit"`.

### Task 7: Full verification + post-move inventory

- [x] **Step 1:** Clean tree check: `git status --short` → empty.
- [x] **Step 2:** Run `dev/bin/verify-dolfinx-m5` end-to-end. Expected: exit 0,
  all 33 steps green, per-gate pass counts equal to their pre-move values.
  Any regression here is a broken move (most likely a `__file__`-relative
  fixture path or a stale pixi list) — fix in place, amend the responsible
  task's commit ONLY if not yet reviewed, otherwise commit the fix separately.
- [x] **Step 3:** Run `dev/bin/inventory-dolfinx-suite`. Record the INVENTORY
  line. Compare against Task 1's baseline: passed should be ≥ baseline (moved
  files unchanged), and the failure/error population should now consist of
  (a) never-ported master files, (b) retained bucket-B ancestors, (c)
  `not_ported`-marked carried tests. Spot-check 3 failures to confirm they are
  genuine not-ported functionality (e.g. `import dolfin` / NameError `df`),
  not move breakage.
- [x] **Step 4:** Save both logs to `/tmp` paths named in the docs commit
  (Task 8 cites them).
- [x] **Step 5: Commit** nothing (verification only) unless fixes were needed.

### Task 8: Documentation closure

**Files:**
- Modify: `docs/superpowers/HANDOVER.md`, `docs/superpowers/acceptance-register.md`,
  `docs/superpowers/master-pixi-parity-manifest.md`,
  `docs/superpowers/capability-status.md` (only if it names sibling paths),
  `docs/superpowers/plans/2026-07-23-sr1-prioritised-plan.md`,
  `dev/dolfinx/porting_map.md`, `transition-notes.org`, this plan file
  (tick boxes).

- [x] **Step 1: Register.** Add row **D30**: canonical-paths convention —
  owner decision 2026-07-27 supersedes the 2026-07-25 sibling convention;
  ports live at master paths; review diff = `git diff b5015c5a..HEAD --
  <path>`; in-tree legacy lane retired to the frozen oracle commit; full-suite
  inventory lane is the parity backlog. Also: flip **M4a/M4b** dispositions
  from "*pending owner decision*" to "ratified 2026-07-25: deferred for SR1"
  (the P5.1 outcome, previously recorded only in HANDOVER/porting_map).
- [x] **Step 2: SR1 plan file.** Tick P5.1/P5.2/P5.3 `[x]` with one-line
  completion notes pointing at HANDOVER (closing the omission from `c1d6a9fb`).
- [x] **Step 3: HANDOVER.** Update the "minimal-diff test-conversion" section:
  sibling convention superseded by canonical paths (D30); how to review
  (git diff against `b5015c5a`); how to run the legacy suite (oracle command);
  the inventory lane command and its current INVENTORY tally from Task 7.
- [x] **Step 4: Manifest.** Add a post-baseline note: canonical-paths
  relocation (paths changed, content unchanged); the inventory lane is now the
  live "which owner-Now families still lack a witness" signal.
- [x] **Step 5:** Update `porting_map.md` + `transition-notes.org` with the
  relocation entry (commit range, INVENTORY tally, retained-ancestor list).
- [x] **Step 6:** `grep -rn "_dolfinx.py" docs/superpowers/*.md` — annotate the
  canonical docs (HANDOVER, capability-status, manifest current-state sections)
  to new paths; leave historical/archived narrative (old plans, old log entries)
  untouched.
- [x] **Step 7: Commit:**
  `git commit -m "Docs closure: canonical test paths (D30), legacy lane retirement, inventory lane"`.

---

## Self-review notes

- Spec coverage: original-path moves (Tasks 2–5), oracle-only legacy execution
  (Task 6), run-everything-visibly (Tasks 1, 7), "keep barmini-suite's FILES,
  retire its RUNNER" (Tasks 4/6), review-diff goal (Task 8 documents it).
- The bucket tables are derived from in-file headers + the P5 audit; the
  headers are declared authoritative on conflict.
- Type consistency: the marker name `not_ported`, task name
  `dolfinx-src-suite-inventory`, and script name
  `dev/bin/inventory-dolfinx-suite` are used identically across Tasks 1–8.
