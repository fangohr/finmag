# Finmag DOLFINx port — FINAL documentation layout (second-opinion architect)

**Repo:** `/home/sam/repos/finmag` · **Branch:** `dolfinx-parity` · **HEAD:** `f4ff725d` · **Date:** 2026-07-29
**Basis:** doc-audit-report.md (same job dir) + `origin/main:README.md` + `origin/main:dolfinx-transition.md` + tree at HEAD.
**Hard constraint honoured:** main's README is adopted nearly verbatim and wins on merge; the transition story
lives in root `dolfinx-transition.md` (Hans's file, EXTENDED not replaced); everything deeper lives under `docs/`.

---

## 1. Final file tree (per-file audience · contract · max length)

### Repo root
| Path | Audience | Contract | Max |
|---|---|---|---|
| `README.md` | everyone, first 60 s | **main's version, nearly verbatim.** Two surgical edits only (owner sign-off): (1) Tests badge `workflows/workflow/badge.svg` → `workflows/test-fast/badge.svg` (the `workflow` workflow does not exist; `test-fast` is the everyday gate); (2) in "About the `main` branch" add one sentence: *"What is already supported is documented in [docs/SUPPORTED.md](docs/SUPPORTED.md)."* Nothing else changes — no DOLFINx install/CI/status prose returns to README (see §2 for where each piece goes). | main's length +3 lines |
| `dolfinx-transition.md` | everyone arriving from README | **Hans's narrative, brought onto this branch and updated** (see §3). The DOOR to the port story: versions/tags, limitations summary, milestones, agents, workflows — each ≤ a screen, each ending in a link into `docs/`. NOT the deep rationale (that is `docs/decisions.md`, linked). | 120 lines |
| `CONTRIBUTING.md` | contributor | *(new, conventional name — GitHub surfaces it automatically)* pixi environment, `dev/bin/*` scripts, how to run each test lane, where to record a decision (the register), the "update capability-status in the same commit" rule. Absorbs the useful half of `agents.md`. | 200 |
| `AGENTS.md` | agents/automation | *(replaces lowercase `agents.md` — `git mv` then slim)* repo conventions + safe-execution protocol from `HANDOVER.md:349-378` + pointers to CONTRIBUTING/docs. Historical progress log → archive. | 150 |
| `doc/` (legacy Sphinx, 9 MB) | nobody (historical) | **LEFT IN PLACE** + new `doc/STATUS.md` banner (~15 lines: "frozen 2019, Python-2 kernels, not built by CI; live docs are in ../docs/"). Physics salvage (`aboutfinmag.rst`, FK-demag report.tex, `finmag_dmi_roadmap.mkd`, `examples/*/doc.rst`) → `docs/physics/` as an OPTIONAL follow-up commit, not blocking. |
| `install/`, `binder/` | historical | leave in place; one line each in `docs/archive/README.md` noting they document the pre-pixi era. Binder badge stays as-is in README (it is main's README; it builds the python2 image, which main's README correctly labels python2). |

### `docs/` (flat — no `status/` subdirectory; see Deviation 1)
| Path | Audience | Contract | Max |
|---|---|---|---|
| `docs/README.md` | everyone | *(new)* one-screen router: three reading paths (user / contributor / owner-decisions), one line per doc. | 60 |
| `docs/SUPPORTED.md` | user | **kept whole and self-contained** — supported API, validation classes + tolerances, Later list, Dropped list, failure modes. It stays the single answer to "works / does not work". Edits: strip 42 machine-local paths + model signature; move §7's inventory-tally *history* to `docs/testing.md` (current tally stays here). | 550 |
| `docs/INSTALL.md` | user/contributor | *(new — receives the branch-README material)* pixi install steps 1–4, `verify-dolfinx-m5` (33/33), quickstart snippet, `sim_with` pointer, the pyproject-version/editable-install/no-wheel packaging notes. | 180 |
| `docs/decisions.md` | owner + contributor | *(new — the deep "what we chose and why" narrative)* port-in-place vs clean-room, Py3/FEM sequencing rule (kills the triplication), frozen FEniCS-2019 oracle contract, box-assemble, restart contract, axis normalisation, Sundials-default restoration, method-name removals. One `##` per decision, prose, each ends with its register-row link. Absorbs `specs/2026-07-06-dolfinx-core-port-design.md`. | 400 |
| `docs/acceptance-register.md` | owner | **moved from `superpowers/` (pure `git mv`, own commit), then reformatted in a second, scripted, content-identical commit**: per-row `###` blocks (ID, Difference, Evidence, Recommendation, Disposition+date) + 57-line one-line index at top. Fix the lying `Reconciled: 2026-07-23` header. Authority unchanged: sole place a disposition changes. | index 70 + 57 blocks |
| `docs/capability-status.md` | contributor | moved from `superpowers/`; C01–C22 matrix, evidence cells trimmed to one sentence + register/commit links; fix the stale `2026-07-23` header. | 200 |
| `docs/testing.md` | contributor | *(new)* CI tiers (`test-fast` ≈ old M5 33-gate, `test-prototypes` ≈ old M4, `test-python` weekly inventory `failed=0 errors=0`, `test-slow` on-demand FULL lane, `docker-image`, `docker-smoke`), inventory lane + **current** tally (769/0/0/46/271) + tally history, legacy-oracle lane, strict-xfail rule, default-branch scheduling caveat, 360-min runner cap. Absorbs HANDOVER test-path narrative + branch-README CI section. | 250 |
| `docs/performance.md` | owner + contributor | *(new)* the optimisation backlog in prose: P1 (~17.6x, form re-assembly root cause), M3 compiled-RHS backend, deferred `std_prob_4`/`magnetic_grain` re-runs, FULL-lane runtime vs runner cap, serial-only integration (C19), `surface_3d`/matplotlib-3.11. Links to the live plan for execution detail. | 200 |
| `docs/plans/2026-07-28-post-sr1-performance.md` | contributor (agentic) | moved from `superpowers/plans/`; unchanged — the one live plan. | — |
| `docs/physics/` *(optional follow-up)* | user | salvaged version-independent physics from `doc/` + `examples/*/doc.rst`. | — |

### `docs/archive/` (append-only material, preserved unedited except correction banners)
| Path | Notes |
|---|---|
| `docs/archive/README.md` | *(new, ≤40)* what is here, why kept, the "frozen snapshot / read chronologically" convention warning; one line each on `install/`, `binder/`, `doc/`. |
| `docs/archive/HANDOVER.md` | retired entry point, after its four live parts are extracted (what-works → already in SUPPORTED; test narrative → testing.md; protocol → AGENTS.md; open items → performance.md/SUPPORTED §5). Leave a 10-line stub at old path? No — `superpowers/` dir is deleted entirely; inbound links fixed instead. SR1 declaration section → `docs/archive/sr1-declaration.md`. |
| `docs/archive/master-pixi-parity-manifest.md` | **mandatory correction header** naming the 9 false `MISSING` rows (N10 N11 N41 N43 N45 N46 N47 N60 N61) + the stale 752/34/59 tally. |
| `docs/archive/interface-audit.md` | correction header for `:131` (Field `cross`/`dot`/`from_generic_vector` shipped). |
| `docs/archive/owner-porting-checklist.md` | extend the addendum to cover hysteresis (D4) + Field (P3.3/P3.4) beliefs; **delete the stale PDF** (regenerable; git history keeps it). |
| `docs/archive/plans/` (7 completed), `docs/archive/specs/` (5 records) | banners kept; core-port design spec absorbed into decisions.md first, then archived like the rest. |
| `docs/archive/porting_map.md`, `transition-notes.org`, `plan.org`, `agents-progress-log.md` (historical half of agents.md), `dev-dolfinx-README.md`, `todo.org`, `failingtests.org` | shared banner: *"Chronological engineering evidence, preserved unedited. Not current status."* One-line correction notes at `porting_map.md:200,221` (Sundials default) and `dev-dolfinx-README` non-scope list. |

---

## 2. README reconciliation (branch README → destinations; main's README wins)

| Branch-README content (lines approx.) | Destination |
|---|---|
| Legacy header/badges/About/Docker/Binder/citation/publications | superseded by main's version verbatim (main already restructured these under "python2") |
| "Installing the DOLFINx port (pixi)" steps 1–4 + provenance | `docs/INSTALL.md` |
| "Verify the install" (verify-dolfinx-m5, inventory-dolfinx-suite) | `docs/INSTALL.md` (verify) + `docs/testing.md` (inventory-lane meaning) |
| Quickstart python snippet + `sim_with` + examples pointer | `docs/INSTALL.md` |
| pyproject version / editable / no-wheel rationale | `docs/INSTALL.md` (packaging appendix) |
| "Continuous integration" section | `docs/testing.md` |
| "Current DOLFINx status and limitations" + performance caveat | merge any delta into `docs/SUPPORTED.md` §1 + `docs/performance.md`; `dolfinx-transition.md` keeps the 5-line summary |
| Broken `workflow` badge | fixed to `test-fast` (also present in main's README — same one-line fix proposed there) |
Nothing in the branch README lacks a destination → adopting main's README loses nothing.
Hop check: user README→transition→SUPPORTED/INSTALL (2–3); contributor README→CONTRIBUTING→testing/capability-status (2–3); owner README→transition→decisions→register (3 to the narrative, register linked per-decision).

## 3. `dolfinx-transition.md` update outline (extend Hans's file; his voice and structure kept)

1. **Bring onto branch**: `git show origin/main:dolfinx-transition.md > dolfinx-transition.md`, commit as-is (clean baseline diff), then edit.
2. **Tags section** — honest reconciliation: keep the `python2`/`python3`/dolfinx-port entries; ADD: *"tag `sr1` (2026-07-28): Support Release 1 on the dolfinx-parity branch — first supported subset (serial deterministic simulator); contract in docs/SUPPORTED.md."* Keep `dolfinx` as "planned end-of-branch tag, does not exist yet" **unless owner says `sr1` supersedes it** (consult point). Fix typos `20217`→`2017`, `dolphinx`→`dolfinx`, `dolfin-parity`→`dolfinx-parity` (mechanical, non-substantive).
3. **Limitations** — keep; links to `docs/SUPPORTED.md` already correct; add one line linking `docs/performance.md`.
4. **Details for developers** — repoint `transition-notes.org`/`plan.org`/`agents.md` links to `docs/archive/…`; add: decisions narrative → `docs/decisions.md`; index → `docs/README.md`; contributing → `CONTRIBUTING.md`. Milestones list stays verbatim.
5. **Workflows** — delete "To be reviewed and updated"; replace stale list with the six current workflows and their lineage: `test-fast.yml` (push/PR, ~13 min; formerly dolfinx-m5), `test-prototypes.yml` (formerly dolfinx-m4), `test-python.yml` (weekly inventory), `test-slow.yml` (on-demand FULL lane), `docker-image.yml`, `docker-smoke.yml`; note python3-m1/m2/m3, python3-core-suite and workflow.yml were deleted (CI nail-down, 2026-07-28); full detail → `docs/testing.md`.
6. **Do NOT add** the deep decision narrative here — file stays ≤120 lines; `docs/decisions.md` carries it (see Deviation 2).

## 4. Migration map (every existing doc → disposition)

| Current path | Disposition |
|---|---|
| `README.md` | REPLACED by main's version + 2 surgical edits (§1); content dispersed per §2 |
| *(from main)* `dolfinx-transition.md` | ADOPT + EXTEND per §3 |
| `docs/SUPPORTED.md` | KEEP (path unchanged) + sweep local paths/signature; tally history → testing.md |
| `docs/superpowers/acceptance-register.md` | MOVE `docs/acceptance-register.md` (git mv) then scripted reformat; fix header date |
| `docs/superpowers/capability-status.md` | MOVE `docs/capability-status.md`; trim cells; fix header date |
| `docs/superpowers/HANDOVER.md` | SPLIT (testing.md / AGENTS.md / performance.md / archive/sr1-declaration.md) then ARCHIVE |
| `docs/superpowers/master-pixi-parity-manifest.md` | ARCHIVE + correction header (9 N-rows + stale tally) |
| `docs/superpowers/interface-audit.md` | ARCHIVE + correction header (`:131`) |
| `docs/superpowers/owner-porting-checklist.md` / `.pdf` | ARCHIVE + extended addendum / DELETE pdf |
| `docs/superpowers/plans/*` (7 done) | ARCHIVE `docs/archive/plans/` |
| `docs/superpowers/plans/2026-07-28-post-sr1-performance.md` | MOVE `docs/plans/` (live) |
| `docs/superpowers/specs/*` (5) | core-port design absorbed into decisions.md; all 5 ARCHIVE `docs/archive/specs/` |
| `agents.md` | SPLIT → `CONTRIBUTING.md` + `AGENTS.md`; progress log → archive |
| `plan.org`, `transition-notes.org`, `todo.org` | ARCHIVE `docs/archive/` |
| `dev/dolfinx/README.md`, `dev/dolfinx/porting_map.md` | ARCHIVE `docs/archive/` + correction notes (`porting_map:200,221`; non-scope list). `dev/dolfinx/` code stays. |
| `dev/failingtests.org`, `dev/sandbox|conversion-to-git|singularity` READMEs | failingtests.org → archive (D17/D19/M-row cross-ref noted); sandbox READMEs stay with their code |
| `doc/` (9 MB) | KEEP IN PLACE + `doc/STATUS.md` banner; physics salvage optional → `docs/physics/` |
| `install/`, `binder/` | KEEP; noted in archive README |
| 42 machine-local paths, 50 model signatures | SWEEP (replace with commit SHAs or delete) |

## 5. Ordered commit sequence (≈14 + 2 optional; each independently reviewable, no source changes)

1. **Correction pass on files in place** (audit's 2b, promoted to first): 9 manifest N-rows, interface-audit `:131`, porting_map `:200,221`, dev/dolfinx/README non-scope, owner-checklist addendum, both stale header dates, HANDOVER stale tally + phantom-debt para. *Stops active misinformation before any file moves.*
2. Adopt `origin/main:dolfinx-transition.md` verbatim onto branch (clean baseline).
3. Update `dolfinx-transition.md` per §3 (tags/workflows/links). *(after 1–2; before link fan-out so links can target final paths named in this spec)*
4. Create `docs/archive/` + README banner; `git mv` manifest, interface-audit, owner-checklist(+delete pdf), completed plans, specs, porting_map, dev-dolfinx README, plan.org, transition-notes.org, todo.org, failingtests.org. Pure moves.
5. `git mv` acceptance-register, capability-status, live plan out of `superpowers/`; delete empty `docs/superpowers/`; fix every inbound link (link-checker run).
6. Write `docs/testing.md` (retires the stale-tally fan-out).
7. Write `docs/performance.md`.
8. Write `docs/INSTALL.md` (extract from branch README).
9. Write `docs/decisions.md` (largest new writing; absorbs core-port design spec + Guiding Rule triplication).
10. Scripted reformat of `docs/acceptance-register.md` to per-row blocks + index (content-identical; reviewed row-by-row; separate from the mv in 5 so blame stays clean).
11. Trim `docs/capability-status.md` cells.
12. Retire HANDOVER: extract remnants → destinations, archive it (+ `sr1-declaration.md`). *(requires 6–9)*
13. Split `agents.md` → `CONTRIBUTING.md` + `AGENTS.md`; archive progress log.
14. Replace `README.md` with main's version + the 2 surgical edits; write `docs/README.md` index; sweep machine-local paths + signatures; add `doc/STATUS.md`.
15. *(optional)* physics salvage → `docs/physics/`. 16. *(optional)* CI link-check job.

## 6. Top deviations from the audit's proposal — see final report. Owner-consult points — see final report.
