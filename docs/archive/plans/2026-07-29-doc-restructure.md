> **ARCHIVED (2026-07-29).** This restructure is complete; the plan is kept
> as a historical record and archived by itself as its own final step.
> Current truth: docs/README.md.

# Documentation Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the owner-approved documentation restructure: three clear
doors (main's README → `dolfinx-transition.md` → `docs/`), one audience per
document, the six verified falsehoods corrected, all historical material
archived unedited, zero information loss.

**Architecture:** The authoritative layout spec is
`docs/superpowers/specs/2026-07-29-doc-restructure-layout.md` (committed
alongside this plan) — every task references its sections instead of
restating them; ON CONFLICT THE SPEC GOVERNS. Owner decisions 2026-07-29:
(1) the `dolfinx` tag is minted ONLY at merge-to-main; `sr1` is documented
as the branch's SR1 snapshot; (2) both surgical README edits approved;
(3) register per-row-block reformat + `owner-porting-checklist.pdf`
deletion approved. Audit evidence (falsehood list, fragmentation map):
/home/sam/.claude/jobs/6b8f36a7/tmp/doc-audit-report.md (session-local;
its falsehood list is restated verbatim in Task 1 so the plan is durable).

**Tech Stack:** git mv (history-preserving), markdown, one small Python
reformat script for the register (committed), pixi only for link-checking
greps. NO source-code changes anywhere in this plan; tag `sr1` must not
move; `dev/benchmarks/` never committed.

## Global Constraints

- Branch `dolfinx-parity`; every implementer/reviewer asserts branch+HEAD
  first (rogue-checkout guard).
- NO information loss: content moves or is archived — never deleted
  (exception, owner-approved: the generated `owner-porting-checklist.pdf`;
  its `.md` source is archived).
- Historical files are archived UNEDITED except a prepended banner:
  `> **ARCHIVED (2026-07-29).** Historical record of the porting process;
  statements reflect their writing date. Current truth: docs/README.md.`
- Two-step register migration (mv commit, then reformat commit) so
  `git log --follow` and blame survive; the reformat must be provably
  content-identical (task specifies the check).
- Every moved file: sweep inbound references repo-wide (`grep -rn
  "<oldpath>" --include="*.md" --include="*.py" --include="*.toml"
  --include="*.yml" .` excluding .git and docs/archive) and repoint
  CURRENT-STATE references; references inside archived files stay
  untouched.
- Max lengths per the spec's per-file contracts; commit messages
  one-line + body, `Co-Authored-By: <model> <noreply@anthropic.com>`.
- After the final task, every markdown link in the non-archive docs must
  resolve (task 8 runs the check).

---

### Task 1: Correct the six verified falsehoods (in place, before anything moves)

**Files (verify each claim in-file before editing; dated-correction style,
never silent rewrites):**
- `docs/superpowers/master-pixi-parity-manifest.md`: rows N10, N11, N41,
  N43, N45, N46, N47, N60, N61 read `MISSING` for features that shipped
  (Field.cross/dot, from_generic_vector, HDF5 round-trip, probe_field,
  vortex initialisers, logging helpers, plot_helpers, mesh diagnostics,
  save_m_in_region — map each row to its shipped feature by reading the
  row text, cross-check against `docs/SUPPORTED.md` §2 and the register
  D-rows, cite the shipping commit where the register names it). Also the
  N41 self-contradiction (~:86 vs ~:234) — reconcile with a dated note.
- `docs/superpowers/interface-audit.md:131`: claims Field.cross/dot/
  from_generic_vector raise NotImplementedError — false since P3.3/P3.4;
  dated correction.
- Obsolete inventory tally `752/34/59` in 3 files (grep for `752` across
  docs/ and dev/dolfinx/) → dated supersession to the current
  `769/0/0/46/271`.
- `docs/superpowers/HANDOVER.md:616-623`: demands a capability-status
  sweep that is already done (grep confirms 0 pendings) — mark discharged.
- `agents.md:182`: calls field.py a `dolfin.Function` wrapper — correct to
  the dolfinx description (one sentence).
- `dev/dolfinx/porting_map.md:200,221`: assert a scipy integrator default
  (reverted to sundials in P1.3) — 2 unmarked occurrences get dated
  corrections.
- Register + capability-status headers still say 2026-07-23 over
  2026-07-28 bodies → update header dates.
- [ ] Step 1: verify each falsehood live (grep/read); fix with dated
  notes; Step 2: commit
  `Docs: correct six verified stale claims (doc-restructure T1)`.

### Task 2: Adopt main's README + bring dolfinx-transition.md onto the branch

**Files:** `README.md` (replace with `git show origin/main:README.md`
+ the two approved surgical edits per spec §1); create
`dolfinx-transition.md` (start from `git show
origin/main:dolfinx-transition.md`, then apply the spec's §"transition
file" outline: tags section — `sr1` documented as the SR1 snapshot,
`dolfinx` tag "minted at merge to main" per owner 2026-07-29; workflows
section — replace the stale list with the six current workflows
(test-fast, test-python, test-slow, test-prototypes, docker-smoke,
docker-image) and their one-liners; add the links into docs/ per spec;
STAY ≤120 lines, Hans's voice, extend don't rewrite).
- [ ] Step 1: diff our README vs main's; extract every branch-added
  fact into a parking file `/home/sam/.claude/jobs/6b8f36a7/tmp/
  readme-parked.md` with its destination per spec §2 (INSTALL.md /
  testing.md / SUPPORTED §1 / transition file) — Task 4 consumes it;
  nothing may be lost.
- [ ] Step 2: write both files; `git diff origin/main -- README.md` must
  show ONLY the two approved edits.
- [ ] Step 3: commit `Adopt main README; bring dolfinx-transition.md
  up to date (doc-restructure T2)`.

### Task 3: docs/ router + archive skeleton + mechanical moves

**Files:** create `docs/README.md` (router per spec: three reading paths,
≤60 lines); create `docs/archive/README.md` (what the archive is, banner
text); `git mv` per the spec's migration map: HANDOVER.md,
master-pixi-parity-manifest.md, interface-audit.md,
owner-porting-checklist.md, completed plans/ and specs/ (NOT the live
perf plan, NOT this plan/spec until the final task), dev/dolfinx/
porting_map.md, transition-notes.org, plan.org, todo/failingtests.org →
`docs/archive/…` mirroring the spec's tree; prepend the ARCHIVED banner
to each; `git rm docs/superpowers/owner-porting-checklist.pdf`
(approved); move `docs/superpowers/plans/2026-07-28-post-sr1-performance.md`
→ `docs/plans/`.
- [ ] Steps: mv batch → banners → inbound-reference sweep (Global
  Constraints) → commit `Docs: router, archive, mechanical moves
  (doc-restructure T3)`.

### Task 4: docs/INSTALL.md + CONTRIBUTING.md + AGENTS.md

Per spec contracts (§1 root + docs tables): INSTALL.md absorbs the parked
README install/quickstart/packaging content + SUPPORTED's install
pointers; CONTRIBUTING.md (pixi env, lanes, how to add a test/feature,
decision-recording rule: "behavioural deviations get a register row");
AGENTS.md replaces/absorbs agents.md (agent conventions, safe-execution
protocol, branch-assert guard) — `git mv agents.md AGENTS.md` then edit,
keeping its corrected content. Commit
`Docs: INSTALL, CONTRIBUTING, AGENTS (doc-restructure T4)`.

### Task 5: docs/decisions.md — the "what we chose and why" narrative

The one substantial writing task (≤400 lines). Sources: the register's
big rationale cells, the D29/anchor verdict docs, HANDOVER's narrative
sections (now archived — cite archive paths), the spec's outline. One
section per major decision, each ending with links to its register
row(s) + evidence. REQUIRED sections per the spec: environment (pixi/
conda-forge, no container); mesh layer → Gmsh/OCC (+D34 maxh); Sundials
default restored; coordinate-aware restart v2; corrected-physics
divergences (D3, D20, D29-drift); canonical test paths + strict-xfail
(D30/D33) and WHY (owner review-diff requirement, criterion 4); the
box-assemble-only energy method (D32/M12) and its P1 performance
consequence + planned fast path; deferred families (normal modes,
thermal, MPI, PBC-D19) with reasons; CI tier philosophy. Voice: plain
prose for a future maintainer, past tense, no process jargon ("agent",
"task", "wave" only where the agent-assisted process itself is the
subject). Commit `Docs: decisions narrative (doc-restructure T5)`.

### Task 6: register + capability-status migration

Two commits, spec §"registers": (a) `git mv
docs/superpowers/acceptance-register.md docs/acceptance-register.md` and
`git mv docs/superpowers/capability-status.md docs/capability-status.md`
+ inbound-reference sweep; (b) reformat: commit a small script
`dev/bin/reformat-register.py` that converts the table to per-row blocks
(## D1 …: Issue/Detail/Recommendation/Disposition as labelled
paragraphs) + builds the 57-line index at top; PROOF of content identity:
the script also emits a normalised-text dump (all cell text, order
preserved, whitespace-collapsed) before and after — `diff` of the dumps
must be empty, shown in the report. capability-status: cells trimmed to
one sentence + link per spec (this IS a content edit — do it by hand,
conservatively, moving trimmed detail into archive-refs or decisions.md
links, never deleting facts that exist nowhere else). Commits
`Docs: move registers (doc-restructure T6a)` and
`Docs: register per-row reformat + capability trim (doc-restructure T6b)`.

### Task 7: docs/testing.md + docs/performance.md + SUPPORTED.md reconcile

Per spec contracts: testing.md (six workflows + milestone lineage table,
the inventory lane + current tally, the oracle lane, how strict-xfail
governs unported tests, how to run everything locally); performance.md
(P1 root cause + numbers, the perf plan pointer, M3, runner ceilings,
serial-integration limitation, what "fast enough" will look like);
SUPPORTED.md: reconcile pointers to the new structure (its §7 mechanism
text, CI mentions, register links to docs/acceptance-register.md) WITHOUT
shrinking its content — it stays the single user contract. Commit
`Docs: testing + performance pages; SUPPORTED reconcile
(doc-restructure T7)`.

### Task 8: link integrity + memory/spec closure

- [ ] Step 1: link check — script or grep every `](…)` target in
  non-archive markdown; every relative link resolves; report table.
- [ ] Step 2: `doc/STATUS.md` banner on the legacy Sphinx tree (spec §3
  deviation 3).
- [ ] Step 3: archive THIS plan + layout spec + audit summary into
  docs/archive (self-archival is the restructure completing itself);
  update dolfinx-transition.md's developer-pointers line if needed.
- [ ] Step 4: commit `Docs: link integrity + closure (doc-restructure
  T8)`.

---

## Self-review notes

- Spec governs on conflict; owner decisions (tag story, README edits,
  register reformat+PDF) recorded in the header and Tasks 2/3/6.
- No-information-loss is enforced three ways: parking file (T2), archive
  banners not edits (T3), register content-identity proof (T6b).
- The falsehood fixes go FIRST (T1) so nothing false gets archived as if
  it were current truth without its dated correction.
