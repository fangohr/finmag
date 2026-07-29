# Towards Dolfin x

Finmag active development was suspended around 2018. Major software used was python2 and dolfin 2017.1. 

In 2026, Hans started at attempt to update these dependencies in (see https://github.com/fangohr/finmag/issues/51). This involves assistance by agents, so it seemed important to keep a reference to the original and intermediate steps.

## Important versions and snapshots

- tag `python2`: last python2 and dolfin 2017 version is tagged as `python2` (https://github.com/fangohr/finmag/releases/tag/python2). Use this as a baseline and if you use the 'old' finmag that effectively can only run in a container. (commit b5015c5a47c244eea1476d8e718286137dca0c83)

- tag `python3`: in the pixi branch [#50](https://github.com/fangohr/finmag/pull/50) we have moved from python 2 to python 3, and updated dolfin from 2017 to 2019. The resulting code runs in a pixi environment. (No need for a container anymore, currently restricted to Linux.).
  
- tag [dolfinx-port branch #53](https://github.com/fangohr/finmag/pull/53) was based on the pixi branch (tag `python3`) and updated pixi branch to use the new dolfinx (ce60ae3e82abdd5f4a362543dd913ff269d3fe62) for exploration of dolfin x.

- tag `sr1` (2026-07-28): a snapshot on the dolfinx-parity branch, at the point where we declared "Support Release 1", i.e. the first subset of finmag we are willing to call supported. That subset is written down in [docs/SUPPORTED.md](docs/SUPPORTED.md). This is a snapshot on the branch, not the end of it. (commit e228e5f3eb38d13e793674d300f28f05cc790f6f)

- tag `dolfinx` (does not exist yet) marks the end of [dolfinx-parity branch #52](https://github.com/fangohr/finmag/pull/52) in which the dolfinx-port is modified and extend to match the functionality and structure of the original master(python2) as close as possible. We will only create it when that branch is merged into main.

## Limitations of dolfinx port

- Tests should confirm that code is ported correctly.
- Performance has been sacrificed in places. For example the 'instant' feature in dolfin 2017 has disappeared, and this has been replaced by numpy code. The file [docs/SUPPORTED.md](docs/SUPPORTED.md) has more details.
- Performance improvements will need to be done later. Where we currently stand, and what we intend to do about it, is in [docs/performance.md](docs/performance.md).
- Not all features have been ported to dolfinx (LLB for example is missing). Tests that test non-ported features have been kept as `xfail`. More details in [docs/SUPPORTED.md](docs/SUPPORTED.md).

At `sr1` the port is a serial, deterministic simulator: the common energy terms, demag, varying materials and regions, local spin-transfer torque, time integration (SciPy and native CVODE), scheduling, restart and the common output. Thermal SLLG/LLB, normal modes, NEB and MPI time stepping are not there yet. It is not full parity with master, and the tests being green means this subset is green, not that everything is back.


## Details for developers

A number of documents document the transition process: [transition-notes.org](docs/archive/transition-notes.org), [plan.org](docs/archive/plan.org), and in parts [agents-progress-log.md](docs/archive/agents-progress-log.md).

The current documentation lives in [docs/](docs/README.md):

- [docs/SUPPORTED.md](docs/SUPPORTED.md) — what works today, how it is validated, and to what tolerance.
- [docs/decisions.md](docs/decisions.md) — what we decided during the port, and why.
- [docs/testing.md](docs/testing.md) — the test lanes, and how to run them.
- [docs/performance.md](docs/performance.md) — what is slow, and the plan for it.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to work on the code.

In the transition process, a number of milestones was used to mark certain progress, and may be referred to in commit messages.

Milestones (the project's verification ladder):

- M1 — proved Python-3 viability: builds the py3-dolfin2017 Docker snapshot image and runs the port's basic checks inside it.

- M2 — proved the pixi/conda-forge FEniCS-2019 environment: import finmag, barmini construct, barmini time-integration smoke, and save/restart smoke, all against current src.

- M3 — proved the legacy test suite on that environment: ran the in-tree master pytest subset (barmini-suite) — since retired; legacy suite now runs only at the frozen oracle commit via run-legacy-oracle.

- M4 — proved DOLFINx viability: frozen early gate over the dev/dolfinx prototypes (env versions, import, smoke, prototype pytest, example + restart example).

- M5 — proves the whole port: the 33-gate aggregate (editable install, native build, provenance, every focused src test gate incl. MPI probes, comparison suite, core Sundials smoke, fast examples).


## Use of agents

In the transition, CODEX and Claude Code was used. Generally, all committed code changes were reviewed by a human. The use of an agent has been recorded through the commit message, or a co-author.


## Workflows

Current workflows on the branch:

- test-fast.yml (push/PR, ~13 min) — the 33-gate M5 aggregate (dev/bin/verify-dolfinx-m5); the everyday green gate. Formerly dolfinx-m5.yml.

- test-python.yml (weekly, and on demand) — a full sweep of the src/finmag suite, which has to come back with failed=0 and errors=0.

- test-slow.yml (on demand) — the heavy examples, run with FINMAG_EXAMPLE_FULL=1.

- test-prototypes.yml (push/PR, ~1.5 min) — the frozen M4 prototype gate; a cheap historical guard. Formerly dolfinx-m4.yml.

- docker-smoke.yml, docker-image.yml — dormant, dispatch-only relics from the python2 era (a Docker pytest smoke, and a pull of the old images).

Two caveats. GitHub only fires scheduled and manually dispatched workflows from the default branch, so test-python and test-slow do not actually run while we sit on dolfinx-parity — they are inert until this branch is merged into main, and we run their equivalents locally in the meantime. And test-slow does not fit on a hosted runner anyway: those stop after 6 hours, while the full example lane currently needs some 10-13 hours (see [docs/performance.md](docs/performance.md)).

The older python3-m1/m2/m3.yml and python3-core-suite.yml workflows were deleted in July 2026, and workflow.yml became docker-smoke.yml.

More detail on all of this is in [docs/testing.md](docs/testing.md).
