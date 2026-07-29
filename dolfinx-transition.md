# Towards Dolfin x

Finmag active development was suspended around 2018. Major software used was python2 and dolfin 2017.1. 

In 2026, Hans started at attempt to update these dependencies in (see https://github.com/fangohr/finmag/issues/51). This involves assistance by agents, so it seemed important to keep a reference to the original and intermediate steps.

## Important versions and snapshots

- tag `python2`: last python2 and dolfin 20217 version is tagged as `python2` (https://github.com/fangohr/finmag/releases/tag/python2). Use this as a baseline and if you use the 'old' finmag that effectively can only run in a container. (commit b5015c5a47c244eea1476d8e718286137dca0c83)

- tag `python3`: in the pixi branch [#50](https://github.com/fangohr/finmag/pull/50) we have moved from python 2 to python 3, and updated dolfin from 2017 to 2019. The resulting code runs in a pixi environment. (No need for a container anymore, currently restricted to Linux.).
  
- tag [dolphinx-port branch #53](https://github.com/fangohr/finmag/pull/53) was based on the pixi branch (tag `python3`) and updated pixi branch to use the new dolphinX (ce60ae3e82abdd5f4a362543dd913ff269d3fe62) for exploration of dolfin x.

- tag `dolfinx` (does not exist yet) marks the end of [dolfin-parity branch #52](https://github.com/fangohr/finmag/pull/52) in which the dolfinx-port is modified and extend to match the functionality and structure of the original master(python2) as close as possible. 

## Limitations of dolfinx port

- Tests should confirm that code is ported correctly.
- Performance has been sacrificed in places. For example the 'instant' feature in dolfin 2017 has disappeared, and this has been replaced by numpy code. The file [docs/SUPPORTED.md](docs/SUPPORTED.md) has more details.
- Performance improvements will need to be done later.
- Not all features have been ported to dolfinx (LLB for example is missing). Tests that test non-ported features have been kept as `xfail`. More details in [docs/SUPPORTED.md](docs/SUPPORTED.md).


## Details for developers

A number of documents document the transition process: [transition-notes.org](transition-notes.org), [plan.org](plan.org), and in parts [agents.md](agents.md).

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

To be reviewed and updated.

Current workflows on the branch:

- dolfinx-m5.yml (push/PR, ~13 min) — runs verify-dolfinx-m5, i.e. everything above in M5; the everyday green gate.

- dolfinx-m4.yml (push/PR, ~1.5 min) — runs the frozen M4 prototype gate; cheap historical guard.

- python3-m1/m2/m3.yml + python3-core-suite.yml — the legacy-era jobs, skip-gated off this branch (m3/core-suite reference the retired lane); scheduled for deletion in CI-T6.

- docker-image.yml, workflow.yml — dispatch-only dormant relics (Docker image build; pre-pixi Docker pytest smoke).
