# Decisions made during the DOLFINx port

This document explains the choices made while porting finmag from the
original Python-2/legacy-`dolfin` (2017) codebase to Python 3 and DOLFINx —
what we chose, and why. It is written for someone who knows the original
finmag well but was not involved in this update at all.

Every section below ends with a link into the
[acceptance register](superpowers/acceptance-register.md), the ledger where
every difference from the original `master` behaviour is recorded with a
decision code (`D<n>`, `M<n>` or `P1`). Where you see a code like "decision
D20", that always refers to a row in that register — the codes have no
meaning outside it. The current state of what works, and to what
tolerance, is [docs/SUPPORTED.md](SUPPORTED.md); this document is about why.

---

## 1. Environment: pixi and conda-forge, no container

The original finmag (Python 2, `dolfin` 2017) only runs inside the
Singularity/Docker container built for it at the time — that container is
effectively the only reliable way to reproduce that environment today.

The port moved to [pixi](https://pixi.sh), which manages a conda-forge
environment declared in `pixi.toml` and pinned in `pixi.lock`, removing the
container requirement: `pixi install` on Linux gives a working,
reproducible environment without building or running an image. The first
working step used conda-forge's `fenics 2019.1.0` package as the Python-3
intermediate stage, rather than reproducing the exact Debian-snapshot
`dolfin 2017.1.0` stack the container used — the old stack was judged not
worth reproducing bit-for-bit once a modern conda-forge package covered
the same workflows.

The same approach carried through to DOLFINx: the `dolfinx` pixi
environment resolves DOLFINx 0.10.0 on Python 3.12, and
`pixi run -e dolfinx dolfinx-install-editable` plus `dolfinx-native-build`
is now the entire install procedure (see [docs/INSTALL.md](INSTALL.md)). A
container remains available for the historical Python-2 code (`doc/`,
`install/`, `binder/`), but is no longer required to run current finmag.

---

## 2. Mesh generation: Gmsh/OCC, not Netgen

The original finmag generated meshes with Netgen, usually from a textual
`.geo`/CSG description, loaded into `dolfin` as an XML file. Neither half
of that pipeline carries over: Netgen's binary backend is not part of the
conda-forge DOLFINx environment, and Netgen's dolfin-XML mesh output cannot
be loaded by DOLFINx at all.

The port replaced this with Gmsh, driven through Gmsh's own Python API
(`gmsh.model.occ` for the geometry, `dolfinx.io.gmsh.model_to_mesh` to
bring the result into DOLFINx directly) — no intermediate XML files, no
shelling out to a `gmsh` binary. The public mesh-generator names and call
signatures in `finmag.util.meshes` are unchanged; only the backend
underneath changed. Before dropping the Netgen paths, we specifically
checked whether any selected test or geometry needed the binary backend or
the old dolfin-XML conversion, and found none.

One consequence is worth knowing in advance: **the same `maxh` mesh-density
parameter now produces a coarser mesh.** Gmsh maps `maxh` directly onto its
own `Mesh.MeshSizeMax` option, and on measured geometry this gives about
22% fewer nodes than the equivalent Netgen 5.3-era mesh at the same `maxh`
(1202 nodes / 4960 tetrahedra here, versus 1537 / 6886 in the archived 2017
Netgen run of the same `.geo` file). This looks like generator-version
drift generally, not something Gmsh-specific: the same symptom is
independently reported for Netgen 6.2 on the separate pixi branch (GitHub
issue #56). If you need Netgen-5.3-comparable density, reduce `maxh` by
roughly 10-20% and confirm the actual node/element count with
`mesh_info()` rather than assuming `maxh` parity across generator
versions. No code change was made; this is a calibration note, not a defect.

Register: [`M4a`, `M4b`](superpowers/acceptance-register.md) (Netgen backend
and `nmesh_to_dolfin`, both dropped), [`D34`](superpowers/acceptance-register.md)
(the `maxh` density drift).

---

## 3. The default time integrator is Sundials again, matching the original

Partway through the port, `Simulation` and `sim_with` temporarily defaulted
to the SciPy integrator rather than the original's native Sundials/CVODE
backend. This was corrected once native construction, advancement,
reset/reinitialisation, scheduling, save and restart were all validated
against Sundials under DOLFINx: **native Sundials is once again the
public default**, exactly as in the original. SciPy remains fully
supported as an explicit opt-in (`integrator_backend="scipy"`), but
nobody gets it by accident any more.

Register: [`D8`](superpowers/acceptance-register.md).

---

## 4. Restart is coordinate-aware (v2), and v1 archives are rejected

The original finmag's restart files stored raw solver-internal ("dof")
arrays — numbers indexed by the finite-element backend's internal
ordering, an implementation detail of the mesh and function space, not a
physical coordinate. DOLFINx and legacy `dolfin` do not agree on that
ordering, so those raw arrays cannot be safely reinterpreted under
DOLFINx.

The port's restart format (v2) instead stores, for each mesh vertex, its
physical coordinate and the magnetisation value there — stable across
finite-element ordering changes, and the same scheme `Field.save_hdf5`/
`.npy` snapshots use. A v1 (raw-dof) archive is rejected outright with a
clear error rather than silently misread: there was no safe way to map it
onto a new mesh, and no archive of that vintage needed supporting.

One thing restart v2 deliberately does **not** do: it reconstructs
magnetisation and simulation time, but does not reapply or validate the
material parameters and interactions (`Ms`, `A`, `K1`, alpha, which
interactions were added) in effect when the archive was written. The
restarting script must rebuild the same `Simulation` — same materials,
same interactions — before calling `restart()`, matching how restart was
actually used in the original codebase (nothing in the port's selected
workflows needed full state reconstruction). Any material/interaction
metadata written into an archive is therefore informational only, not read
back as state; for spatially varying fields it is a lossy scalar summary
and cannot reconstruct the original per-node values.

Register: [`D12`](superpowers/acceptance-register.md) (v2 only, v1 rejected),
[`D14`](superpowers/acceptance-register.md) (coordinate/value `.npy` format),
[`D16a`, `D16b`](superpowers/acceptance-register.md) (restart does not
reapply/validate material metadata; varying-field metadata is a lossy summary).

---

## 5. Places where the port's physics is correct and the original's was not

Three places surfaced where the frozen original finmag's behaviour was
itself wrong, and the port ships the corrected physics rather than
reproducing the historical bug. Each was investigated individually and
recorded as a decision, not changed silently.

### 5.1 `DiscreteTimeZeeman` energy no longer goes stale

In the original finmag, `DiscreteTimeZeeman.compute_energy()` could return
a stale value after the applied field was updated at a time step: the
original `update()` method rebound the field array (`self.H`) to the new
value but never rebuilt the cached energy (`self.E`) that
`compute_energy()` actually read, so the returned energy stayed frozen even
though the field was current. This was confirmed by reading the frozen
original source directly — the committed regression fixture for this
energy family turned out not sensitive enough to expose the bug
numerically (the correct energies are of order 1e-37 J, far below the
fixture's comparison tolerance), so it had gone undetected.

The port fixes this: the cached energy is rebuilt whenever the field is
updated, so `compute_energy()` always reflects the current field. Field
computation itself was never affected and remains correct in both
versions.

Register: [`D3`](superpowers/acceptance-register.md).

### 5.2 `LLG.M` and `LLG.M_average` are now correct physics in the right units

In the frozen original source, `LLG.M` (the magnetisation, as opposed to
the unit vector `m`) raised an exception on every access — it read
`self.m`, itself hard-coded to raise
`RuntimeError("DON'T USE llg.m UNTIL FURTHER NOTICE!!!!")`. Separately,
`LLG.M_average` computed a ratio of two integrals that were, on
inspection, identical expressions, collapsing the ratio to exactly `1` and
making `M_average` silently equal to `m_average` — a dimensionless
unit-vector average, not the magnetisation average in A/m the docstring
promised. No code in the original test suite or examples called `M` or
`M_average`, so there was no working reference behaviour to preserve.

The port implements the documented intent instead: `LLG.M = Ms * m` in
A/m, and `LLG.M_average` is the `Ms`-weighted volume average of `m`,
genuinely in A/m as documented.

Register: [`D20`](superpowers/acceptance-register.md).

### 5.3 The ~8% anisotropy/Magpar residual is mesh drift, not a field defect

The comparison test checking the ported uniaxial anisotropy field against a
checked-in reference computed by Magpar (a third-party micromagnetics code)
needed its tolerance loosened from the original's `5e-7` to `8e-2`; the
measured worst-case disagreement is about 5.1%. Because that maximum
happened to fall on a mesh node whose coordinates exactly coincide between
the two meshes — where no interpolation should be involved — this was
treated as a possible masked defect rather than accepted on trust, and a
dedicated physics investigation was run before recording a disposition.

Finding: both finmag and Magpar compute the anisotropy field with the same
"box" (lumped vertex-volume) method, which evaluates a weighted average of
the field over each node's surrounding tetrahedra, not a pointwise value.
That average depends on the local mesh shape, and the reference Magpar
mesh and the mesh the port regenerates today are no longer the same
tessellation (1537 nodes / 6886 tetrahedra in the archived 2017 Magpar run
versus 1202 / 4960 today — the same drift as §2), so even identical
box-method code gives slightly different nodal values on the two meshes,
largest at edge/corner nodes — which is also where the coincident-node set
concentrates, explaining why the worst residual lands there. Two checks
confirmed there is no defect: the port's field agrees with an independent
reimplementation of the box-method formula to `3.6e-16` (it *is* the
formula, to machine precision), and the residual's node-by-node size is
accurately predicted (correlation 0.999) purely from the geometric
difference between the two meshes, and forcing a uniform magnetisation
control (where that dependence should vanish) collapsed the residual to
machine precision on both meshes.

The `8e-2` tolerance stands as a deliberate allowance for mesh-regeneration
drift on a comparison against a foreign saved mesh, not a statement about
finmag's anisotropy accuracy — the field itself is separately validated at
machine-precision tolerances against the oracle and analytic results
elsewhere. Full derivation: the
[D29 verdict](archive/specs/2026-07-27-d29-verdict.md).

Register: [`D29`](superpowers/acceptance-register.md).

---

## 6. Ported tests live at their original file paths, and unported tests fail visibly by name

### The path convention

Early in the port, DOLFINx versions of tests were written as sibling files
next to the originals (e.g. `foo_dolfinx.py` beside the original `foo.py`).
This was changed: **every ported test now lives at the exact file path its
original occupied.** The sibling convention was retired because it defeated
the thing the port process most needed — reviewability. A port sitting
beside its original under a different filename requires manually hunting
down and comparing two files to check faithfulness. With the port at the
*same* path, that comparison is one ordinary command,
`git diff <original-commit>..HEAD -- <path>`, showing exactly what
changed and nothing else. That diff is the actual review artefact used
throughout the port, and is intended to remain how anyone — including
someone from the original team — checks a port's faithfulness in future.

### Why unported tests stay in the tree and fail on purpose

A test whose original has not been ported is not deleted. It is either
kept as a whole original file, or, for a partly-ported file, its specific
unported test functions are transcribed verbatim under a labelled banner.
Either way it is marked so it is expected to fail, with a message naming
the missing capability and the relevant register decision — e.g. "not
ported: nonlocal LLG_STT spin-accumulation model (register M5)". This is a
"strict" expected-failure marker: if the feature is later ported and the
test starts passing, the suite reports that as a **failure**, not a silent
pass, forcing the stale marker off. Nothing stays "carried but broken"
once the capability actually works.

The reasoning is the same reviewability requirement as the path
convention: a capability simply absent from the tree is invisible — no way
to tell "never ported" from "never needed". A capability that fails
loudly, by name, with a register reference, is discoverable and honest.
So: **every original test is accounted for at its original path, either
passing as a live port or failing by name as a documented gap** — no
third, silent option.

Register: [`D30`](superpowers/acceptance-register.md) (canonical test
paths), [`D33`](superpowers/acceptance-register.md) (the same mechanism
extended to whole never-ported files, so a raw import failure becomes a
named expected failure instead of an opaque collection error).

---

## 7. Energy assembly uses one method now, and it has a performance cost

### The interface change

The original finmag's energy interactions (`Exchange`, `DMI`,
`UniaxialAnisotropy`, and so on) supported five interchangeable numerical
methods for assembling the field, selected by a `method=` string,
defaulting to `"box-matrix-petsc"`. All five agreed to `1e-13` on the
original — implementation/performance variants of the same physics, not
different physics. The port keeps only `"box-assemble"`, as both the sole
supported method and the new default; the other four names
(`box-matrix-numpy`, `box-matrix-petsc`, `project`, `direct`) raise a clear
error naming the removed option. A script explicitly requesting a removed
name fails loudly rather than being silently remapped; a script relying on
the original default simply gets box assembly, with the same numbers.

Register: [`D32`](superpowers/acceptance-register.md) (the interface
change), [`M12a`, `M12b`, `M12c`](superpowers/acceptance-register.md) (the
three removed methods, dropped permanently).

### The performance consequence

Reducing to one method exposed a real performance problem, unrelated to
which method was chosen: the port's box-assembly implementation currently
rebuilds and reassembles the field operator **on every field evaluation**,
whereas the original's default `box-matrix-petsc` assembled that operator
**once**, at setup, and merely applied it thereafter. Applying a
precomputed operator is far cheaper than rebuilding it each time, and the
port currently does the expensive thing every time. This was measured on
the `std_prob_3` example: a full run of its bisection procedure takes
about 8.8 hours under the port, against the script's own header comment
claiming about 30 minutes historically — roughly a 17.6x gap. Whether that
comment was ever literally true is not independently known, but the root
cause of the slowness is understood: the repeated re-assembly above,
compounded by the port's plain NumPy right-hand-side evaluation in place
of the original's compiled `Equation` backend.

### The planned fix

A fix is planned but not yet executed: rebuild "assemble once, apply per
evaluation" as an internal optimisation of the existing `box-assemble`
method — the mechanism the original's `box-matrix-petsc` used — without
bringing back the removed `method=` names as a public option. It also
covers caching other per-call setup, warm-starting the demag solver, and
reducing RHS/ordering conversions if measurement shows they matter. Every
step must first prove, on a spatially-varying case, that the fast path
matches the slow path to about `1e-12` relative difference, with every
existing scientific test staying green — a performance change only, never
a physics change. See
[the performance plan](plans/2026-07-28-post-sr1-performance.md) for the
full breakdown.

Register: [`P1`](superpowers/acceptance-register.md).

---

## 8. Capability families deferred to later, and why

The following families of the original finmag are not yet available under
DOLFINx. None was dropped as unwanted — each remains part of the eventual
full-parity target, deferred for a specific, recorded reason.

**Normal-mode analysis** (eigensolvers, ringdown, FFT/power-spectral-density
analysis) is a substantial, largely self-contained numerical subsystem
with its own solver requirements; sequenced after the core serial
simulator because nothing else in the supported workflow set depends on
it.

**Thermal dynamics** (stochastic LLG/LLB and the `StochasticHeunIntegrator`
driver it needs) is a distinct numerical technique — stochastic time
integration — layered on the deterministic core; deferred as a whole
family, with the integrator held pending the same decision rather than
judged separately obsolete.

**General MPI (multi-rank) time stepping** is not yet supported: the port
is validated as a serial, deterministic simulator, and ownership-sensitive
parallel correctness for full time-stepping is a separate, larger
validation problem (some MPI ownership probes exist for specific pieces of
machinery; general parallel time stepping does not).

**Function-space periodic boundary conditions**
(`Simulation(pbc='1d'/'2d')`) were specifically investigated rather than
postponed by default: the supported DOLFINx environment lacks the pieces
the original relied on (`dolfinx_mpc` is absent; DOLFINx 0.10 has no
`constrained_domain` on `functionspace`), and a direct probe confirmed
periodicity cannot be recovered by copying values across the mesh boundary
afterwards — it must be built into the assembled exchange/DMI operator as
a genuine multipoint constraint. The separate demag-side periodicity
(`MacroGeometry` image lattice) already works and is unaffected; a request
for function-space PBC fails immediately, naming the feature.

**Nonlocal `LLG_STT`** (the spin-accumulation spin-transfer-torque model)
is a distinct numerical model from the *local* STT modes (Slonczewski and
Zhang-Li) the port fully supports, and was not part of the selected
first-release workflow set; a request for it fails by name.

See rows C16-C19 of the [capability-status matrix](superpowers/capability-status.md)
for each family's status. Register: [`M16`](superpowers/acceptance-register.md)
(thermal driver), [`D19`](superpowers/acceptance-register.md) (function-space
PBC), [`M5`](superpowers/acceptance-register.md) (nonlocal LLG_STT). Full
list: [docs/SUPPORTED.md §5](SUPPORTED.md).

---

## 9. Why there are several test tiers instead of one

Running the entire original test suite at full physical workload takes far
too long for every change (multiple hours; some individual examples alone
take hours under the not-yet-optimised assembly path in §7), so requiring
it on every push would make ordinary development unworkable. At the same
time, a single fast gate is not enough evidence to trust a release, and
silently skipping the slow tests forever would let real breakage go
unnoticed indefinitely. The port therefore runs different subsets on
different schedules, matched to cost and how likely a regression there is:

- A **fast gate** (`test-fast.yml`, ~13 minutes) runs on every push and
  pull request: roughly thirty focused test gates covering every ported
  source area, the comparison suite, MPI ownership probes and a core
  Sundials smoke test — the everyday signal a contributor sees.
- A **full-suite inventory sweep** (`test-python.yml`) runs weekly and on
  demand: it collects and runs the *entire* `src/finmag` test tree,
  including the unported-capability tests from §6, expected to report as
  named expected-failures, not passes. Its contract is that *unexpected*
  failures and collection errors must be zero; the expected-failure count
  is allowed to be large and is only the kept worklist of what remains to
  be ported, not a defect signal.
- A **slow lane** (`test-slow.yml`) runs the heavy, full-workload examples
  on demand only, since they do not fit an ordinary hosted runner's budget.

This keeps the everyday gate fast while the more complete checks still run
often enough to catch regressions. Detail and local commands: [docs/testing.md](testing.md).

---

## How the port was executed

The port was carried out with agent assistance (Codex and Claude Code),
with every committed code change reviewed by a human before it landed.
Each use of an agent is recorded in the corresponding commit message,
either directly or through a co-author line, so the provenance of any
specific change is traceable from the commit history itself.
