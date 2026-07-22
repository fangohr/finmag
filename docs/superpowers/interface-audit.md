# Public Interface Audit — Finmag DOLFINx Port

**Scope:** every public interface surface a user or `examples/` script touches,
across `src/finmag` and `examples/`.
**Yardstick:** the parity contract — the ported package exposes the SAME public
interface as legacy, changing it ONLY where DOLFINx/Py-3.12 forces it, each such
change documented + accepted.
**Commits compared** (three-way where lineage matters):
- Legacy oracle (frozen): pixi tip `ba9280934e188d7f3800e7b9865e70a9422f7687`
  (Python 3 + FEniCS 2019).
- Original master `b5015c5a` (Python 2 + dolfin-2017) — consulted where a
  "change" might predate the port.
- Ported: `dolfinx-parity` HEAD `b9006786`.

Read-only audit. No source was modified. Evidence is `git show <commit>:<path>`.

---

## 1. Summary verdict

**Surfaces audited: 45** (14 energy/demag constructors + families, 13 Simulation
methods/ctor + `sim_with`, 12 Field public methods + the ordering contract, 11
mesh generators + `from_geofile`/`from_csg` + `mesh_templates`; overlapping rows
consolidated).

| Class | Count |
|---|---|
| NO-CHANGE (interface identical to legacy) | 31 |
| FORCED (DOLFINx/Py-3.12 genuinely removed the legacy mechanism) | 5 |
| REGISTERED-DEVIATION (already in the acceptance register / drift table) | 3 |
| **UNNECESSARY (gratuitous — the findings)** | **0 (REVERTED — see §3)** |
| PRE-EXISTING / stale-example (not introduced by the port) | 1 (CubicAnisotropy) |

**Headline:** there WERE six gratuitous changes, but they were all of ONE benign
kind — **six first-positional-parameter RENAMES** with **zero practical blast
radius** (every caller in `examples/` and the test-suite, including internal
`src/finmag` callers, passed them positionally; not a single keyword caller
existed). No argument was reordered, no default that changes behaviour was
flipped without registration, no method silently renamed on a surface a user
relies on. All six have since been REVERTED to their legacy names (commit
"Revert 6 gratuitous parameter renames to legacy names", see §3) — the port is
now fully interface-faithful on this axis.

**CubicAnisotropy — the user's specific concern — is a MISATTRIBUTION.** The
class constructor is **byte-identical at all three commits**:

```
master  b5015c5a  src/finmag/energies/cubic_anisotropy.py:30
        def __init__(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy', assemble=False)
oracle  ba928093  src/finmag/energies/cubic_anisotropy.py:30
        def __init__(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy', assemble=False)
ported  b9006786  src/finmag/energies/cubic_anisotropy.py:98
        def __init__(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy', assemble=False)
```

The port **did not reorder CubicAnisotropy's arguments**. The `(K1, u1, K2, u2,
K3, u3)` order the user remembers lived only in the *stale example*
`examples/cubic_anisotropy/hysteresis.py`, which called
`CubicAnisotropy(K1, u1, K2, u2, K3, u3)` at BOTH master and pixi
(`git show b5015c5a:examples/cubic_anisotropy/hysteresis.py:31` and
`git show ba928093:...:31` are identical) — i.e. that example was already
inconsistent with its own contemporaneous class signature (and the *sibling*
example `examples/cubic_anisotropy/sim.py:54` at the oracle already used the
correct `(u1, u2, K1)` order). The port's edit to `(u1, u2, K1, K2, K3)` is a
**stale-example correction to the long-standing legacy signature**, not a
port-induced interface change. Verdict: **pre-existing / misattribution**.

---

## 2. Full interface table

Legend for "needed?": Y = interface identical or change genuinely forced/registered;
N = gratuitous (UNNECESSARY).

### Energy & demag constructors

| Surface | Legacy signature (pixi `ba928093`) | Ported signature (`b9006786`) | Class | Needed? | Revert if UNNECESSARY |
|---|---|---|---|---|---|
| `Exchange` | `(A, method='box-matrix-petsc', name='Exchange')` | `(A, method='box-assemble', name='Exchange')` | REGISTERED | Y | — (Task 5: only `box-assemble` supported; others raise `NotImplementedError`) |
| `UniaxialAnisotropy` | `(K1, axis, K2=0, method='box-matrix-petsc', name='Anisotropy', assemble=True)` | `(K1, axis, K2=0, method='box-assemble', name='Anisotropy', assemble=True)` | REGISTERED | Y | — (Task 5, same as above) |
| `CubicAnisotropy` | `(u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy', assemble=False)` | identical | NO-CHANGE | Y | — |
| `DMI` | `(D, method='box-matrix-petsc', name='DMI', dmi_type='auto')` | `(D, method='box-assemble', name='DMI', dmi_type='auto')` | REGISTERED | Y | — (Task 5) |
| `Zeeman` | `(H, name='Zeeman', **kwargs)` | identical | NO-CHANGE | Y | — |
| `DipolarField` | `(pos, m, magnitude=None, name='DipolarField')` | identical | NO-CHANGE | Y | — |
| `TimeZeeman` | `(field_expression, t_off=None, name='TimeZeeman')` | identical (signature); `field_expression` now a Python callable, not a dolfin `Expression` | FORCED | Y | — (Expression mechanism removed by DOLFINx; documented Task 15) |
| `DiscreteTimeZeeman` | `(field_expression, dt_update=None, t_off=None, name='DiscreteTimeZeeman')` | identical | NO-CHANGE | Y | — (stale-energy legacy bug preserved, register #3) |
| `TimeZeemanPython` | `(df_expression, time_fun, t_off=None, name='TimeZeemanPython')` | `(H0_value, time_fun, t_off=None, name='TimeZeemanPython')` | FORCED | Y | — (1st param renamed to reflect the forced Expression→value contract; documented transition-notes.org:4368) |
| `OscillatingZeeman` | `(H0, freq, phase=0, t_off=None, name='OscillatingZeeman')` | identical | NO-CHANGE | Y | — |
| `ThinFilmDemag` | `(direction='z', field_strength=None, in_jacobian=False, name='ThinFilmDemag')` | identical | NO-CHANGE | Y | — |
| `Demag` | `Demag(solver='FK', *args, **kwargs)` | identical | NO-CHANGE | Y | — |
| `FKDemag` | `(name='Demag', thin_film=False, macrogeometry=None, solver_type=None, parameters=None)` | identical | NO-CHANGE | Y | — |
| `MacroGeometry` | `(nx=None, ny=None, dx=None, dy=None, Ts=None)` | identical | NO-CHANGE | Y | — |
| `EnergyBase` (base ctor) | `(method='box-matrix-petsc', in_jacobian=False)` | `(method='box-assemble', in_jacobian=False)` | REGISTERED | Y | — (Task 5) |

### Simulation surface + `sim_with`

| Surface | Legacy (pixi) | Ported | Class | Needed? | Revert if UNNECESSARY |
|---|---|---|---|---|---|
| `Simulation.__init__` | `(mesh, Ms, unit_length=1, name='unnamed', kernel='llg', integrator_backend='sundials', pbc=None, average=False, parallel=False)` | same but `integrator_backend='scipy'` | REGISTERED | Y | — (register #8 / Task 20, USER ACCEPTANCE PENDING; factory default IS sundials) |
| `sim_with` | `(mesh, Ms, m_init, alpha=0.5, unit_length=1, integrator_backend='sundials', A=None, K1=None, K1_axis=None, H_ext=None, demag_solver='FK', demag_solver_type=None, nx=None, ny=None, spacing_x=None, spacing_y=None, demag_solver_params={}, D=None, name='unnamed', pbc=None, sim_class=Simulation)` | same but `integrator_backend='scipy'` and `demag_solver_params=None` | REGISTERED (backend) + benign | Y | — (backend = register #8; `{}`→`None` is a mutable-default-arg fix, behaviourally identical) |
| `set_m` | `(value, normalise=True, **kwargs)` | identical | NO-CHANGE | Y | — |
| `add` | `(interaction, with_time_update=None)` | identical | NO-CHANGE | Y | — |
| `run_until` | `(t)` | identical | NO-CHANGE | Y | — |
| `m_average` (property) | — | identical | NO-CHANGE | Y | — |
| `effective_field` | `()` | identical | NO-CHANGE | Y | — |
| `schedule` | `(func, *args, **kwargs)` | identical | NO-CHANGE | Y | — |
| `set_stt` | `(current_density, polarisation, thickness, direction, Lambda=2, epsilonprime=0.0, with_time_update=None)` | identical | NO-CHANGE | Y | — |
| `toggle_stt` | `(new_state=None)` | identical | NO-CHANGE | Y | — |
| `set_zhangli` | `(J_profile=(1e10,0,0), P=0.5, beta=0.01, using_u0=False, with_time_update=None)` | identical | NO-CHANGE | Y | — |
| `relax` / `hysteresis` / `hysteresis_loop` | (bound to legacy `sim_relax.py`/`hysteresis.py`) | signatures preserved | NO-CHANGE | Y | — (no-re-relax legacy defect preserved, register #4) |
| `mark_regions` | `(fun_regions)` | `(fun_regions)` | **REVERTED** | Y | done — see "Revert 6 gratuitous parameter renames to legacy names" |

### Field public API

| Surface | Legacy (pixi) | Ported | Class | Needed? | Revert |
|---|---|---|---|---|---|
| `from_callable` | `(func)` | `(func)` | **REVERTED** | Y | done — see "Revert 6 gratuitous parameter renames to legacy names" |
| `from_constant` | `(constant)` | `(constant)` | **REVERTED** | Y | done — see "Revert 6 gratuitous parameter renames to legacy names" |
| `from_array` | `(arr)` | `(arr)` | **REVERTED** | Y | done — see "Revert 6 gratuitous parameter renames to legacy names" |
| `from_sequence` | `(seq)` | `(seq)` | **REVERTED** | Y | done — see "Revert 6 gratuitous parameter renames to legacy names" |
| `from_field` | `(field)` | identical | NO-CHANGE | Y | — |
| `from_function` | `(function)` | identical (interpolation superset added, register #7) | NO-CHANGE | Y | — |
| `set` / `set_with_numpy_array_debug` / `as_array` / `as_vector` / `is_constant` / `as_constant` / `mesh*` / `value_dim` / `coords_and_values` / `cross` / `dot` / `coerce_scalar_field` | (as legacy) | identical | NO-CHANGE | Y | — |
| `average` | `(dx=df.dx)` | `(dx=dx)` (ufl `dx`) | FORCED | Y | — (dolfin `dx` object removed) |
| `set_random_values` | `(vrange=[-1, 1])` | `(vrange=(-1.0, 1.0))` | benign | Y | — (mutable-default fix; identical behaviour) |
| `save_pvd` | `(filename)` | `(filename, t=0.0)` | superset | Y | — (backward-compatible optional arg) |
| `from_expression` | `(expr, **kwargs)` | raises `NotImplementedError` | FORCED | Y | — (dolfin `Expression`/`UserExpression` removed) |
| `probe` / `__call__` (point eval) | `(coord)` evaluates at a point | raises `NotImplementedError` | FORCED | Y | — (point-in-cell eval not provided; drift #4/#12) |
| `save_hdf5` / `close_hdf5` / `plot_with_*` / `get_spherical` | (as legacy) | deferred / raise | FORCED/deferred | Y | — (dolfinh5tools / mayavi / paraview not in stack) |
| **Public array component ORDERING** (`get_numpy_array_debug`, `get_ordered_numpy_array_xxx`, every `compute_field()`, `effective_field()`) | component-blocked `[x1..xN, y1..yN, z1..zN]` | **CORRECTED (Task 31)** back to component-blocked | REGISTERED-corrected | Y | — (drift #3/#6 de-registered; now legacy-faithful, no residual) |

### Meshes, `from_geofile`/`from_csg`, `mesh_templates`

| Surface | Legacy (pixi) | Ported | Class | Needed? | Revert |
|---|---|---|---|---|---|
| `box`, `sphere`, `cylinder`, `nanodisk`, `elliptic_cylinder`, `elliptical_nanodisk`, `ellipsoid`, `ring`, `truncated_cone`, `pair_of_disks` | `(...geom, maxh, save_result=True, filename='', directory='')` (+ `ring` `with_middle_plane=False`) | identical | NO-CHANGE | Y | — |
| `from_geofile` | `(geofile, save_result=True)` | `(geofile, save_result=True, filename='', directory='', *, maxh=None)` | superset | Y | — (added optional args; `from_geofile(f, False)` still binds `save_result`; maxh keyword-only by design) |
| `from_csg` | `(csg, save_result=True, filename='', directory='')` | `(csg, save_result=True, filename='', directory='', *, maxh=None)` | **REVERTED** (1st param) | Y | done — see "Revert 6 gratuitous parameter renames to legacy names" (the added keyword-only `maxh` is fine and unaffected) |
| `mesh_templates`: `MeshTemplate`, `MeshSum`, `MeshDifference`, `Sphere`, `Box`, `EllipticalNanodisk`, `Nanodisk` | (as legacy) | identical | NO-CHANGE | Y | — |
| deferred generators (`elliptical_nanodisk_with_cuboid_shell`, `sphere_inside_box`, `disk_with_internal_layers`, `regular_polygon*`, `line_mesh`, `embed3d`, …) | present | not ported (out of scope) | deferred | Y | — (unported, not a change to a preserved surface) |

---

## 3. UNNECESSARY findings (the actionable output) — REVERTED

**Status: all six REVERTED.** See commit "Revert 6 gratuitous parameter renames
to legacy names" (branch `dolfinx-parity`, on top of `70df2938`). The ported
public signatures now match the legacy oracle (pixi `ba928093`) exactly for all
six surfaces below; each revert touched the `def` line plus in-body uses (and,
for `Field.from_constant`, a body-local variable was renamed from `constant` to
`constant_arr` to avoid shadowing the now-restored `constant` parameter — an
implementation-only detail, not part of the public interface).

All six had been the **same benign class**: the FIRST POSITIONAL parameter of a
method was renamed for no reason DOLFINx required. Nothing was reordered.
**Blast radius was zero in practice** — `grep` across `examples/` and the entire
test-suite (including internal `src/finmag` callers) found **no keyword caller**
of any of these params; every call site passed the argument positionally
(verified: no `fun_regions=`/`func=`/`constant=`/`arr=`/`seq=`/`csg=` keyword
call existed before or after the revert). No caller needed updating.

| # | Surface | File:line (ported) | Change | Status | Blast radius |
|---|---|---|---|---|---|
| U1 | `Simulation.mark_regions` | `src/finmag/sim/sim.py:675` | `fun` → `fun_regions` | REVERTED | none — ported tests call positionally (`test_variable_params_dolfinx.py`, `test_simulation_dolfinx.py`); legacy `sim_test.py:1729` passes a positional variable named `fun_regions` |
| U2 | `Field.from_callable` | `src/finmag/field.py:99` | `function` → `func` | REVERTED | none |
| U3 | `Field.from_constant` | `src/finmag/field.py:105` | `value` → `constant` | REVERTED | none |
| U4 | `Field.from_array` | `src/finmag/field.py:158` | `array` → `arr` | REVERTED | none |
| U5 | `Field.from_sequence` | `src/finmag/field.py:183` | `sequence` → `seq` | REVERTED | none |
| U6 | `from_csg` | `src/finmag/util/geofile.py:525` | `csg_string` → `csg` | REVERTED | none |

Note: these are cosmetic and defensible as "clearer names," but under the strict
parity contract they are interface deltas that DOLFINx did not force, so they are
correctly classified UNNECESSARY. If the controller decides clarity outweighs
strict parity, they can be ACCEPTED-and-registered instead of reverted — but they
should not remain silent.

**Explicitly NOT flagged as UNNECESSARY** (behaviour-preserving defensive fixes,
called out for transparency, not action):
- `sim_with(demag_solver_params={})` → `None` — Python mutable-default-arg
  anti-pattern fix; the body converts `None` to `{}`. No behaviour change.
- `Field.set_random_values(vrange=[-1,1])` → `(-1.0,1.0)` — list→tuple default;
  identical numeric behaviour.
- `Field.save_pvd(filename)` → `(filename, t=0.0)` and `from_geofile`'s added
  `filename`/`directory`/`maxh` — pure supersets; every legacy call still binds.

---

## 4. FORCED changes (the minimal TRUE interface delta)

These are the only user-visible interface changes DOLFINx/Py-3.12 genuinely
compelled — the irreducible cost of the port:

1. **String `Expression` → Python callable.** Anywhere legacy accepted a dolfin
   `df.Expression("...")` (initial `m`, anisotropy `axis`, `alpha`, applied
   field, `TimeZeeman.field_expression`, `Field.from_expression`), the port
   accepts only a vectorised NumPy callable. DOLFINx has no `Expression`/
   `UserExpression`. Consequence: `Field.from_expression` raises
   `NotImplementedError`; `TimeZeemanPython`'s 1st param was renamed
   `df_expression`→`H0_value` to name the new value contract (transition-notes:4368).
2. **Point evaluation removed.** `Field.probe(coord)` / `Field.__call__(pt)` /
   `energy_density_function()([x,y,z])` raise `NotImplementedError`; examples read
   nodal data via `coords_and_values()` or explicit DOLFINx point-in-cell eval
   (drift #4, #12).
3. **dolfin `dx` object gone.** `Field.average(dx=df.dx)` → `dx=` the ufl/DOLFINx
   measure. Same call shape, different default object.
4. **HDF5 read-back / legacy plotting gone.** `save_hdf5`/`close_hdf5`
   (dolfinh5tools), `plot_with_dolfin`, `plot_with_paraview`, `get_spherical`
   are deferred/raise — those backends are not in the DOLFINx/Py-3.12 stack.
5. **`method='box-matrix-petsc'` legacy assembly algorithms removed.** The
   NumPy-matrix / project / direct energy paths do not exist under DOLFINx; the
   only supported algorithm is `box-assemble`, which is now the default for
   `Exchange`/`UniaxialAnisotropy`/`DMI`/`EnergyBase`. Every other `method=` value
   raises `NotImplementedError`. (Registered, Task 5 — porting_map.md:261-266.)

Registered deviations that are NOT strictly forced but are documented + pinned +
awaiting sign-off: `Simulation.integrator_backend` default `'scipy'` (register
#8; factory default is legacy `'sundials'`), plus the behavioural-quirk register
entries (#1–#10) that do not alter signatures.

---

## 5. Examples doubled-findings

**None.** No `examples/` edit exists solely because a src interface changed
gratuitously. Cross-checking every converted example against the six UNNECESSARY
renames: not one example uses any of the renamed parameters by keyword, so no
example was touched to accommodate U1–U6.

Every non-mechanical example edit traces to a FORCED change or a stale example,
not to a gratuitous src change (per the `transition-notes.org` drift table, rows
1–12):
- Expression rewrites (drift #2), point-probe workarounds (#4, #12), ordering
  (#3/#6 — now CORRECTED so the examples use the legacy `reshape((3,-1))` /
  `get_numpy_array_debug()` verbatim), `llg.m_numpy`/`llg.alpha` state access
  (#5, #7), `hysteresis(list)` vs py2 `ndarray==[]` (#8), `mesh_info` →
  `num_vertices` (#9) — all FORCED or unported-surface workarounds.
- The one `CubicAnisotropy` example edit
  (`examples/cubic_anisotropy/hysteresis.py`,
  `(K1,u1,K2,u2,K3,u3)`→`(u1,u2,K1,K2,K3)`) is a **stale-example fix**, not a
  doubled finding: the class signature was unchanged by the port (§1); the
  example had been inconsistent with the legacy signature since master.

---

## Appendix — evidence commands

```
# CubicAnisotropy three-way (identical):
git show b5015c5a:src/finmag/energies/cubic_anisotropy.py            # :30
git show ba928093...:src/finmag/energies/cubic_anisotropy.py         # :30
git show b9006786:src/finmag/energies/cubic_anisotropy.py            # :98
# Stale example proof (same broken call at master AND pixi):
git show b5015c5a:examples/cubic_anisotropy/hysteresis.py            # :31 (K1,u1,K2,u2,K3,u3)
git show ba928093...:examples/cubic_anisotropy/hysteresis.py         # :31 (identical)
git show ba928093...:examples/cubic_anisotropy/sim.py                # :54 (u1,u2,K1) — sibling already correct
# UNNECESSARY renames:
git show ba928093...:src/finmag/sim/sim.py | sed -n '1322p'          # mark_regions(fun_regions)
git show b9006786:src/finmag/sim/sim.py | sed -n '675p'              # mark_regions(fun)
git show ba928093...:src/finmag/field.py                             # from_callable(func) / from_constant(constant) / from_array(arr) / from_sequence(seq)
git show b9006786:src/finmag/field.py                                # from_callable(function) / from_constant(value) / from_array(array) / from_sequence(sequence)
git show ba928093...:src/finmag/util/meshes.py | sed -n '116p'       # from_csg(csg,...)
git show b9006786:src/finmag/util/geofile.py | sed -n '525p'         # from_csg(csg_string,...)
# No keyword callers of any renamed param:
grep -rn 'fun_regions=\|from_callable( *func=\|from_constant( *constant=\|from_array( *arr=\|from_sequence( *seq=\|from_csg( *csg=' examples/ src/finmag/   # -> empty
```

## CubicAnisotropy argument-order concern — RESOLVED (user review 2026-07-22)

Direct `git show` evidence: `CubicAnisotropy.__init__` is
`(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy', assemble=False)`
**byte-identical** at master `b5015c5a`, pixi oracle `ba928093`, and the port
(current HEAD). Master's own `cubic_anisotropy_test.py:43` calls
`CubicAnisotropy(u1, u2, K1, K2, K3)`, matching the class. Only master's
`examples/cubic_anisotropy/hysteresis.py:31` used
`CubicAnisotropy(K1, u1, K2, u2, K3, u3)` — which did **not** match its own
class on master (integer `K1` passed where tuple `u1` is expected; a stale,
almost certainly never-run Python-2 example). The port kept the class
unchanged and corrected the stale example call. **Verdict: not a
port-introduced interface change; user confirmed keep-as-is.**

(Record note: this resolution was first written to the working tree during a
concurrent revert slice and was discarded by that slice's clean-tree guard;
re-committed here on a clean tree. Lesson logged: the controller must not edit
tracked files while an implementer subagent owns the working tree.)
