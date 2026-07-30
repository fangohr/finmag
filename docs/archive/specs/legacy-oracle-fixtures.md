> **ARCHIVED (2026-07-29).** Historical record of the porting process; statements reflect their writing date. Current truth: docs/README.md.

# Legacy Oracle Reference Fixtures

## Immutable Oracle

Focused legacy results are generated from commit
`ba9280934e188d7f3800e7b9865e70a9422f7687`, using its checked-in
`pixi.lock`. Run a command with:

```bash
dev/bin/run-legacy-oracle -- pixi run --locked import-finmag
```

The helper creates a detached Git worktree below `mktemp -d`, clears inherited
`PYTHONPATH` and `PYTHONHOME`, executes the supplied argv without `eval`, and
removes the worktree on success, failure, or a handled signal. It compares the
active checkout's complete tracked diff before and after the command. Status
messages go to stderr, so a reference generator's stdout can be captured
without contamination:

```bash
dev/bin/run-legacy-oracle -- \
  pixi run --locked python -c \
  'import json; print(json.dumps({"oracle": "ba9280934e188d7f3800e7b9865e70a9422f7687"}, sort_keys=True))' \
  > /tmp/reference.json
```

Generate into `/tmp` first, inspect the result, and then add an intentional
fixture separately. Do not redirect straight into the active checkout: the
shell opens the destination before the helper can snapshot tracked state, and
new fixture files are untracked. The guard detects mutations made while the
oracle command is running; it cannot police caller-side redirection.

A referenced generator script or task must already exist at the oracle commit.
Do not reach into the active checkout to run generator code added later: that
would freeze the old Finmag implementation but not the calculation used to
extract its result. For a new focused calculation, use recorded inline Python
as above or an existing oracle module/test, and preserve the exact argv in the
fixture.

The Docker/DOLFIN-2017 witnesses are not interchangeable with this oracle.
The immutable oracle is the Python-3/FEniCS-2019 Pixi commit above. A future
container may execute the helper's command only if it reproduces that locked
environment; the current M1 images do not.

## JSON Schema Version 1

Fixtures are small JSON documents with this shape:

```json
{
  "schema_version": 1,
  "oracle": {
    "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
    "command": ["pixi", "run", "--locked", "reference-task"]
  },
  "mesh": {
    "recipe": "dolfin.IntervalMesh(1, 0.0, 0.5)",
    "parameters": {"cells": 1, "x0": 0.0, "x1": 0.5}
  },
  "physical_parameters": {
    "Ms": {"value": 800000.0, "unit": "A/m"},
    "unit_length": {"value": 1e-9, "unit": "m"}
  },
  "coordinates": {
    "unit": "mesh_coordinate",
    "ordering": "lexicographic_xyz",
    "values": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
  },
  "quantities": [
    {
      "name": "effective_field",
      "unit": "A/m",
      "value_shape": [3],
      "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      "tolerances": {"absolute": 1e-10, "relative": 1e-10}
    }
  ]
}
```

Required semantics:

- `oracle.commit` is the exact 40-character commit above; `oracle.command` is
  the argv used to generate the data, not an opaque shell command.
- `mesh.recipe` is executable or unambiguous enough to reconstruct the mesh;
  all variable recipe inputs are repeated in `mesh.parameters`.
- Every physical parameter and result has an explicit unit. Use `"1"` for a
  dimensionless value. `mesh_coordinate` means coordinates are in the mesh's
  native coordinate system; `unit_length` records its physical scale.
- `coordinates.values` contains unique points padded to `(x, y, z)` and sorted
  by ascending lexicographic `(x, y, z)` order. Each quantity has one value row
  per coordinate. `value_shape` is `[]` for a scalar and `[n]` for an
  `n`-component vector.
- In MPI runs, emit owned rows only on each rank, gather them to rank zero,
  reject conflicting duplicate coordinates, globally deduplicate identical
  rows, and then lexicographically sort. Ghost values must never contribute a
  second row. Prefer a serial fixture unless partitioning itself is the
  behavior under test.
- `tolerances.absolute` is expressed in the quantity's unit and
  `tolerances.relative` is dimensionless. Tolerances must come from numerical
  conditioning or an established scientific test, not from the observed
  difference being accommodated.

Raw legacy degree-of-freedom arrays are forbidden: their ordering is an
implementation detail and is not comparable to DOLFINx layouts. A generator
must pair values with coordinates and sort those pairs. This version of the
schema intentionally covers unique nodal coordinates; discontinuous/entity-
valued data requires a future schema version with explicit entity identity.

## When Not to Create a Fixture

Prefer an analytic test when a closed-form result completely exercises the
scientific invariant, including sign, scaling, and units. Examples include a
zero exchange field for constant magnetisation, `H_eff == H` for uniform
Zeeman field, and parallel/perpendicular uniaxial-anisotropy states. A fixture
would only duplicate the formula and add stale data in those cases.

Use an oracle fixture when legacy behavior is material but no practical
analytic result isolates it—for example, a nonuniform assembled field, a
coordinate-mapped LLG right-hand side, or a demagnetising-field reference.
Keep the smallest mesh and quantity set that distinguishes the behavior under
test. Do not generate a fixture merely to make a failing port pass.
