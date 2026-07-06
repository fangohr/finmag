# Finmag DOLFINx Core Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a compatibility-shaped `finmag` package under `dev/dolfinx/finmag` that runs the first physical DOLFINx simulation workflow while leaving legacy `src/finmag` unchanged.

**Architecture:** Mirror the legacy package boundaries—`Field`, interaction classes, `EffectiveField`, `LLG`, drivers, and `Simulation`—inside the staged replacement package. Reuse the legacy API and numerical definitions, translate only the FEM operations to DOLFINx, and use the existing prototype solely as implementation evidence.

**Tech Stack:** Python 3.12, DOLFINx 0.10.x, UFL, PETSc/mpi4py, NumPy, SciPy VODE/BDF, pytest, pixi.

---

## File Structure

Create the staged package without changing `src/finmag`:

```text
dev/dolfinx/finmag/
  __init__.py
  constants.py
  field.py
  errors.py
  energies/
    __init__.py
    energy_base.py
    exchange.py
    zeeman.py
    anisotropy.py
  physics/
    __init__.py
    effective_field.py
    llg.py
  drivers/
    __init__.py
    llg_integrator.py
    scipy_integrator.py
  sim/
    __init__.py
    sim.py
  tests/
    test_import.py
    test_field.py
    test_energies.py
    test_effective_field.py
    test_llg.py
    test_drivers.py
    test_simulation.py
```

The existing files directly under `dev/dolfinx` remain unchanged during this
slice.

### Task 1: Align the repository strategy and environment

**Files:**
- Modify: `pixi.toml`
- Modify: `pixi.lock`
- Modify: `plan.org`
- Modify: `transition-notes.org`
- Modify: `agents.md`
- Modify: `dev/dolfinx/README.md`
- Modify: `dev/dolfinx/porting_map.md`
- Modify: `.github/workflows/dolfinx-m4.yml`
- Modify: `dev/bin/verify-dolfinx-m4`

- [ ] **Step 1: Pin the verified FEM stack and add the staged-package tasks**

Change the DOLFINx feature to:

```toml
[feature.dolfinx.dependencies]
python = "3.12.*"
fenics-dolfinx = "0.10.*"
pytest = "*"
numpy = "*"
scipy = "*"

[feature.dolfinx.tasks]
dolfinx-import = "python -c \"from mpi4py import MPI; import dolfinx; print('dolfinx', dolfinx.__version__, 'rank', MPI.COMM_WORLD.rank)\""
dolfinx-smoke = "python -c \"from mpi4py import MPI; import numpy as np; from dolfinx import mesh, fem; m = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2); V = fem.functionspace(m, ('Lagrange', 1)); u = fem.Function(V); u.interpolate(lambda x: x[0] + 2*x[1]); print('dofs', V.dofmap.index_map.size_global, 'sum', float(np.sum(u.x.array)))\""
dolfinx-pytest = "python -m pytest -q dev/dolfinx"
dolfinx-finmag-import = "PYTHONPATH=dev/dolfinx python -c \"import finmag; print(finmag.__file__)\""
dolfinx-finmag-pytest = "PYTHONPATH=dev/dolfinx python -m pytest -q dev/dolfinx/finmag"
dolfinx-finmag-smoke = "PYTHONPATH=dev/dolfinx python -m finmag.tests.smoke"
```

- [ ] **Step 2: Refresh and verify the lock file**

Run:

```bash
pixi install -e dolfinx
pixi run -e dolfinx python -c "import dolfinx, scipy; assert dolfinx.__version__.startswith('0.10.'); print(dolfinx.__version__, scipy.__version__)"
```

Expected: DOLFINx reports a `0.10.x` version and SciPy imports successfully.

- [ ] **Step 3: Correct the canonical migration documentation**

Add the same explicit statement to the roadmap, transition notes, agent
orientation, prototype README, and porting map:

```text
The existing modules directly under dev/dolfinx are an exploration lane.
The compatibility-shaped replacement is developed as dev/dolfinx/finmag while
legacy src/finmag remains unchanged and runnable. Promotion to src/finmag is a
separate final operation after the accepted scientific and workflow gates pass.
```

Update the DOLFINx workflow and verifier so they run the existing prototype
gate and the new staged-package import/test/smoke tasks as separate commands.

- [ ] **Step 4: Verify documentation and configuration integrity**

Run:

```bash
git diff --check
rg -n "dev/dolfinx/finmag|src/finmag remains unchanged" \
  plan.org transition-notes.org agents.md \
  dev/dolfinx/README.md dev/dolfinx/porting_map.md
```

Expected: no whitespace errors and every canonical migration document describes
the staging boundary.

- [ ] **Step 5: Commit the strategy correction**

```bash
git add pixi.toml pixi.lock plan.org transition-notes.org agents.md \
  dev/dolfinx/README.md dev/dolfinx/porting_map.md \
  .github/workflows/dolfinx-m4.yml dev/bin/verify-dolfinx-m4
git commit -m "docs: define staged DOLFINx Finmag replacement"
```

### Task 2: Scaffold an independently importable replacement package

**Files:**
- Create: `dev/dolfinx/finmag/__init__.py`
- Create: `dev/dolfinx/finmag/constants.py`
- Create: `dev/dolfinx/finmag/errors.py`
- Create: `dev/dolfinx/finmag/tests/__init__.py`
- Create: `dev/dolfinx/finmag/tests/test_import.py`

- [ ] **Step 1: Write the failing import-boundary test**

```python
# dev/dolfinx/finmag/tests/test_import.py
from pathlib import Path


def test_imports_staged_finmag_without_legacy_dolfin():
    import finmag

    package_path = Path(finmag.__file__).resolve()
    assert "dev/dolfinx/finmag" in package_path.as_posix()
    assert finmag.FEM_BACKEND == "dolfinx"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_import.py
```

Expected: collection fails because the staged `finmag` package does not exist.

- [ ] **Step 3: Add the minimal package identity**

```python
# dev/dolfinx/finmag/constants.py
from math import pi

MU0 = 4.0 * pi * 1e-7
GAMMA = 2.210173e5
```

```python
# dev/dolfinx/finmag/errors.py
class UnknownInteraction(ValueError):
    def __init__(self, requested, available):
        super().__init__(
            "Unknown interaction {!r}; available interactions: {}".format(
                requested, ", ".join(sorted(available)) or "<none>"
            )
        )
```

```python
# dev/dolfinx/finmag/__init__.py
FEM_BACKEND = "dolfinx"
```

Create empty `dev/dolfinx/finmag/tests/__init__.py`.

- [ ] **Step 4: Run the test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Commit the package boundary**

```bash
git add dev/dolfinx/finmag
git commit -m "feat: scaffold staged DOLFINx Finmag package"
```

### Task 3: Port the legacy `Field` abstraction

**Files:**
- Create: `dev/dolfinx/finmag/field.py`
- Create: `dev/dolfinx/finmag/tests/test_field.py`
- Reference only: `src/finmag/field.py`
- Reference only: `dev/dolfinx/field_adapter.py`

- [ ] **Step 1: Write failing compatibility tests**

Tests must cover the first required legacy surface:

```python
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.field import Field


@pytest.fixture
def domain():
    return mesh.create_unit_square(MPI.COMM_SELF, 2, 2)


def test_vector_field_set_normalise_and_average(domain):
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    field = Field(space, (2.0, 0.0, 0.0), normalised=True, name="m")

    assert field.name == "m"
    assert field.value_dim() == 3
    assert np.allclose(field.average(), (1.0, 0.0, 0.0))
    values = field.get_ordered_numpy_array_xyz().reshape((-1, 3))
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)


def test_scalar_field_volume_average(domain):
    space = fem.functionspace(domain, ("DG", 0))
    field = Field(space, 3.5)
    assert field.is_scalar_field()
    assert field.average() == pytest.approx(3.5)


def test_xyz_order_round_trip(domain):
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    field = Field(space, (1.0, 0.0, 0.0))
    ordered = np.arange(field.as_array().size, dtype=float)
    field.set_with_ordered_numpy_array_xyz(ordered)
    assert np.array_equal(field.get_ordered_numpy_array_xyz(), ordered)


def test_zero_vector_cannot_be_normalised(domain):
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    with pytest.raises(ValueError, match="zero"):
        Field(space, (0.0, 0.0, 0.0), normalised=True)
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_field.py
```

Expected: collection fails because `finmag.field` does not exist.

- [ ] **Step 3: Implement the minimal DOLFINx `Field`**

Implement `Field` with the legacy attributes `functionspace`, `f`, `name`, and
`unit`, plus:

```python
class Field:
    def __init__(
        self, functionspace, value=None, normalised=False, name=None, unit=None
    ):
        self.functionspace = functionspace
        self.f = fem.Function(functionspace)
        self.name = name
        self.unit = unit
        if name is not None:
            self.f.name = name
        if value is not None:
            self.set(value, normalised=normalised)

    def set(self, value, normalised=False, **kwargs):
        if isinstance(value, Field):
            return self.from_field(value)
        if isinstance(value, fem.Function):
            return self.from_function(value)
        if callable(value):
            self.f.interpolate(value)
            self.f.x.scatter_forward()
            if normalised:
                self.normalise()
            return self
        return self.from_constant(value, normalised=normalised)

    def as_array(self):
        return self.f.x.array.copy()

    @property
    def np(self):
        return self.as_array()
```

Also implement constant/callable assignment, same-space function copying,
field interpolation, scalar/vector inspection, volume-average form assembly,
raw array assignment, per-node normalisation, and coordinate-derived
mesh-vertex permutation. `get_ordered_numpy_array_xyz` and its setter return
and accept flat, node-interleaved arrays to match the legacy API.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2.

Expected: all field tests pass without warnings.

- [ ] **Step 5: Run the existing adapter tests as a regression witness**

```bash
pixi run -e dolfinx python -m pytest -q dev/dolfinx/test_field_adapter.py
```

Expected: existing prototype adapter tests remain green.

- [ ] **Step 6: Commit the field port**

```bash
git add dev/dolfinx/finmag/field.py dev/dolfinx/finmag/tests/test_field.py
git commit -m "feat: port Finmag Field to DOLFINx"
```

### Task 4: Port nodal-volume and `EnergyBase` behavior

**Files:**
- Create: `dev/dolfinx/finmag/energies/__init__.py`
- Create: `dev/dolfinx/finmag/energies/energy_base.py`
- Create: `dev/dolfinx/finmag/tests/test_energies.py`
- Reference only: `src/finmag/energies/energy_base.py`

- [ ] **Step 1: Write failing box-method tests**

```python
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.energies.energy_base import nodal_volume


def test_scalar_nodal_volume_sums_to_mesh_volume():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    space = fem.functionspace(domain, ("Lagrange", 1))
    assert nodal_volume(space).sum() == pytest.approx(1.0)


def test_vector_nodal_volume_repeats_volume_per_component():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    volume = nodal_volume(space).reshape((-1, 3))
    assert np.allclose(volume.sum(axis=0), (1.0, 1.0, 1.0))
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_energies.py
```

Expected: import fails because `finmag.energies` does not exist.

- [ ] **Step 3: Implement owned-dof-safe lumped volume assembly**

```python
def nodal_volume(function_space):
    domain = function_space.mesh
    test = ufl.TestFunction(function_space)
    value_size = int(np.prod(function_space.element.value_shape)) or 1
    if value_size == 1:
        linear_form = fem.form(test * ufl.dx)
    else:
        ones = fem.Constant(domain, np.ones(value_size, dtype=np.float64))
        linear_form = fem.form(ufl.inner(ones, test) * ufl.dx)
    result = fem.assemble_vector(linear_form)
    result.scatter_reverse(la.InsertMode.add)
    return result.array.copy()
```

Create `EnergyBase` with legacy constructor validation, `setup`, energy
assembly, functional differentiation, box-method `compute_field`,
`compute_energy`, and `average_field`. Preserve the legacy convention:

```python
self.E = self.E_integrand * ufl.dx
self.dE_dm = (-1.0 / MU0) * ufl.derivative(
    self.E_integrand / self.Ms.f * ufl.dx, self.m.f
)
```

Energy assembly multiplies by `unit_length ** mesh_dim`; gradient interactions
include their own inverse-length factors, as in the legacy modules.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2.

Expected: both nodal-volume tests pass.

- [ ] **Step 5: Commit the energy foundation**

```bash
git add dev/dolfinx/finmag/energies dev/dolfinx/finmag/tests/test_energies.py
git commit -m "feat: add DOLFINx energy base"
```

### Task 5: Port Exchange, Zeeman, and uniaxial anisotropy

**Files:**
- Create: `dev/dolfinx/finmag/energies/exchange.py`
- Create: `dev/dolfinx/finmag/energies/zeeman.py`
- Create: `dev/dolfinx/finmag/energies/anisotropy.py`
- Modify: `dev/dolfinx/finmag/energies/__init__.py`
- Modify: `dev/dolfinx/finmag/tests/test_energies.py`
- Reference only: `src/finmag/energies/exchange.py`
- Reference only: `src/finmag/energies/zeeman.py`
- Reference only: `src/finmag/energies/anisotropy.py`

- [ ] **Step 1: Add failing analytic interaction tests**

```python
from finmag.constants import MU0
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.field import Field


def _fields(domain, m_value, ms=2.0):
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    scalar_space = fem.functionspace(domain, ("DG", 0))
    return Field(vector_space, m_value, normalised=True), Field(scalar_space, ms)


def test_zeeman_field_and_energy_on_unit_square():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    m, ms = _fields(domain, (1.0, 0.0, 0.0), ms=3.0)
    zeeman = Zeeman((2.0, 0.0, 0.0))
    zeeman.setup(m, ms)
    assert np.allclose(zeeman.compute_field().reshape((-1, 3)), (2.0, 0.0, 0.0))
    assert zeeman.compute_energy() == pytest.approx(-6.0 * MU0)


def test_exchange_energy_for_linear_magnetisation():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    m, ms = _fields(domain, (1.0, 0.0, 0.0))
    m.set(lambda x: np.vstack((x[0], np.zeros_like(x[0]), np.zeros_like(x[0]))))
    exchange = Exchange(2.5)
    exchange.setup(m, ms)
    assert exchange.compute_energy() == pytest.approx(2.5)


def test_uniaxial_energy_parallel_and_perpendicular():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    parallel, ms = _fields(domain, (0.0, 0.0, 1.0))
    perpendicular, _ = _fields(domain, (1.0, 0.0, 0.0))
    interaction = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0))
    interaction.setup(parallel, ms)
    assert interaction.compute_energy() == pytest.approx(0.0)
    interaction = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0))
    interaction.setup(perpendicular, ms)
    assert interaction.compute_energy() == pytest.approx(4.0)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_energies.py
```

Expected: imports fail for the missing interaction classes.

- [ ] **Step 3: Implement the three legacy-shaped interactions**

Implement the original forms:

```python
# Exchange
self.A = Field(dg_space, self.A_value, name="A")
self.exchange_factor = 1.0 / unit_length**2
energy = self.exchange_factor * self.A.f * ufl.inner(
    ufl.grad(m.f), ufl.grad(m.f)
)
super().setup(energy, m, Ms, unit_length)
```

```python
# Zeeman
self.H = Field(vector_space, self.H_value, name="H_ext")
self.E_integrand = -MU0 * self.Ms.f * ufl.dot(self.m.f, self.H.f)
```

```python
# UniaxialAnisotropy
projection = ufl.dot(self.axis.f, m.f)
energy = self.K1.f * (1.0 - projection**2) - self.K2.f * projection**4
super().setup(energy, m, Ms, unit_length)
```

Retain the legacy names, constructor parameters, stored fields, and
`compute_energy`/`compute_field`/`average_field` methods. Delete deferred
constructor values only after successful setup, as the legacy classes do.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2.

Expected: all interaction and nodal-volume tests pass.

- [ ] **Step 5: Commit the common interactions**

```bash
git add dev/dolfinx/finmag/energies dev/dolfinx/finmag/tests/test_energies.py
git commit -m "feat: port common Finmag energies to DOLFINx"
```

### Task 6: Port the named `EffectiveField` registry

**Files:**
- Create: `dev/dolfinx/finmag/physics/__init__.py`
- Create: `dev/dolfinx/finmag/physics/effective_field.py`
- Create: `dev/dolfinx/finmag/tests/test_effective_field.py`
- Reference only: `src/finmag/physics/effective_field.py`

- [ ] **Step 1: Write failing interaction-registry tests**

```python
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.energies import Zeeman
from finmag.errors import UnknownInteraction
from finmag.field import Field
from finmag.physics.effective_field import EffectiveField


def _effective_field():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    scalar_space = fem.functionspace(domain, ("DG", 0))
    m = Field(vector_space, (1.0, 0.0, 0.0))
    ms = Field(scalar_space, 1.0)
    return EffectiveField(m, ms, 1.0)


def test_add_get_compute_and_remove_interactions():
    effective = _effective_field()
    effective.add(Zeeman((1.0, 0.0, 0.0), name="first"))
    effective.add(Zeeman((0.0, 2.0, 0.0), name="second"))
    assert effective.all() == ["first", "second"]
    assert np.allclose(
        effective.compute().reshape((-1, 3)), (1.0, 2.0, 0.0)
    )
    effective.remove("first")
    assert effective.all() == ["second"]


def test_duplicate_and_unknown_interactions_fail():
    effective = _effective_field()
    effective.add(Zeeman((1.0, 0.0, 0.0), name="field"))
    with pytest.raises(ValueError, match="unique"):
        effective.add(Zeeman((2.0, 0.0, 0.0), name="field"))
    with pytest.raises(UnknownInteraction):
        effective.get("missing")
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_effective_field.py
```

Expected: import fails for missing `finmag.physics.effective_field`.

- [ ] **Step 3: Port the legacy registry behavior**

Implement the legacy methods without prototype parameter coupling:

```python
class EffectiveField:
    def __init__(self, m, Ms, unit_length):
        self.m_field = m
        self.Ms = Ms
        self.unit_length = unit_length
        self.H_eff = np.zeros_like(m.as_array())
        self.interactions = {}
        self.need_time_update = []

    def add(self, interaction, with_time_update=None):
        if interaction.name in self.interactions:
            raise ValueError(
                "Interaction names must be unique: {!r}".format(interaction.name)
            )
        interaction.setup(self.m_field, self.Ms, self.unit_length)
        self.interactions[interaction.name] = interaction
        if with_time_update is not None:
            self.need_time_update.append(with_time_update)

    def update(self, t=None):
        if t is None and self.need_time_update:
            raise ValueError("Interactions require a simulation time update")
        for callback in self.need_time_update:
            callback(t)
        self.H_eff.fill(0.0)
        for interaction in self.interactions.values():
            self.H_eff += interaction.compute_field()
```

Add `compute`, `total_energy`, `exists`, `get`, `all`, and `remove` with the
legacy return shapes and errors.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2.

Expected: all registry tests pass.

- [ ] **Step 5: Commit the registry port**

```bash
git add dev/dolfinx/finmag/physics dev/dolfinx/finmag/tests/test_effective_field.py
git commit -m "feat: port Finmag effective-field registry"
```

### Task 7: Port physical LLG dynamics

**Files:**
- Create: `dev/dolfinx/finmag/physics/llg.py`
- Modify: `dev/dolfinx/finmag/physics/__init__.py`
- Create: `dev/dolfinx/finmag/tests/test_llg.py`
- Reference only: `src/finmag/physics/llg.py`

- [ ] **Step 1: Write failing physical-RHS tests**

```python
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.constants import GAMMA
from finmag.energies import Zeeman
from finmag.physics.llg import LLG


def _llg():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    scalar = fem.functionspace(domain, ("Lagrange", 1))
    vector = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    llg = LLG(scalar, vector)
    llg.Ms = 8.6e5
    llg.set_alpha(0.5)
    llg.set_m((1.0, 0.0, 0.0))
    llg.effective_field.add(Zeeman((0.0, 0.0, 1.0)))
    return llg


def test_llg_uses_physical_gamma_and_total_field():
    llg = _llg()
    derivative = llg.solve(0.0).reshape((-1, 3))
    scale = GAMMA / (1.0 + 0.5**2)
    assert np.allclose(derivative[:, 0], 0.0)
    assert np.allclose(derivative[:, 1], scale)
    assert np.allclose(derivative[:, 2], 0.5 * scale)


def test_set_m_normalises_each_node():
    llg = _llg()
    llg.set_m((2.0, 0.0, 0.0))
    values = llg.m_field.as_array().reshape((-1, 3))
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_llg.py
```

Expected: import fails for missing `LLG`.

- [ ] **Step 3: Implement the legacy-shaped LLG core**

Create scalar `alpha`, scalar `Ms`, vector `m`, and `EffectiveField` state.
Implement `set_alpha`, `set_m`, `solve_for`, and `solve` using:

```python
def _gilbert_rhs(m, field, alpha, gamma, do_precession):
    precession = np.cross(m, field)
    damping = np.cross(m, precession)
    if do_precession:
        return -gamma / (1.0 + alpha[:, None] ** 2) * (
            precession + alpha[:, None] * damping
        )
    return -gamma * alpha[:, None] / (
        1.0 + alpha[:, None] ** 2
    ) * damping
```

All arrays use DOLFINx node-interleaved layout. `solve(t)` must call
`self.effective_field.compute(t)` so every registered interaction drives the
trajectory.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2.

Expected: both LLG tests pass.

- [ ] **Step 5: Commit the LLG port**

```bash
git add dev/dolfinx/finmag/physics dev/dolfinx/finmag/tests/test_llg.py
git commit -m "feat: port physical LLG dynamics to DOLFINx"
```

### Task 8: Port the SciPy driver abstraction

**Files:**
- Create: `dev/dolfinx/finmag/drivers/__init__.py`
- Create: `dev/dolfinx/finmag/drivers/scipy_integrator.py`
- Create: `dev/dolfinx/finmag/drivers/llg_integrator.py`
- Create: `dev/dolfinx/finmag/tests/test_drivers.py`
- Reference only: `src/finmag/drivers/scipy_integrator.py`
- Reference only: `src/finmag/drivers/llg_integrator.py`

- [ ] **Step 1: Write failing adaptive-integration tests**

```python
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.drivers.llg_integrator import llg_integrator
from finmag.energies import Zeeman
from finmag.physics.llg import LLG


def _integrator():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    scalar = fem.functionspace(domain, ("Lagrange", 1))
    vector = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    llg = LLG(scalar, vector)
    llg.Ms = 8.6e5
    llg.set_m((1.0, 0.0, 0.0))
    llg.effective_field.add(Zeeman((0.0, 0.0, 1e5)))
    return llg, llg_integrator(llg, llg.m_field, backend="scipy")


def test_scipy_integrator_advances_physical_time_and_state():
    llg, integrator = _integrator()
    before = llg.m_field.as_array()
    integrator.advance_time(1e-12)
    assert integrator.cur_t == pytest.approx(1e-12)
    assert not np.allclose(llg.m_field.as_array(), before)


def test_scipy_integrator_rejects_backwards_time():
    _, integrator = _integrator()
    integrator.advance_time(1e-12)
    with pytest.raises(RuntimeError, match="past"):
        integrator.advance_time(0.5e-12)
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_drivers.py
```

Expected: import fails for missing drivers.

- [ ] **Step 3: Port the legacy SciPy wrapper**

Use `scipy.integrate.ode` with `vode`, BDF, and the legacy tolerances. Preserve
`cur_t`, `n_rhs_evals`, `advance_time`, and `reinit`. Add explicit backward-time
validation before calling SciPy. After an accepted step, write the state to the
shared `Field` and normalise it.

Implement backend dispatch:

```python
def llg_integrator(llg, m0, backend="scipy", **kwargs):
    if backend == "scipy":
        return ScipyIntegrator(llg, m0, **kwargs)
    if backend == "sundials":
        raise ImportError(
            "The native Sundials backend has not yet been ported to the "
            "staged DOLFINx package"
        )
    raise ValueError("backend must be either scipy or sundials")
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2.

Expected: both driver tests pass.

- [ ] **Step 5: Commit the driver port**

```bash
git add dev/dolfinx/finmag/drivers dev/dolfinx/finmag/tests/test_drivers.py
git commit -m "feat: port Finmag SciPy LLG driver"
```

### Task 9: Port `Simulation`, `sim_with`, and public exports

**Files:**
- Create: `dev/dolfinx/finmag/sim/__init__.py`
- Create: `dev/dolfinx/finmag/sim/sim.py`
- Modify: `dev/dolfinx/finmag/__init__.py`
- Create: `dev/dolfinx/finmag/tests/test_simulation.py`
- Reference only: `src/finmag/sim/sim.py`
- Reference only: `src/finmag/init.py`

- [ ] **Step 1: Write failing public-API tests**

```python
import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

import finmag
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman


def test_simulation_manages_interactions_and_advances():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    sim = finmag.Simulation(
        domain,
        Ms=8.6e5,
        unit_length=1e-9,
        integrator_backend="scipy",
    )
    sim.set_m((1.0, 0.0, 1.0))
    sim.add(Exchange(13e-12))
    sim.add(Zeeman((0.0, 0.0, 1e5)))

    assert sim.interactions() == ["Exchange", "Zeeman"]
    assert np.isfinite(sim.total_energy())
    sim.run_until(1e-12)
    assert sim.t == pytest.approx(1e-12)
    values = sim.m_field.as_array().reshape((-1, 3))
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)


def test_sim_with_preserves_supported_legacy_convenience_api():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    sim = finmag.sim_with(
        domain,
        Ms=8.6e5,
        m_init=(1.0, 0.0, 1.0),
        alpha=0.5,
        unit_length=1e-9,
        integrator_backend="scipy",
        A=13e-12,
        K1=2e5,
        K1_axis=(0.0, 0.0, 1.0),
        H_ext=(0.0, 0.0, 1e5),
        demag_solver=None,
    )
    assert sim.interactions() == ["Anisotropy", "Exchange", "Zeeman"]


def test_anisotropy_drives_the_simulation_without_an_applied_field():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    sim = finmag.Simulation(
        domain,
        Ms=8.6e5,
        unit_length=1e-9,
        integrator_backend="scipy",
    )
    sim.set_m((1.0, 0.0, 0.01))
    sim.add(UniaxialAnisotropy(2e5, (0.0, 0.0, 1.0)))
    before = sim.m.copy()
    sim.run_until(1e-12)
    assert not np.allclose(sim.m, before)


def test_demag_fails_explicitly():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    with pytest.raises(NotImplementedError, match="demag"):
        finmag.sim_with(
            domain,
            Ms=1.0,
            m_init=(1.0, 0.0, 0.0),
            demag_solver="FK",
        )
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pixi run -e dolfinx env PYTHONPATH=dev/dolfinx \
  python -m pytest -q dev/dolfinx/finmag/tests/test_simulation.py
```

Expected: `finmag.Simulation` and `sim_with` are absent.

- [ ] **Step 3: Implement the first legacy-shaped `Simulation` slice**

Preserve the legacy constructor shape and core delegation:

```python
class Simulation:
    def __init__(
        self,
        mesh,
        Ms,
        unit_length=1,
        name="unnamed",
        kernel="llg",
        integrator_backend="scipy",
        pbc=None,
        average=False,
        parallel=False,
    ):
        if mesh.comm.size != 1:
            raise NotImplementedError(
                "DOLFINx Finmag time integration currently supports one rank"
            )
        if kernel != "llg":
            raise NotImplementedError("Only the llg kernel is currently ported")
        if pbc is not None or parallel:
            raise NotImplementedError("PBC and parallel stepping are not ported")
        self.name = name
        self.mesh = mesh
        self.unit_length = float(unit_length)
        self.integrator_backend = integrator_backend
        self.S1 = fem.functionspace(mesh, ("Lagrange", 1))
        self.S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
        self.llg = LLG(self.S1, self.S3, unit_length=unit_length)
        self.Ms = Ms
        self._integrator = None
        self.reltol = 1e-6
        self.abstol = 1e-6
```

Implement the legacy core properties and methods by delegating to `LLG`,
`EffectiveField`, and the driver. `set_m` must reinitialise an existing
integrator. `run_until` delegates to `advance_time` in this slice because the
scheduler is deferred.

Implement `sim_with` with legacy argument names. Add Exchange, anisotropy,
Zeeman, and DMI only when supported. Any non-`None` demag request raises
`NotImplementedError` rather than silently dropping the interaction.

Export `Simulation`, `sim_with`, `Field`, and the three interaction classes from
the staged `finmag` package.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2.

Expected: all simulation tests pass.

- [ ] **Step 5: Run the complete staged-package gate**

```bash
pixi run -e dolfinx dolfinx-finmag-import
pixi run -e dolfinx dolfinx-finmag-pytest
```

Expected: import resolves under `dev/dolfinx/finmag` and all staged-package
tests pass.

- [ ] **Step 6: Commit the simulation API**

```bash
git add dev/dolfinx/finmag
git commit -m "feat: add DOLFINx Finmag Simulation core"
```

### Task 10: Add the executable core smoke and final verification

**Files:**
- Create: `dev/dolfinx/finmag/tests/smoke.py`
- Modify: `dev/dolfinx/finmag/tests/test_simulation.py`
- Modify: `dev/bin/verify-dolfinx-m4`
- Modify: `dev/dolfinx/README.md`
- Modify: `dev/dolfinx/porting_map.md`
- Modify: `plan.org`
- Modify: `transition-notes.org`
- Modify: `agents.md`

- [ ] **Step 1: Write the smoke module**

```python
import json

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from finmag import Simulation
from finmag.energies import Exchange, Zeeman


def main():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    sim = Simulation(
        domain,
        Ms=8.6e5,
        unit_length=1e-9,
        integrator_backend="scipy",
    )
    sim.set_m((1.0, 0.0, 1.0))
    sim.add(Exchange(13e-12))
    sim.add(Zeeman((0.0, 0.0, 1e5)))
    initial_energy = sim.total_energy()
    sim.run_until(1e-12)
    values = sim.m_field.as_array().reshape((-1, 3))
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)
    print(
        json.dumps(
            {
                "backend": "dolfinx",
                "interactions": sim.interactions(),
                "initial_energy": initial_energy,
                "final_energy": sim.total_energy(),
                "time": sim.t,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke and verify it exercises the staged package**

```bash
pixi run -e dolfinx dolfinx-finmag-smoke
```

Expected: JSON reports backend `dolfinx`, interactions `Exchange` and `Zeeman`,
finite energies, and time `1e-12`.

- [ ] **Step 3: Run both the untouched legacy and staged DOLFINx witnesses**

```bash
pixi run import-finmag
pixi run -e dolfinx dolfinx-finmag-import
PIXI_HOME=/tmp/pixi-cache XDG_CACHE_HOME=/tmp/pixi-cache \
  PIXI_CACHE_DIR=/tmp/pixi-cache dev/bin/verify-dolfinx-m4
```

Expected:

- the default environment imports legacy `src/finmag`;
- the DOLFINx environment imports `dev/dolfinx/finmag`;
- prototype and staged-package DOLFINx gates pass.

- [ ] **Step 4: Update status documentation with measured results**

Record the exact DOLFINx version, test count, smoke output contract, supported
API, explicit non-scope, and the fact that `src/finmag` remained unchanged.
Do not claim demag, restart, scheduler, Sundials, or MPI support.

- [ ] **Step 5: Run final verification**

```bash
git diff --check
git status --short
pixi run -e dolfinx dolfinx-finmag-pytest
pixi run -e dolfinx dolfinx-pytest
pixi run -e dolfinx dolfinx-finmag-smoke
```

Expected: no whitespace errors, only intentional files changed, both test suites
pass, and the core smoke succeeds without warnings or NaNs.

- [ ] **Step 6: Commit the verified core slice**

```bash
git add dev/dolfinx/finmag dev/bin/verify-dolfinx-m4 \
  dev/dolfinx/README.md dev/dolfinx/porting_map.md \
  plan.org transition-notes.org agents.md
git commit -m "test: verify staged DOLFINx Finmag core"
```
