# Installing and running the DOLFINx port

This is the install guide for `src/finmag`, the Python 3.12 / `fenics-dolfinx`
port on branch `dolfinx-parity`. Linux only.

**Read [`SUPPORTED.md`](SUPPORTED.md) before using this lane.** It states what
SR1 supports, how each family is validated and to what tolerance, what is
still waiting to be ported, what was dropped, and how an unavailable surface
fails.

## Install (pixi, Linux)

The only supported install path is a regular **editable** install via
[pixi](https://pixi.sh) and `pyproject.toml` (setuptools, src-layout) — there
is no published wheel (see [Packaging notes](#packaging-notes)).

1. Create/sync the `dolfinx` pixi environment (conda-forge `fenics-dolfinx`,
   `mpi4py`, `petsc4py`, `sundials`, the native build toolchain, …):

   ```sh
   pixi install -e dolfinx
   ```

2. Install `finmag` into that environment as an editable package:

   ```sh
   pixi run -e dolfinx dolfinx-install-editable
   ```

   `--no-deps`: numpy/scipy/the FEM stack/SUNDIALS come from the conda/pixi
   environment, not PyPI — dolfinx/mpi4py/petsc4py/sundials are not reliable
   pip installs for this project and must stay tied to the conda build they
   are linked against.

3. Build the native extensions (`bem_arrays.so`, `sundials.so`,
   `treecode_bem`) via `native/Makefile`:

   ```sh
   pixi run -e dolfinx dolfinx-native-build
   ```

   Optional up front — `finmag.native` triggers the same `make` as an import
   side effect — but recommended, so a broken native toolchain fails fast and
   visibly instead of inside the first test that imports `finmag.native`.

4. Confirm `import finmag` resolves to this checkout, not some other
   installed copy:

   ```sh
   pixi run -e dolfinx dolfinx-provenance-check
   ```

A `PYTHONPATH=src` fallback (the earlier transitional workaround, see
`docs/archive/transition-notes.org`) is kept as a single pixi task,
`dolfinx-src-import-pythonpath-fallback`, exercised by the verifier below; it
is not a supported alternative to steps 1–4.

## Verify the install

```sh
dev/bin/verify-dolfinx-m5
```

Expect **33/33 green**. This performs the editable install, native build and
provenance check itself, then runs the 33 focused port gates — do not re-run
steps 1–4 redundantly unless you are diagnosing an install/build failure. See
`testing.md` (arrives with the restructure) for what each gate means and the
other CI lanes.

## Quickstart

```python
import finmag

sim = finmag.example.barmini()   # 3x3x10 nm Py bar, m0 = (1, 0, 1)
sim.run_until(1e-10)             # deterministic time integration
print(sim.m_average)             # volume-averaged magnetisation
print(sim.total_energy())        # total energy, J
sim.relax()                      # relax to equilibrium
```

Build your own simulation with

```python
finmag.sim_with(mesh, Ms, m_init, A=..., K1=..., K1_axis=..., H_ext=...,
                 D=..., alpha=..., unit_length=...)
```

where `mesh` is a `dolfinx.mesh.Mesh`. `examples/` holds seventeen converted
examples; `pixi run -e dolfinx dolfinx-src-examples-pytest` runs the fast lane
of them (see `testing.md` for the heavy `FINMAG_EXAMPLE_FULL=1` lane).

## Packaging notes

- `pyproject.toml`'s static `version = "0.1.0"` is packaging metadata only
  and has no runtime meaning. Runtime provenance remains
  `finmag.__version__` — a git revision SHA written into
  `src/finmag/__version__.py` by `native/Makefile`'s `add_version` target,
  gated by `WRITE_FINMAG_VERSION`. The two are intentionally decoupled: a git
  SHA is not a valid PEP 440 version string, so `pyproject.toml` cannot
  derive its `version` from `finmag.__version__`.
- There is deliberately **no build-backend hook** invoking
  `native/Makefile` during `pip install`. The native modules are C++/
  Boost.Python extensions linked directly against this pixi environment's
  conda-provided compiler, Boost and SUNDIALS libraries, and are rebuilt in
  place under `src/finmag/native/`, which an editable install picks up
  automatically with no reinstall step.
- A non-editable (built) **wheel is explicitly a non-goal** for this lane —
  the compiled `.so` files are linked against this specific conda
  environment's libraries (Boost.Python ABI tag, SUNDIALS 7 sonames, etc.)
  and are not portable/redistributable the way a wheel implies.

## Next steps

- [`SUPPORTED.md`](SUPPORTED.md) — what is supported, to what tolerance.
- `testing.md` — CI tiers, the inventory lane, running everything locally
  (arrives with the restructure).
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — pixi environment, test lanes,
  how to add a test/feature.
