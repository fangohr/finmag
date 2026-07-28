<img src="dev/logos/finmag_logo.png" width="300" align="right">

# FinMag: finite-element micromagnetic simulation tool
Marc-Antonio Bisotti<sup>1</sup>, Marijan Beg<sup>1,2</sup>, Weiwei Wang<sup>1</sup>, Maximilian Albert<sup>1</sup>, Dmitri Chernyshenko<sup>1</sup>, David Cortés-Ortuño<sup>1</sup>, Ryan A. Pepper<sup>1</sup>, Mark Vousden<sup>1</sup>, Rebecca Carey<sup>1</sup>, Hagen Fuchs<sup>3</sup>, Anders Johansen<sup>1</sup>, Gabriel Balaban<sup>1</sup>, Leoni Breth<sup>1</sup>, Thomas Kluyver<sup>1,2</sup>, and Hans Fangohr<sup>1,2,4</sup>

<sup>1</sup> *Faculty of Engineering and the Environment, University of Southampton, Southampton SO17 1BJ, United Kingdom*  
<sup>2</sup> *European XFEL GmbH, Holzkoppel 4, 22869 Schenefeld, Germany*  
<sup>3</sup> *Helmholtz-Zentrum Dresden-Rossendorf, Bautzner Landstraße 400, 01328 Dresden, Germany*  
<sup>4</sup> *Max Planck Institute for the Structure and Dynamics of Matter, Luruper Chaussee 149, 22761 Hamburg, Germany*  

| Description | Badge |
| --- | --- |
| Tests | [![workflow](https://github.com/fangohr/finmag/workflows/workflow/badge.svg)](https://github.com/fangohr/finmag/actions) |
|       | [![docker-image](https://github.com/fangohr/finmag/workflows/docker-image/badge.svg)](https://github.com/fangohr/finmag/actions) |
| Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fangohr/finmag/HEAD?filepath=binder%2Findex.ipynb) |
| License | [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) |
| DockerHub | [![DockerHub](https://img.shields.io/badge/DockerHub-finmag-blue.svg)](https://hub.docker.com/u/finmag/) |
| DOI | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1216011.svg)](https://doi.org/10.5281/zenodo.1216011) |


## About

- Finmag was intended to be a thin (and mostly) Python layer on top of [FEniCS](https://fenicsproject.org/) to enable Python-scripted multi-physics micromagnetic simulations. Accordingly, the name FINmag originates from the dolFIN interface to FEniCS. Some compiled code moved into the project.

- The code has been developed from 2011 to 2018 by [Hans Fangohr](http://fangohr.github.io)'s group at the University of Southampton (UK) and European XFEL GmbH (Germany).

- The GitHub page of the project with the most recent version is https://github.com/fangohr/finmag.

- This is a working prototype which is not polished, with some (in large parts outdated) attempts at documentation. There is also some outdated code in the repository.

- We do not consider the codebase, documentation, and other content of sufficient quality to encourage uptake in the community. (Experts are welcome!) This is primarily a resource problem.

- Does not execute efficiently in parallel (time integration is serial).

- There is no support available.

- Contributions and pull requests to both the code and documentation are welcome, but no promise can be made that these will be reviewed and/or integrated.

- The code has been used for a number of scientific studies and publications (see the [Publications](#Publications) section).

- The repository may well be of historical value and probably captures some of the typical research software engineering challenges. (We should write up a summary of our gathered experiences.)

- There has not been dedicated funding to support the software development.

## Installation / Using the tool via Docker

There is a dedicated organisation on [DockerHub](https://hub.docker.com/) named [`finmag`](https://hub.docker.com/u/finmag/). We provide pre-built images in the [`finmag/finmag`](https://hub.docker.com/r/finmag/finmag/) repository. More information about Docker, as well as on how to install it on your system, can be found [here](https://www.docker.com/).

### Getting the image

The easiest way to get the most recent image is by pulling it from the DockerHub [`finmag/finmag`](https://hub.docker.com/r/finmag/finmag/) repository

    $ docker pull finmag/finmag:latest

Alternatively, you can navigate to `install/docker/latest` and run `make pull`. You can also build it on your own machine by navigating to `install/docker/latest`, and running

    $ make build

### Testing

After you pulled/built the `finmag/finmag` image, you can test it with

    $ docker run -ti -w="/finmag" --rm finmag/finmag bash -c "py.test -v"

or by running `make test` in `install/docker/latest` directory.

### Running the container

To run your Finmag code inside Docker, please navigate to the directory where your `my-finmag-script.py` file is (`cd path/to/your/file`) and run

    $ docker run -ti -v $(pwd):/io --rm finmag/finmag bash -c "python my-finmag-script.py"

If you want to run code interactively inside the container, then you can start with

    $ docker run -ti -v $(pwd):/io --rm finmag/finmag

### Finmag dependencies container

Docker image which contains all of the dependencies necessary to run finmag is hosted on DockerHub as `finmag/finmag:dependencies`. Similar to previous sections, if you navigate to `install/docker/dependencies`, you can run `make pull`, `make run`, etc.

### Installing on host

More detailed comments on the installation of finmag on a host machine are in [`install/README.md`](install/README.md).

### Installing the DOLFINx port (pixi)

The DOLFINx port (`src/finmag` on Python 3.12 + `fenics-dolfinx`) is installed
as a regular editable Python package via [pixi](https://pixi.sh) and a
`pyproject.toml` (setuptools, src-layout). This is currently the only supported
install path for this lane: there is no published wheel, and `PYTHONPATH=src`
(the earlier transitional workaround, see `transition-notes.org`) is kept only
as a single fallback task.

**Read [`docs/SUPPORTED.md`](docs/SUPPORTED.md) before using this lane.** It is
the user-facing statement of exactly what SR1 supports, how each family is
validated and to what tolerance, what is still waiting to be ported, what was
dropped, and how an unavailable surface fails.

```bash
# 1. Create/sync the dolfinx pixi environment (conda-forge fenics-dolfinx,
#    mpi4py, petsc4py, sundials, the native build toolchain, ...).
pixi install -e dolfinx

# 2. Install finmag itself into that environment as an editable package.
#    --no-deps: numpy/scipy/the FEM stack/SUNDIALS come from the conda/pixi
#    environment above, not from PyPI (see pyproject.toml's dependency
#    comment for why: dolfinx/mpi4py/petsc4py/sundials are not reliable pip
#    installs for this project and must stay tied to the conda build they are
#    linked against).
pixi run -e dolfinx dolfinx-install-editable

# 3. Build the native extensions (bem_arrays.so, sundials.so and
#    treecode_bem) via the
#    existing native/Makefile. Optional up front -- `finmag.native` triggers
#    the same `make` as an import side effect -- but recommended so a broken
#    native toolchain fails fast and visibly instead of inside the first test
#    that imports finmag.native.
pixi run -e dolfinx dolfinx-native-build

# 4. Confirm `import finmag` resolves to this checkout (not some other
#    installed copy).
pixi run -e dolfinx dolfinx-provenance-check
```

#### Verify the install

One command runs the whole supported verification (it performs the editable
install, native build and provenance check itself, then the 33 focused gates —
so do not run steps 1-4 above redundantly unless you are diagnosing an
install/build failure):

```bash
dev/bin/verify-dolfinx-m5      # expect: 33/33 green
```

For the non-gating parity backlog (the whole `src/finmag` tree, ~30 min, **not**
a verdict — see `docs/SUPPORTED.md` §7):

```bash
dev/bin/inventory-dolfinx-suite
```

#### Quickstart

```python
import finmag

sim = finmag.example.barmini()      # 3x3x10 nm Py bar, m0 = (1, 0, 1)
sim.run_until(1e-10)                # deterministic time integration
print(sim.m_average)                # volume-averaged magnetisation
print(sim.total_energy())           # total energy, J
sim.relax()                         # relax to equilibrium
```

Build your own simulation with `finmag.sim_with(mesh, Ms, m_init, A=...,
K1=..., K1_axis=..., H_ext=..., D=..., alpha=..., unit_length=...)`, where
`mesh` is a `dolfinx.mesh.Mesh`. See
[`docs/SUPPORTED.md`](docs/SUPPORTED.md) for the full supported API,
[`examples/`](examples) for seventeen converted examples, and
`pixi run -e dolfinx dolfinx-src-examples-pytest` to run the fast lane of them.

`pyproject.toml`'s static `version = "0.1.0"` is packaging metadata only; it
has no runtime meaning. Runtime provenance remains `finmag.__version__` (a
git revision SHA written into `src/finmag/__version__.py` by
`native/Makefile`'s `add_version` target, gated by `WRITE_FINMAG_VERSION`).
The two are intentionally decoupled: a git SHA is not a valid PEP 440
version string, so `pyproject.toml` cannot derive its `version` from
`finmag.__version__` directly.

There is deliberately no build-backend hook that invokes `native/Makefile`
during `pip install`: the native modules are C++/Boost.Python extensions
linked directly against this pixi environment's conda-provided compiler,
Boost, and SUNDIALS libraries, and are rebuilt in place under
`src/finmag/native/`, which an editable install picks up automatically with
no reinstall step. A non-editable (built) wheel is explicitly a non-goal for
this lane: the compiled `.so` files are linked against this specific conda
environment's libraries (Boost.Python ABI tag, SUNDIALS 7 sonames, etc.) and
are not portable/redistributable the way a wheel implies.

Every focused DOLFINx port gate (`pixi run -e dolfinx dolfinx-src-*`) and the
aggregated `dev/bin/verify-dolfinx-m5` witness run against this installed
package. The immutable legacy-oracle comparison lane
(`dev/bin/run-legacy-oracle`) is unaffected: its reference checkouts predate
`pyproject.toml` and do not need it.

### Continuous integration

- `.github/workflows/dolfinx-m5.yml` — runs on every push/PR: the fast,
  focused port-gate witness (~33 gates).
- `.github/workflows/test-python.yml` — weekly (Mondays) + on-demand: the
  full `src/finmag` suite inventory, gated on `failed=0 errors=0`.
- `.github/workflows/test-slow.yml` — on-demand only (no schedule): the
  heavy `FINMAG_EXAMPLE_FULL=1` example lane; long-running by design.

GitHub only fires `schedule`/`workflow_dispatch` from the repo's default
branch, so `test-python.yml`/`test-slow.yml` are inert on non-default
branches until merged; run their equivalents locally meanwhile:
`dev/bin/inventory-dolfinx-suite` and `FINMAG_EXAMPLE_FULL=1 pixi run -e
dolfinx dolfinx-src-examples-pytest`. Also note GitHub-hosted runners cap
jobs at 360 minutes, below the FULL lane's current ~10-13h runtime (register
P1); `test-slow.yml`'s `timeout-minutes: 360` documents the tier rather than
guaranteeing completion until the post-SR1 performance work lands.

### Current DOLFINx status and limitations

The DOLFINx branch is at **SR1** (tag `sr1`, declared 2026-07-28): a serial
deterministic micromagnetic simulator, **not** yet full parity with original
`master`. It provides the common energy terms, FK and treecode/MacroGeometry
demag, varying materials, regions, local spin-transfer torque, SciPy and native
CVODE integration, scheduling, restart and common output. The aggregate test
gate being green means this **ported subset** passes; it does not mean every
original Finmag feature is available — and the full-resolution example lane
stands at **15 of 17 entries green**, with `std_prob_4` and `magnetic_grain`
deferred by owner decision to a re-run after the performance work below.

Major work still outstanding includes thermal SLLG/LLB, normal modes and
FFT/PSD, legacy NEB, general MPI time stepping, `Simulation(pbc=)`, nonlocal
STT, live external comparison harnesses, VTK/XDMF function readback, and
specialist plotting/utilities. Some top-level compatibility names still expose
legacy `dolfin` imports. Native Sundials is the public `Simulation`/`sim_with`
default; SciPy remains a fully supported explicit opt-in
(`integrator_backend="scipy"`).

**Performance caveat:** the port is currently ~17.6x slower than legacy-era
expectations on the heaviest examples (register `P1`); the root cause is
per-evaluation form re-assembly and a fix is planned in
[`docs/superpowers/plans/2026-07-28-post-sr1-performance.md`](docs/superpowers/plans/2026-07-28-post-sr1-performance.md).
Budget accordingly.

**Start here:** [`docs/SUPPORTED.md`](docs/SUPPORTED.md) — supported API,
validation classes and tolerances, the "waiting to be ported" list, the
permanently-dropped list, and how deferred surfaces fail.

For the exact tested boundary and known failures, see
[`docs/superpowers/capability-status.md`](docs/superpowers/capability-status.md).
For every recorded behavior deviation and its owner disposition, see
[`docs/superpowers/acceptance-register.md`](docs/superpowers/acceptance-register.md).
For the file-by-file master/pixi test and example mapping, see
[`docs/superpowers/master-pixi-parity-manifest.md`](docs/superpowers/master-pixi-parity-manifest.md).
For a printable functionality-by-functionality owner discussion, use
[`docs/superpowers/owner-porting-checklist.md`](docs/superpowers/owner-porting-checklist.md).
Unavailable public surfaces should raise a feature-specific error; a raw
`dolfin` import failure is a tracked port bug, not an installation instruction.

## Binder

If you want to try using Finmag in the cloud you can do it on [Binder](https://mybinder.org/v2/gh/fangohr/finmag/HEAD?filepath=binder%2Findex.ipynb). This does not require you to have anything installed and no files will be created on your machine. You only need a web browser.

## Documentation

The documentation in the form of [Jupyter](http://jupyter.org/) notebooks is available in [`doc/ipython_notebooks_src`](doc/ipython_notebooks_src) directory. Large parts of documentation are currently outdated.

## How to cite

If you use Finmag in your research, please cite it as

- Marc-Antonio Bisotti, Marijan Beg, Weiwei Wang, Maximilian Albert, Dmitri Chernyshenko, David Cortés-Ortuño, Ryan A. Pepper, Mark Vousden, Rebecca Carey, Hagen Fuchs, Anders Johansen, Gabriel Balaban, Leoni Breth, Thomas Kluyver, and Hans Fangohr. FinMag: finite-element micromagnetic simulation tool (Version 0.1). Zenodo. DOI: http://doi.org/10.5281/zenodo.1216011

## License

Finmag is licensed under the BSD 3-Clause "New" or "Revised" License. For details, please refer to the [LICENSE](LICENSE) file. However, portions of the source code (e.g. src/util/numpy.h) are subject to the Boost Software License.

## Support

We do not provide support for Finmag. However, you are welcome to raise an issue in the GitHub [fangohr/finmag](https://github.com/fangohr/finmag) repository, but no promise can be made that the issue will be addressed.

## Publications

Finmag was used to run micromagnetic simulations in the following publications (in reversed chronological order):

- M. Beg, R. A. Pepper, D. Cortés-Ortuño, B. Atie, M. A. Bisotti, G. Downing, T. Kluyver, O. Hovorka, H. Fangohr. Stable and manipulable Bloch point. [Scientific Reports 9, 7959](https://doi.org/10.1038/s41598-019-44462-2) (2019). (arXiv:1808.10772)

- R. A. Pepper, M. Beg, D. Cortés-Ortuño, T. Kluyver, M.-A. Bisotti, R. Carey, M. Vousden, M. Albert, W. Wang, O. Hovorka, and H. Fangohr. Skyrmion states in thin confined polygonal nanostructures. [Journal of Applied Physics 9, 093903](http://aip.scitation.org/doi/10.1063/1.5022567) (2018). (arXiv:1801.03275)

- D. Cortés-Ortuño, W. Wang, M. Beg, R. A. Pepper, M.-A. Bisotti, R. Carey, M. Vousden, T. Kluyver, O. Hovorka, and H. Fangohr. Thermal stability and topological protection of skyrmions in nanotracks. [Scientific Reports 7, 4061](http://www.nature.com/articles/s41598-017-03391-8) (2017). (arXiv:1611.07079)

- M. Beg, M. Albert, M.-A. Bisotti, D. Cortés-Ortuño, W. Wang, R. Carey, M. Vousden, O. Hovorka, C. Ciccarelli, C. S. Spencer, C. H. Marrows, and H. Fangohr. Dynamics of skyrmionic states in confined helimagnetic nanostructures. [Physical Review B 95, 014433](http://link.aps.org/doi/10.1103/PhysRevB.95.014433) (2017). (arXiv:1604.08347)

- A. Baker, M. Beg, G. Ashton, M. Albert, D. Chernyshenko, W. Wang, S. Zhang, M.-A. Bisotti, M. Franchin, C. Lian Hu, R. L. Stamps, T. Hesjedal, and H. Fangohr. Proposal of a micromagnetic standard problem for ferromagnetic resonance simulations. [Journal of Magnetism and Magnetic Materials 421, 428-439](http://linkinghub.elsevier.com/retrieve/pii/S0304885316307545) (2017). (arXiv:1603.05419)

- P. J. Metaxas, M. Albert, S. Lequeux, V. Cros, J. Grollier, P. Bortolotti, A. Anane, and H. Fangohr. Resonant translational, breathing, and twisting modes of transverse magnetic domain walls pinned at notches. [Phys. Rev. B 93, 054414](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.054414) (2016).

- J. P. Fried, H. Fangohr, M. Kostylev, and P. J. Metaxas. Exchange-mediated, nonlinear, out-of-plane magnetic field dependence of the ferromagnetic vortex gyrotropic mode frequency driven by core deformation. [Phys. Rev. B 94, 224407](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.224407) (2016).

- R. Carey, M. Beg, M. Albert, M.-A. Bisotti, D. Cortés-Ortuño, M. Vousden, W. Wang, O. Hovorka, and H. Fangohr. Hysteresis of nanocylinders with Dzyaloshinskii-Moriya interaction. [Applied Physics Letters 109, 122401](http://scitation.aip.org/content/aip/journal/apl/109/12/10.1063/1.4962726) (2016). (arXiv:1606.05181)

- M. Sushruth, J. Ding, J. Duczynski, R. C. Woodward, R. A. Begley, H. Fangohr, R. O. Fuller, A. O. Adeyeye, M. Kostylev, and P. J. Metaxas. Resonance-Based Detection of Magnetic Nanoparticles and Microbeads Using Nanopatterned Ferromagnets. [Phys. Rev. Applied 6, 044005](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.6.044005) (2016).

- M. Albert, M. Beg, D. Chernyshenko, M.-A. Bisotti, R. L. Carey, H. Fangohr, and P. J. Metaxas. Frequency-based nanoparticle sensing over large field ranges using the ferromagnetic resonances of a magnetic nanodisc. [Nanotechnology 27, 455502](http://stacks.iop.org/0957-4484/27/i=45/a=455502?key=crossref.2ac6ca2e40700c0c20b17814ae4f6a9d) (2016). (arXiv:1604.07277)

- M. Vousden, M. Albert, M. Beg, M.-A. Bisotti, R. Carey, D. Chernyshenko, D. Cortés-Ortuño, W. Wang, O. Hovorka, C. H. Marrows, and H. Fangohr. Skyrmions in thin films with easy-plane magnetocrystalline anisotropy. [Applied Physics Letters 108, 132406](http://aip.scitation.org/doi/10.1063/1.4945262) (2016). (arXiv:1602.02064)

- J. P. Fried and P. J. Metaxas. Localized magnetic fields enhance the field sensitivity of the gyrotropic resonance frequency of a magnetic vortex. [Phys. Rev. B 93, 064422](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.064422) (2016).

- M. Beg, R. Carey, W. Wang, D. Cortés-Ortuño, M. Vousden, M.-A. Bisotti, M. Albert, D. Chernyshenko, O. Hovorka, R. L. Stamps, and H. Fangohr. Ground state search, hysteretic behaviour, and reversal mechanism of skyrmionic textures in confined helimagnetic nanostructures. [Scientific Reports 5, 17137](http://www.nature.com/articles/srep17137) (2015). (arXiv:1312.7665)

- P. J. Metaxas, M. Sushruth, R. A. Begley,   J. Ding, R. C. Woodward, I. S. Maksymov, M. Albert, W. Wang, H. Fangohr, A. O. Adeyeye, and M. Kostylev. Sensing magnetic nanoparticles using nano-confined ferromagnetic resonances in a magnonic crystal. [Appl. Phys. Lett. 106, 232406](https://aip.scitation.org/doi/abs/10.1063/1.4922392) (2015).

- W. Wang, M. Albert, M. Beg, M.-A. Bisotti, D. Chernyshenko, D. Cortés-Ortuño, I. Hawke, and H. Fangohr. Magnon driven domain wall motion with Dzyaloshinskii-Moriya interaction. [Physical Review Letters 114, 087203](http://link.aps.org/doi/10.1103/PhysRevLett.114.087203) (2015). (arXiv:1406.5997)

- W. Wang, M. Beg, B. Zhang, W. Kuch, and H. Fangohr. Driving magnetic skyrmions with microwave fields. [Physical Review B (Rapid Communications) 92, 020403](http://link.aps.org/doi/10.1103/PhysRevB.92.020403) (2015). (arXiv:1505.00445)

- W. Wang, M. Dvornik, M.-A. Bisotti, D. Chernyshenko, M. Beg, M. Albert, A. Vansteenkiste, B. V. Waeyenberge, A. N. Kuchko, V. V. Kruglyak, and H. Fangohr. Phenomenological description of the nonlocal magnetization relaxation in magnonics, spintronics, and domain-wall dynamics. [Physical Review B 92, 054430](http://link.aps.org/doi/10.1103/PhysRevB.92.054430) (2015). (arXiv:1508.01478)

- B. Zhang, W. Wang, M. Beg, H. Fangohr, and W. Kuch. Microwave-induced dynamic switching of magnetic skyrmion cores in nanodots. [Applied Physics Letters 106, 102401](http://scitation.aip.org/content/aip/journal/apl/106/10/10.1063/1.4914496) (2015). (arXiv:1503.02869)

## Acknowledgements

- EPSRC’s [Doctoral Training Centre in Complex System Simulation](http://www.icss.soton.ac.uk) (EP/G03690X/1),

- EPSRC's [Centre for Doctoral Training in Next Generation Computational Modelling](http://ngcm.soton.ac.uk) (#EP/L015382/1),

- Horizon 2020 European Research Infrastructure project [OpenDreamKit](http://opendreamkit.org/) (676541),

- EPSRC's [Programme grant on Skyrmionics](https://www.skyrmions.ac.uk/) (EP/N032128/1),

- The [Gordon and Betty Moore Foundation](https://www.moore.org/) through Grant GBMF #4856, by the Alfred P. Sloan Foundation and by the Helmsley Trust.

## See also

- [Fidimag](https://github.com/computationalmodelling/fidimag): finite-difference micromagnetic simulation tool
