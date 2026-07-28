# Python 3 DOLFIN 2017 Snapshot Image

This directory holds an experimental Docker path for getting a Python 3
runtime for the old DOLFIN generation used by Finmag.

## Why this exists

The current reference image `finmag/finmag:latest` contains:

- Python 2.7 with `dolfin 2017.1.0`
- Python 3.5 without `dolfin`

That is enough for syntax-porting work, but not enough to run Finmag under
Python 3.

## What was established

The following facts were verified during local probing:

- the installed `python-dolfin` package in the Finmag image is
  `2017.1.0-1~ppa3~xenial1`;
- official DOLFIN `2017.1.0` release notes mention:
  switching the default Python version to Python 3 and using
  `-DDOLFIN_USE_PYTHON3=off` to force Python 2;
- Debian snapshot `20170920T000000Z sid` contains:
  - `python3-dolfin=2017.1.0-4`
  - `python3-ffc=2017.1.0-2`
  - `python3-ufl=2017.1.0-2`
  - `python3-dijitso=2017.1.0-2`
  - `python3-fiat=2017.1.0-2`
  - `python3-instant=2017.1.0-2`
  - matching `python3-petsc4py` and `python3-slepc4py`

This is the first concrete package set found that can plausibly provide an
old-style Python 3 DOLFIN runtime without building the entire stack from
scratch.

## Build

```bash
docker build \
  -f install/docker/python3-dolfin2017-snapshot/Dockerfile \
  -t finmag-py3-dolfin2017 .
```

## Intended milestone

The image should provide:

- `python3`
- `import dolfin`
- `dolfin.__version__ == "2017.1.0"`

Once that is working, the next step is to copy the Finmag checkout into the
image and start running:

- `python3 -m py_compile ...`
- `python3 -c "import finmag"`

before attempting any broader acceptance-suite runs.

## Caveats

- This is an experimental path.
- It intentionally uses an old Debian snapshot and unauthenticated snapshot
  packages.
- It is separate from the current Ubuntu/xenial based Finmag images.
- Finmag's own native extensions are still Python-2-biased today, so getting
  `python3-dolfin` is necessary but not sufficient for a working Python 3
  Finmag runtime.

## If the build fails

The most likely causes are:

- snapshot package dependency drift,
- missing runtime packages after `dist-upgrade`,
- additional exact-version dependencies not yet pinned.

In that case:

1. inspect the failing `apt-get install` line,
2. record the missing package/version,
3. add the exact pin to the Dockerfile if it also comes from the same
   `20170920T000000Z` snapshot.
