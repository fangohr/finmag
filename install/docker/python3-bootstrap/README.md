# Python 3 Bootstrap Environment

This directory is for the first phase of the Python 3 migration.

## Scope

It is intentionally a bootstrap environment, not a complete Python 3 Finmag runtime.

The current legacy FEniCS packaging used by Finmag provides:

- `python-dolfin`

but not:

- `python3-dolfin`

for the `dolfin 2017.1.0` stack used by the project.

That means we can use Python 3 in a container for:

- syntax checks,
- package/import cleanup,
- compatibility edits,
- limited smoke checks that do not import `dolfin`,

but not yet for:

- full `import finmag`,
- running the acceptance suite under Python 3,
- validating native module loading through a Python 3 `dolfin` runtime.

## Current Findings

From `finmag/finmag:latest`:

- `python` is Python 2.7.12
- `python3` is Python 3.5.x
- `python` can import `dolfin 2017.1.0`
- `python3` cannot import `dolfin`

## Intended Use

Build the bootstrap image:

```bash
docker build -f install/docker/python3-bootstrap/Dockerfile -t finmag-py3-bootstrap .
```

Then run Python 3 checks against a writable copy of the checkout.

The helper script:

- `dev/bin/check-python3-in-container`

does this using `finmag/finmag:latest` directly and is the fastest way to
exercise the current Python 3 porting work without bind-mount write issues.

Examples:

```bash
dev/bin/check-python3-in-container
```

This runs the current default Python 3 migration file set.

```bash
dev/bin/check-python3-in-container src/finmag/__init__.py src/finmag/init.py
```

This runs a narrower syntax check on just the listed files.

What the script does:

1. bind-mounts the repo read-only into the container,
2. copies it to `/tmp/finmag-py3` inside the container,
3. runs `python3 -m py_compile` on the requested files from that writable copy.

This avoids the native-build write issues seen when running directly from a
bind-mounted checkout.

## Next Step

To get a real Python 3 Finmag runtime, a Python 3 build of the old DOLFIN stack
is required. That will likely need a dedicated source build or a different base
image strategy rather than Ubuntu 16.04 package installs.
