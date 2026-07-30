"""Print the pinned DOLFINx runtime versions as stable, machine-readable JSON."""

import json
import platform

import dolfinx
import mpi4py
from mpi4py import MPI
import numpy
import petsc4py
from petsc4py import PETSc
import scipy


def version_report():
    """Return package and runtime versions needed to reproduce this lane."""
    petsc_version = ".".join(str(part) for part in PETSc.Sys.getVersion())
    mpi_standard = ".".join(str(part) for part in MPI.Get_version())
    mpi_library = " ".join(MPI.Get_library_version().replace("\x00", "").split())

    return {
        "dolfinx": dolfinx.__version__,
        "mpi": {
            "library": mpi_library,
            "mpi4py": mpi4py.__version__,
            "standard": mpi_standard,
        },
        "numpy": numpy.__version__,
        "petsc": {
            "petsc4py": petsc4py.__version__,
            "runtime": petsc_version,
        },
        "python": platform.python_version(),
        "scipy": scipy.__version__,
    }


if __name__ == "__main__":
    # Sorted, indented JSON is diffable in CI logs and easy to archive with a
    # numerical reference fixture. [Codex GPT-5]
    print(json.dumps(version_report(), indent=2, sort_keys=True))
