"""Smoke tests for the isolated M4 DOLFINx dependency lane.

These tests intentionally do not import :mod:`finmag`: the M4 environment is a
separate DOLFINx probe and must not inherit the FEniCS-2019/Python-3.11 M2/M3
stack. They document the minimal mesh/function operations that the future
Finmag-on-DOLFINx prototype can build on. [Codex gpt-5.5 high]
"""

import numpy as np
from mpi4py import MPI

import dolfinx
from dolfinx import fem, mesh


def test_dolfinx_imports_on_single_rank():
    """Check that the conda-forge DOLFINx package imports in the M4 env."""
    assert dolfinx.__version__
    assert MPI.COMM_WORLD.rank == 0


def test_unit_square_function_interpolation():
    """Exercise the smallest useful DOLFINx mesh/function workflow."""
    unit_square = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(unit_square, ("Lagrange", 1))

    field = fem.Function(function_space)
    field.interpolate(lambda x: x[0] + 2 * x[1])

    assert function_space.dofmap.index_map.size_global == 9
    assert np.isclose(np.sum(field.x.array), 13.5)
