"""Tests for the first DOLFINx-only micromagnetic prototype helpers.

These tests are not a legacy Finmag regression yet. They provide a small,
documented target for M4 so the future port can grow from checked DOLFINx
semantics rather than from untested exploratory snippets. [Codex gpt-5.5 high]
"""

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from dev.dolfinx.prototype import MU0, constant_vector_function
from dev.dolfinx.prototype import vector_function_space, zeeman_energy


def test_constant_vector_function_sets_all_components():
    """Check that vector field setup produces the requested constant values."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)

    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    values = magnetisation.x.array.reshape((-1, 3))
    assert np.allclose(values, (1.0, 0.0, 0.0))


def test_constant_vector_function_rejects_wrong_component_count():
    """Keep component-count mistakes explicit during the prototype phase."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)

    with pytest.raises(ValueError, match="function space expects 3"):
        constant_vector_function(function_space, (1.0, 0.0))


def test_zeeman_energy_for_constant_field_on_unit_square():
    """Compare DOLFINx Zeeman assembly with the analytical unit-square value."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = zeeman_energy(
        magnetisation,
        field=(2.0, 0.0, 0.0),
        saturation_magnetisation=3.0,
    )

    assert np.isclose(energy, -6.0 * MU0)


def test_zeeman_energy_applies_unit_length_scaling():
    """DOLFINx integrates over mesh coordinates, so physical length must scale."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = zeeman_energy(
        magnetisation,
        field=(2.0, 0.0, 0.0),
        saturation_magnetisation=3.0,
        unit_length=1e-9,
    )

    assert np.isclose(energy, -6.0 * MU0 * 1e-18)
