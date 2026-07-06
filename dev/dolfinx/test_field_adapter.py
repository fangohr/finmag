"""Tests for the DOLFINx-backed field compatibility adapter.

These tests intentionally target behaviours from ``finmag.field.Field`` that
the future DOLFINx port should preserve where practical. The adapter remains a
prototype under ``dev/dolfinx`` until it is mapped into the real Finmag design.
[Codex gpt-5.5 high]
"""

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from dev.dolfinx.field_adapter import DOLFINxField
from dev.dolfinx.prototype import vector_function_space


def test_scalar_field_sets_constant_and_reports_average():
    """A scalar field should accept constants and compute volume averages."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))

    field = DOLFINxField(function_space, value=3.0, name="Ms", unit="A/m")

    assert field.name == "Ms"
    assert field.unit == "A/m"
    assert field.is_scalar_field()
    assert field.value_dim() == 1
    assert np.allclose(field.nodal_values(), 3.0)
    assert np.isclose(field.average(), 3.0)


def test_scalar_field_rejects_vector_value():
    """Setting a vector into a scalar field should fail explicitly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)

    with pytest.raises(ValueError, match="cannot set scalar field with vector value"):
        field.set((1.0, 2.0, 3.0))


def test_vector_field_sets_constant_and_reports_average():
    """A vector field should keep component count and average semantics."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)

    field = DOLFINxField(function_space, value=(1.0, 2.0, 3.0))

    assert not field.is_scalar_field()
    assert field.value_dim() == 3
    assert np.allclose(field.nodal_values(), (1.0, 2.0, 3.0))
    assert np.allclose(field.average(), (1.0, 2.0, 3.0))


def test_vector_field_can_normalise_constant_values():
    """The adapter should preserve the legacy normalised constructor option."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)

    field = DOLFINxField(function_space, value=(3.0, 0.0, 4.0), normalised=True)

    assert np.allclose(field.nodal_values(), (0.6, 0.0, 0.8))
    assert np.allclose(field.average(), (0.6, 0.0, 0.8))


def test_vector_field_rejects_wrong_component_count():
    """Component-count mistakes should be visible before DOLFINx interpolation."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space)

    with pytest.raises(ValueError, match="field expects 3"):
        field.set((1.0, 2.0))


def test_field_accepts_dolfinx_callable_interpolation():
    """Callable interpolation is the first DOLFINx equivalent of Field.set."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))

    field = DOLFINxField(function_space)
    field.set(lambda x: x[0] + 2.0 * x[1])

    assert np.isclose(field.average(), 1.5)


def test_from_array_sets_raw_dof_array():
    """``from_array`` should assign the raw dof array directly (no reorder)."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space, value=0.0)

    raw = np.full(field.f.x.array.shape, 5.0)
    field.from_array(raw)

    assert np.allclose(field.f.x.array, 5.0)
    assert np.isclose(field.average(), 5.0)


def test_from_array_rejects_wrong_shape():
    """A raw dof array of the wrong shape should fail explicitly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)

    with pytest.raises(ValueError, match="from_array expects the raw dof array"):
        field.from_array(np.zeros(3))


def test_get_ordered_numpy_array_xyz_matches_mesh_vertex_order_scalar():
    """Ordered scalar values should line up with ``mesh.geometry.x`` rows."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)
    field.set(lambda x: x[0])

    ordered = field.get_ordered_numpy_array_xyz()

    assert np.allclose(ordered, domain.geometry.x[:, 0])


def test_get_ordered_numpy_array_xyz_matches_mesh_vertex_order_vector():
    """Ordered vector values should be per-node and match mesh vertex order."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space)
    field.set(lambda x: np.vstack((x[0], x[1], np.zeros(x.shape[1]))))

    ordered = field.get_ordered_numpy_array_xyz()

    assert ordered.shape == (domain.geometry.x.shape[0], 3)
    assert np.allclose(ordered[:, 0], domain.geometry.x[:, 0])
    assert np.allclose(ordered[:, 1], domain.geometry.x[:, 1])
    assert np.allclose(ordered[:, 2], 0.0)


def test_set_with_ordered_numpy_array_xyz_round_trips():
    """Setting then getting an ordered array should reproduce it exactly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space)

    n_vertices = domain.geometry.x.shape[0]
    rng = np.random.default_rng(0)
    ordered_values = rng.uniform(size=(n_vertices, 3))

    field.set_with_ordered_numpy_array_xyz(ordered_values)

    assert np.allclose(field.get_ordered_numpy_array_xyz(), ordered_values)


def test_set_with_ordered_numpy_array_xyz_rejects_wrong_shape():
    """A mesh-vertex-ordered array of the wrong shape should fail explicitly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space)

    with pytest.raises(
        ValueError, match="set_with_ordered_numpy_array_xyz expects shape"
    ):
        field.set_with_ordered_numpy_array_xyz(np.zeros((2, 3)))


def test_ordered_numpy_array_rejects_non_vertex_aligned_space():
    """DG0 has one dof per cell, not per vertex, so ordering is unsupported."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = fem.functionspace(domain, ("DG", 0))
    field = DOLFINxField(function_space, value=1.0)

    with pytest.raises(ValueError, match="one dof per mesh vertex"):
        field.get_ordered_numpy_array_xyz()
