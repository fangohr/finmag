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
    assert np.array_equal(ordered, field.get_ordered_numpy_array_xxx())


def test_get_ordered_numpy_array_xyz_matches_mesh_vertex_order_vector():
    """Ordered vector values should be per-node and match mesh vertex order."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space)
    field.set(lambda x: np.vstack((x[0], x[1], np.zeros(x.shape[1]))))

    ordered = field.get_ordered_numpy_array_xyz()

    assert ordered.shape == (domain.geometry.x.shape[0] * 3,)
    values = ordered.reshape((-1, 3))
    assert np.allclose(values[:, 0], domain.geometry.x[:, 0])
    assert np.allclose(values[:, 1], domain.geometry.x[:, 1])
    assert np.allclose(values[:, 2], 0.0)


def test_set_with_ordered_numpy_array_xyz_round_trips():
    """Setting then getting an ordered array should reproduce it exactly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space)

    n_vertices = domain.geometry.x.shape[0]
    rng = np.random.default_rng(0)
    ordered_values = rng.uniform(size=(n_vertices, 3))

    field.set_with_ordered_numpy_array_xyz(ordered_values.reshape(-1))

    assert np.allclose(
        field.get_ordered_numpy_array_xyz(), ordered_values.reshape(-1)
    )

    component_blocked = ordered_values.T.reshape(-1)
    assert np.allclose(field.get_ordered_numpy_array_xxx(), component_blocked)

    round_trip = DOLFINxField(function_space)
    round_trip.set_with_ordered_numpy_array_xxx(component_blocked)
    assert np.allclose(
        round_trip.get_ordered_numpy_array_xyz(), ordered_values.reshape(-1)
    )


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


def test_set_dispatches_to_from_field_for_dolfinx_field():
    """Setting from another DOLFINxField should copy values on matching spaces."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    source = DOLFINxField(function_space, value=4.0)
    target = DOLFINxField(function_space)

    target.set(source)

    assert np.allclose(target.nodal_values(), 4.0)


def test_from_field_interpolates_between_different_spaces():
    """from_field should interpolate when the function spaces differ."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    source_space = fem.functionspace(domain, ("Lagrange", 1))
    target_space = fem.functionspace(domain, ("Lagrange", 2))
    source = DOLFINxField(source_space, value=3.0)
    target = DOLFINxField(target_space)

    target.from_field(source)

    assert np.isclose(target.average(), 3.0)


def test_from_field_rejects_non_field_argument():
    """from_field should reject anything that is not a DOLFINxField."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)

    with pytest.raises(TypeError, match="from_field requires another DOLFINxField"):
        field.from_field(3.0)


def test_set_dispatches_to_from_function():
    """Setting from a raw fem.Function should copy its dof array directly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    source_function = fem.Function(function_space)
    source_function.interpolate(lambda x: x[0])
    field = DOLFINxField(function_space)

    field.set(source_function)

    assert np.isclose(field.average(), 0.5)


def test_from_function_rejects_mismatched_function_space():
    """A function on a different space should fail explicitly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    space_a = fem.functionspace(domain, ("Lagrange", 1))
    space_b = fem.functionspace(domain, ("Lagrange", 2))
    other_function = fem.Function(space_b)
    field = DOLFINxField(space_a)

    with pytest.raises(ValueError, match="same function space"):
        field.from_function(other_function)


def test_set_random_values_fills_within_range():
    """set_random_values should draw values from the given range uniformly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)

    field.set_random_values(vrange=(-2.0, -1.0))

    assert np.all(field.f.x.array >= -2.0)
    assert np.all(field.f.x.array <= -1.0)


def test_is_constant_and_as_constant():
    """A uniform scalar field should be reported as constant with that value."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space, value=2.5)

    assert field.is_constant()
    assert np.isclose(field.as_constant(), 2.5)


def test_is_constant_false_for_varying_field():
    """A spatially varying scalar field should not be reported as constant."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)
    field.set(lambda x: x[0])

    assert not field.is_constant()
    with pytest.raises(RuntimeError, match="does not have a unique constant value"):
        field.as_constant()


def test_is_constant_rejects_vector_field():
    """is_constant/as_constant are only defined for scalar fields."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space, value=(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="only defined for scalar fields"):
        field.is_constant()
    with pytest.raises(ValueError, match="only defined for scalar fields"):
        field.as_constant()


def test_normalise_makes_every_node_unit_length():
    """normalise() should rescale every nodal vector to unit length."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space, value=(3.0, 0.0, 4.0))

    field.normalise()

    values = field.nodal_values()
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)
    assert np.allclose(values, (0.6, 0.0, 0.8))


def test_normalise_rejects_scalar_field():
    """normalise() is only meaningful for vector fields."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space, value=1.0)

    with pytest.raises(ValueError, match="only defined for vector fields"):
        field.normalise()


def test_normalise_rejects_zero_vector():
    """normalise() should reject a field with a zero-length nodal vector."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space, value=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="cannot normalise a zero vector"):
        field.normalise()


def test_coords_and_values_line_up_with_mesh_vertices():
    """coords_and_values should pair mesh coordinates with matching values."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)
    field.set(lambda x: x[0] + x[1])

    coords, values = field.coords_and_values()

    assert np.allclose(values, coords[:, 0] + coords[:, 1])


def test_allclose_compares_two_fields():
    """allclose should report whether two fields match within tolerance."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field_a = DOLFINxField(function_space, value=1.0)
    field_b = DOLFINxField(function_space, value=1.0)
    field_c = DOLFINxField(function_space, value=2.0)

    assert field_a.allclose(field_b)
    assert not field_a.allclose(field_c)


def test_allclose_rejects_non_field_argument():
    """allclose should reject anything that is not a DOLFINxField."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space, value=1.0)

    with pytest.raises(TypeError, match="allclose requires another DOLFINxField"):
        field.allclose(1.0)


def test_mesh_dim_reports_topological_dimension():
    """mesh_dim() should report the mesh's topological dimension."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space)

    assert field.mesh_dim() == 2


def test_save_pvd_writes_time_series(tmp_path):
    """save_pvd should append time steps to a single .pvd/.vtu series."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1))
    field = DOLFINxField(function_space, value=1.0, name="Ms")

    output_path = str(tmp_path / "field")
    field.save_pvd(output_path, t=0.0)
    field.save_pvd(output_path, t=1.0)
    field.close_pvd()

    assert (tmp_path / "field.pvd").exists()
    assert not hasattr(field, "_pvd_file")


def test_save_xdmf_writes_mesh_and_function(tmp_path):
    """save_xdmf should write an .xdmf/.h5 pair viewable in Paraview."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    field = DOLFINxField(function_space, value=(1.0, 0.0, 0.0), name="m")

    output_path = str(tmp_path / "field")
    field.save_xdmf(output_path, t=0.0)
    field.save_xdmf(output_path, t=1.0)
    field.close_xdmf()

    assert (tmp_path / "field.xdmf").exists()
    assert (tmp_path / "field.h5").exists()
    assert not hasattr(field, "_xdmf_file")
