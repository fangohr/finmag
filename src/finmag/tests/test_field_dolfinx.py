"""Focused production tests for the DOLFINx-backed Field."""

import sys

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from ufl import Measure, dx

import finmag
from finmag.field import Field, associated_scalar_space


@pytest.fixture
def spaces():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    scalar = fem.functionspace(domain, ("Lagrange", 1))
    vector = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    return domain, scalar, vector


def test_public_field_resolves_without_legacy_dolfin():
    assert finmag.Field is Field
    assert "dolfin" not in sys.modules


def _domain(dimension):
    if dimension == 1:
        return mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    if dimension == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
    return mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)


@pytest.mark.parametrize("mesh_dimension", (1, 2, 3))
@pytest.mark.parametrize("components", (None, 2, 3, 4))
def test_dimensions_ordering_and_underlying_accessors(mesh_dimension, components):
    domain = _domain(mesh_dimension)
    element = (
        ("Lagrange", 1)
        if components is None
        else ("Lagrange", 1, (components,))
    )
    function_space = fem.functionspace(domain, element)
    value = (
        2.5
        if components is None
        else np.arange(1, components + 1, dtype=np.float64)
    )
    field = Field(function_space, value)

    assert field.functionspace is function_space
    assert field.f.function_space is function_space
    assert field.mesh() is domain
    assert field.mesh_dim() == mesh_dimension
    assert field.mesh_dofmap().bs == function_space.dofmap.bs
    assert np.array_equal(
        field.mesh_dofmap().list, function_space.dofmap.list
    )
    assert field.vector() is field.f.x
    assert field.as_vector() is field.f.x
    # Task 31: get_numpy_array_debug() returns the legacy component-blocked
    # (``xxx``) owned-vertex ordering, not the raw backend-order as_array().
    assert np.array_equal(
        field.get_numpy_array_debug(), field.get_ordered_numpy_array_xxx()
    )
    assert field.petsc_vector().getSize() == (
        function_space.dofmap.index_map.size_global
        * function_space.dofmap.index_map_bs
    )
    assert field.is_scalar_field() is (components is None)
    assert field.value_dim() == (1 if components is None else components)

    coordinates, values = field.coords_and_values()
    assert coordinates.shape[1] == mesh_dimension
    if components is None:
        assert values.shape == (coordinates.shape[0],)
        assert np.allclose(field.get_ordered_numpy_array_xyz(), values)
        assert np.allclose(field.get_ordered_numpy_array(), values)
    else:
        assert values.shape == (coordinates.shape[0], components)
        with pytest.raises(ValueError, match="scalar fields"):
            field.get_ordered_numpy_array()
        assert np.allclose(
            field.get_ordered_numpy_array_xyz(), values.reshape(-1)
        )
        assert np.allclose(
            field.get_ordered_numpy_array_xxx(), values.T.reshape(-1)
        )


def test_one_component_vector_is_not_scalar_and_can_be_normalised():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    function_space = fem.functionspace(domain, ("Lagrange", 1, (1,)))
    field = Field(function_space, (2.0,), normalised=True)

    assert not field.is_scalar_field()
    assert field.value_dim() == 1
    assert np.allclose(field.coords_and_values()[1], 1.0)
    assert np.allclose(field.average(), (1.0,))
    with pytest.raises(ValueError, match="scalar fields"):
        field.assert_is_scalar_field()


def test_constants_metadata_and_exact_associated_scalar_element(spaces):
    domain, scalar_space, vector_space = spaces
    scalar = Field(scalar_space, 4.25, name="Ms", unit="A/m")
    vector = Field(vector_space, fem.Constant(domain, (1.0, 2.0, 3.0)))

    assert scalar.name == "Ms"
    assert scalar.unit == "A/m"
    assert scalar.f.name == "Ms"
    assert scalar.is_constant()
    assert scalar.as_constant() == pytest.approx(4.25)
    assert np.allclose(vector.as_array().reshape((-1, 3)), (1.0, 2.0, 3.0))

    associated = associated_scalar_space(vector_space)
    assert associated.ufl_element() == scalar_space.ufl_element()

    dg_vector = fem.functionspace(domain, ("DG", 1, (3,)))
    dg_scalar = associated_scalar_space(dg_vector)
    assert dg_scalar.ufl_element() == fem.functionspace(
        domain, ("DG", 1)
    ).ufl_element()


def test_vectorized_and_branchy_pointwise_callables(spaces):
    _, scalar_space, vector_space = spaces
    scalar = Field(
        scalar_space,
        lambda point: 1.0 if point[0] < 0.5 else 2.0,
    )
    vector = Field(
        vector_space,
        lambda x: np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] + x[1])),
    )

    def pointwise_vector(point):
        if point[0] < 0.5:
            return (1.0, 0.0, 0.0)
        return (0.0, 1.0, 0.0)

    branchy_vector = Field(vector_space, pointwise_vector)

    coordinates, scalar_values = scalar.coords_and_values()
    expected_scalar = np.where(coordinates[:, 0] < 0.5, 1.0, 2.0)
    assert np.array_equal(scalar_values, expected_scalar)

    coordinates, vector_values = vector.coords_and_values()
    expected_vector = np.column_stack(
        (
            1.0 + coordinates[:, 0],
            2.0 + coordinates[:, 1],
            3.0 + coordinates[:, 0] + coordinates[:, 1],
        )
    )
    assert np.allclose(vector_values, expected_vector)
    coordinates, branchy_values = branchy_vector.coords_and_values()
    assert np.array_equal(
        branchy_values[:, 0], (coordinates[:, 0] < 0.5).astype(float)
    )
    assert np.array_equal(
        branchy_values[:, 1], (coordinates[:, 0] >= 0.5).astype(float)
    )


def test_set_routes_owned_raw_storage_and_scatter(spaces):
    _, scalar_space, vector_space = spaces
    source = Field(vector_space, (1.0, 2.0, 3.0))
    target = Field(vector_space, source.f)
    assert target.allclose(source)

    target.set(source)
    assert target.allclose(source)

    raw = np.arange(target.as_array().size, dtype=np.float64)
    target.set(raw)
    assert np.array_equal(target.as_array(), raw)

    scalar = Field(scalar_space, 0.0)
    scalar.set(np.full(scalar.as_array().shape, 7.0))
    assert scalar.as_constant() == pytest.approx(7.0)
    with pytest.raises(ValueError, match="owned"):
        scalar.set(np.zeros(scalar.local_array_with_ghosts().size + 1))


def test_from_field_interpolates_between_compatible_spaces(spaces):
    domain, scalar_space, _ = spaces
    quadratic_space = fem.functionspace(domain, ("Lagrange", 2))
    source = Field(scalar_space, lambda x: 1.0 + x[0] + 2.0 * x[1])
    target = Field(quadratic_space, source)
    owned = quadratic_space.dofmap.index_map.size_local
    coordinates = quadratic_space.tabulate_dof_coordinates()[:owned]

    assert np.allclose(
        target.as_array(), 1.0 + coordinates[:, 0] + 2.0 * coordinates[:, 1]
    )


def test_from_function_interpolates_dg0_into_cg1(spaces):
    """Value-level pin for the Task 16 ``Field.from_function`` cross-space
    interpolation superset (whole-branch review finding 7): a coefficient
    ``dolfinx.fem.Function`` living on a *different* space than the target
    ``Field`` -- e.g. a DG0 coefficient placed into a CG1 ``Field``, exactly
    the shape mismatch legacy hard-errored on -- is interpolated rather than
    rejected. A spatially uniform DG0 source makes the expected CG1 vertex
    values unambiguous regardless of which owning cell each vertex's
    interpolation samples from."""
    domain, scalar_space, _ = spaces
    dg0_space = fem.functionspace(domain, ("DG", 0))
    dg0_function = fem.Function(dg0_space)
    dg0_function.x.array[:] = 3.5

    target = Field(scalar_space, 0.0)
    target.from_function(dg0_function)

    owned = scalar_space.dofmap.index_map.size_local
    np.testing.assert_allclose(
        target.as_array()[:owned], 3.5, rtol=0, atol=1e-12)


def test_flat_xyz_xxx_round_trips_and_coords(spaces):
    _, _, vector_space = spaces
    field = Field(
        vector_space,
        lambda x: np.vstack((x[0], 10.0 + x[1], 20.0 + x[0] + x[1])),
    )
    coordinates, expected = field.coords_and_values()
    xyz = field.get_ordered_numpy_array_xyz()
    xxx = field.get_ordered_numpy_array_xxx()

    assert xyz.ndim == 1
    assert np.allclose(xyz.reshape((-1, 3)), expected)
    assert np.allclose(xxx.reshape((3, -1)), expected.T)
    assert np.allclose(field.np, expected.T)

    replacement = np.arange(xyz.size, dtype=np.float64)
    field.set_with_ordered_numpy_array_xyz(replacement)
    assert np.array_equal(field.get_ordered_numpy_array_xyz(), replacement)
    field.set_with_ordered_numpy_array_xxx(xxx)
    assert np.allclose(field.coords_and_values()[0], coordinates)
    assert np.allclose(field.get_ordered_numpy_array_xxx(), xxx)


def test_scalar_and_vector_fem_average_accept_dx_keyword(spaces):
    domain, scalar_space, vector_space = spaces
    scalar = Field(scalar_space, lambda x: x[0] + 2.0 * x[1])
    vector = Field(
        vector_space,
        lambda x: np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] + x[1])),
    )

    assert scalar.average(dx=dx) == pytest.approx(1.5)
    assert np.allclose(vector.average(), (1.5, 2.5, 4.0))

    cells = mesh.locate_entities(
        domain,
        domain.topology.dim,
        lambda x: x[0] <= 0.5 + 1e-12,
    )
    tags = mesh.meshtags(
        domain,
        domain.topology.dim,
        cells,
        np.ones(cells.size, dtype=np.int32),
    )
    left_half = Measure("dx", domain=domain, subdomain_data=tags)(1)
    x_coordinate = Field(scalar_space, lambda x: x[0])
    assert x_coordinate.average(dx=left_half) == pytest.approx(0.25)


def test_average_preserves_legitimate_tiny_physical_measure():
    domain = mesh.create_interval(MPI.COMM_WORLD, 2, (0.0, 1e-9))
    function_space = fem.functionspace(domain, ("Lagrange", 1))

    assert Field(function_space, 3.0).average() == pytest.approx(3.0)


def test_normalise_is_collective_and_normalised_constructor_routes(spaces):
    _, _, vector_space = spaces
    expected = np.array((2.0, 0.0, 0.0))
    function = fem.Function(vector_space)
    function.interpolate(lambda x: np.repeat(expected[:, None], x.shape[1], axis=1))
    function.x.scatter_forward()
    field = Field(vector_space, function, normalised=True)

    norms = np.linalg.norm(field.as_array().reshape((-1, 3)), axis=1)
    assert np.allclose(norms, 1.0)
    with pytest.raises(ValueError, match="zero vector"):
        Field(vector_space, (0.0, 0.0, 0.0), normalised=True)


def test_random_allclose_and_nonconstant_scalar(spaces):
    domain, scalar_space, _ = spaces
    first = Field(scalar_space, 0.0).set_random_values((-2.0, -1.0))
    second = Field(scalar_space, first)

    assert np.all(first.as_array() >= -2.0)
    assert np.all(first.as_array() <= -1.0)
    assert first.allclose(second)
    assert not first.is_constant()
    with pytest.raises(RuntimeError, match="unique constant"):
        first.as_constant()

    equivalent_but_distinct_space = fem.functionspace(
        domain, ("Lagrange", 1)
    )
    with pytest.raises(ValueError, match="same mesh and function space"):
        first.allclose(Field(equivalent_but_distinct_space, first))


def test_vtk_xdmf_output_and_filename_tracking(spaces, tmp_path):
    _, scalar_space, _ = spaces
    field = Field(scalar_space, lambda x: x[0] + x[1], name="value")
    vtk_path = str(tmp_path / "field")
    xdmf_path = str(tmp_path / "field")

    field.save_pvd(vtk_path, 0.0).save_pvd(vtk_path, 1.0)
    with pytest.raises(ValueError, match="already writing VTK"):
        field.save_pvd(str(tmp_path / "other"), 2.0)
    field.close_pvd()
    field.save_xdmf(xdmf_path, 0.0).save_xdmf(xdmf_path, 1.0)
    with pytest.raises(ValueError, match="already writing XDMF"):
        field.save_xdmf(str(tmp_path / "other"), 2.0)
    field.close_xdmf()

    assert (tmp_path / "field.pvd").exists()
    assert (tmp_path / "field.xdmf").exists()
    assert (tmp_path / "field.h5").exists()


def test_legacy_only_features_fail_precisely(spaces):
    _, scalar_space, _ = spaces
    field = Field(scalar_space, 1.0)

    with pytest.raises(NotImplementedError, match="string Expressions"):
        field.set("x[0]")
    with pytest.raises(NotImplementedError, match="Expression/UserExpression"):
        field.from_expression(object())
    # Point probing (field(x) / field.probe(x)) and get_spherical() are no
    # longer legacy-only failures -- restored in Task 26a; see
    # test_io_utils_dolfinx.py for their dedicated coverage.
    with pytest.raises(NotImplementedError, match="save_xdmf"):
        field.save_hdf5("field.h5")
    with pytest.raises(NotImplementedError, match="Field addition"):
        field + field


def test_coordinate_order_rejects_non_vertex_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    discontinuous = fem.functionspace(domain, ("DG", 0))
    field = Field(discontinuous, 1.0)

    with pytest.raises(ValueError, match="one dof per mesh vertex"):
        field.coords_and_values()


def test_owned_vertex_to_dof_raises_when_no_match_within_tolerance():
    """The tolerance-based vertex<->dof match must still fail loudly when the
    dof coordinates cannot be paired with the mesh vertices within tolerance
    (guarding the drift #11 fix against silently mis-ordering unrelated
    coordinates). A shim shifts every dof coordinate far from its vertex so no
    match is within the scale-relative tolerance."""
    from finmag.field import _owned_vertex_to_dof

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    scalar = fem.functionspace(domain, ("Lagrange", 1))

    class _ShiftedSpace:
        def __init__(self, base, shift):
            self._base = base
            self._shift = shift
            self.mesh = base.mesh
            self.dofmap = base.dofmap

        def tabulate_dof_coordinates(self):
            return self._base.tabulate_dof_coordinates() + self._shift

    # An exact (unshifted) shim still matches -- baseline for the guard.
    assert _owned_vertex_to_dof(_ShiftedSpace(scalar, 0.0)) is not None
    with pytest.raises(ValueError, match="could not match DOLFINx dofs"):
        _owned_vertex_to_dof(_ShiftedSpace(scalar, 100.0))


def test_field_coordinate_roundtrip_on_generated_mesh():
    """Regression for drift #11: on a Gmsh/from_csg-generated mesh (which
    carries ~1e-13 coordinate FP noise) the tolerance-based vertex<->dof match
    must round-trip a set field through ``coords_and_values`` without raising."""
    from finmag.util.geofile import from_csg

    domain = from_csg(
        "algebraic3d\nsolid c = orthobrick(0,0,0;10,10,10) -maxh=4.0;\ntlo c;\n",
        save_result=False,
    )
    vector = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    field = Field(vector)
    field.set(lambda x: np.stack(
        [np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])]))

    coords, values = field.coords_and_values()
    assert coords.shape[0] == values.shape[0]
    assert coords.shape[0] == domain.geometry.index_map().size_local
    assert np.allclose(values[:, 0], 1.0)
    assert np.allclose(values[:, 1:], 0.0)
