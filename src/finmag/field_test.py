import dolfin as df
import numpy as np
from field import Field


class TestField(object):
    def setup(self):
        # Create meshes of several dimensions.
        self.mesh1d = df.UnitIntervalMesh(10)
        self.mesh2d = df.UnitSquareMesh(7, 10)
        self.mesh3d = df.UnitCubeMesh(5, 7, 10)

        # All function spaces are CG (Lagrange)
        # with degree=1 unless named explicitly.

        # Create scalar function spaces.
        self.fs_1d_scalar = df.FunctionSpace(self.mesh1d,
                                             family="CG", degree=1)
        self.fs_2d_scalar = df.FunctionSpace(self.mesh2d,
                                             family="CG", degree=1)
        self.fs_3d_scalar = df.FunctionSpace(self.mesh3d,
                                             family="CG", degree=1)

        # Create vector (2d) function spaces.
        self.fs_1d_vector2d = df.VectorFunctionSpace(self.mesh1d,
                                                     family="CG",
                                                     degree=1, dim=2)
        self.fs_2d_vector2d = df.VectorFunctionSpace(self.mesh2d,
                                                     family="CG",
                                                     degree=1, dim=2)
        self.fs_3d_vector2d = df.VectorFunctionSpace(self.mesh3d,
                                                     family="CG",
                                                     degree=1, dim=2)

        # Create vector (3d) function spaces.
        self.fs_1d_vector3d = df.VectorFunctionSpace(self.mesh1d,
                                                     family="CG",
                                                     degree=1, dim=3)
        self.fs_2d_vector3d = df.VectorFunctionSpace(self.mesh2d,
                                                     family="CG",
                                                     degree=1, dim=3)
        self.fs_3d_vector3d = df.VectorFunctionSpace(self.mesh3d,
                                                     family="CG",
                                                     degree=1, dim=3)

        # Set the tolerance used throughout all tests
        # mainly due to interpolation errors.
        self.tol1 = 1e-13  # at mesh node
        self.tol2 = 1e-2  # outside mesh node

    def test_init_scalar_constant(self):
        # Scalar function spaces on 1d, 2d and 3d meshes.
        functionspaces = [self.fs_1d_scalar,
                          self.fs_2d_scalar,
                          self.fs_3d_scalar]

        # Scalar field different expressions for constant value 42.
        values = [df.Constant("42"),
                  df.Constant("42.0"),
                  df.Constant(42),
                  "42",
                  42,
                  42.0]

        # Test initialisation for all functionspaces and
        # all different expressions of constant 42.
        for functionspace in functionspaces:
            for value in values:
                field = Field(functionspace, value)

                # Check values in vector (numpy array) (should be exact).
                assert np.all(field.f.vector().array() == 42)

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values == 42)

                # Check values that are interpolated,
                # dolfin is fairly inaccurate here, see field_test.ipynb.
                # Probing is not at mesh node, but self.tol1 is used since
                # the field is constant and big discrepancy is not expected.
                probing_point = field.mesh_dim() * (0.55,)
                probed_value = field.probe_field(probing_point)
                assert abs(probed_value - 42) < self.tol1

    def test_init_scalar_expression(self):
        # Scalar function spaces on 1d, 2d and 3d meshes.
        functionspaces = [self.fs_1d_scalar,
                          self.fs_2d_scalar,
                          self.fs_3d_scalar]

        # Different expressions for scalar field initialisation,
        # depending on the mesh dimension (1d, 2d, and 3d).
        expressions = [df.Expression("11.2*x[0]"),
                       df.Expression("11.2*x[0] - 3.1*x[1]"),
                       df.Expression("11.2*x[0] - 3.1*x[1] + 2.7*x[2]*x[2]")]

        # Test initialisation for all functionspaces and
        # an appropriate expression for that functionspace.
        for functionspace in functionspaces:
            # Get the mesh dimension (1, 2, or 3).
            mesh_dim = functionspace.mesh().topology().dim()
            # Get the mesh coordinates.
            coords = functionspace.mesh().coordinates()
            if mesh_dim == 1:
                # Initialise the field using the first expression for 1d mesh.
                field = Field(functionspace, expressions[0])
                # Compute expected values at all mesh nodes.
                expected_values = 11.2*coords[:, 0]
                # Compute expected probed value (not at mesh node).
                expected_probed_value = 11.2*0.55
            elif mesh_dim == 2:
                # Initialise the field using the second expression for 2d mesh.
                field = Field(functionspace, expressions[1])
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1]
                expected_probed_value = 11.2*0.55 - 3.1*0.55
            elif mesh_dim == 3:
                # Initialise the field using the third expression for 3d mesh.
                field = Field(functionspace, expressions[2])
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1] + \
                    2.7*coords[:, 2]*coords[:, 2]
                expected_probed_value = 11.2*0.55 - 3.1*0.55 + 2.7*0.55**2

            # Check the field value at all nodes (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values == expected_values)

            # Check the probed field value (not exact - interpolation).
            probing_point = field.mesh_dim() * (0.55,)
            probed_value = field.probe_field(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol2

    def test_init_vector_constant(self):
        # 2d and 3d vector function spaces on 1d, 2d and 3d meshes.
        functionspaces = [self.fs_1d_vector2d,
                          self.fs_2d_vector2d,
                          self.fs_3d_vector2d,
                          self.fs_1d_vector3d,
                          self.fs_2d_vector3d,
                          self.fs_3d_vector3d]

        # Different constant expressions for 2d vector fields.
        constants2d = [df.Constant((0.1, -2.3)),
                       df.Constant([0.1, -2.3]),
                       df.Constant(np.array([0.1, -2.3])),
                       (0.1, -2.3),
                       [0.1, -2.3],
                       np.array([0.1, -2.3])]

        # Different constant expressions for 3d vector fields.
        constants3d = [df.Constant((0.15, 4.3, -6.41)),
                       df.Constant([0.15, 4.3, -6.41]),
                       df.Constant(np.array([0.15, 4.3, -6.41])),
                       (0.15, 4.3, -6.41),
                       [0.15, 4.3, -6.41],
                       np.array([0.15, 4.3, -6.41])]

        # Test initialisation for all functionspaces and
        # a whole set of constants appropriate for that functionspace.
        for functionspace in functionspaces:
            field = Field(functionspace)

            # Choose an appropriate set of constants and expected value
            if field.value_dim() == 2:  # 2d vector
                constants = constants2d
                expected_value = (0.1, -2.3)
            elif field.value_dim() == 3:  # 3d vector
                constants = constants3d
                expected_value = (0.15, 4.3, -6.41)

            n_nodes = functionspace.mesh().num_vertices()
            for constant in constants:
                field.set(constant)

                # Check values in vector (numpy array) (should be exact).
                f_array = field.f.vector().array()
                assert np.all(f_array[0:n_nodes] == expected_value[0])
                assert np.all(f_array[n_nodes:2*n_nodes] ==
                              expected_value[1])
                if field.value_dim() == 3:  # only for 3d vectors
                    assert np.all(f_array[2*n_nodes:3*n_nodes] ==
                                  expected_value[2])

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])
                if field.value_dim() == 3:  # only for 3d vectors
                    assert np.all(field_values[:, 2] == expected_value[2])

                # Check values that are interpolated,
                # dolfin is fairly inaccurate here, see field_test.ipynb.
                # Probing is not at mesh node, but self.tol1 is used since
                # the field is constant and big discrepancy is not expected.
                probing_point = field.mesh_dim() * (0.55,)
                probed_value = field.probe_field(probing_point)
                assert abs(probed_value[0] - expected_value[0]) < self.tol1
                assert abs(probed_value[1] - expected_value[1]) < self.tol1
                if field.value_dim() == 3:  # only for 3d vectors
                    assert abs(probed_value[2] - expected_value[2]) < self.tol1

    def test_coords_and_values_scalar_field(self):
        mesh = self.mesh3d
        f = Field(self.fs_3d_scalar)
        f.set(df.Expression('15.3*x[0] - 2.3*x[1] + 96.1*x[2]'))
        coords, values = f.coords_and_values()
        assert(np.allclose(coords, mesh.coordinates()))
        assert(values,
               15.3*coords[:, 0] - 2.3*coords[:, 1] + 96.1*coords[:, 2])

    def test_coords_and_values_vector_field(self):
        functionspaces2d = [self.fs_1d_vector2d,
                            self.fs_2d_vector2d,
                            self.fs_3d_vector2d]
        functionspaces3d = [self.fs_1d_vector3d,
                            self.fs_2d_vector3d,
                            self.fs_3d_vector3d]
        expressions2d = [df.Expression(['x[0]', 'x[0]']),
                         df.Expression(['x[0]', 'x[1]']),
                         df.Expression(['x[0] + x[1]', 'x[1]'])]
        expressions3d = [df.Expression(['x[0]', 'x[0]', 'x[0]']),
                         df.Expression(['x[0]', 'x[1]', 'x[0]+x[1]']),
                         df.Expression(['x[0]', '2*x[1]', 'x[2]'])]

        # TODO: Expand for 2d vectors.
        for functionspace in functionspaces3d:
            mesh_dim = functionspace.mesh().topology().dim()
            expected_coords = functionspace.mesh().coordinates()
            if mesh_dim == 1:
                expression = expressions3d[0]
            elif mesh_dim == 2:
                expression = expressions3d[1]
            elif mesh_dim == 3:
                expression = expressions3d[2]

            field = Field(functionspace, expression)
            coords, values = field.coords_and_values()
            assert np.allclose(coords, expected_coords)
            for i in xrange(len(coords)):
                assert np.allclose(values[i, :],
                                   field.probe_field(expected_coords[i]))

    def test_probe_field_scalar(self):
        functionspaces = (self.fs_1d_scalar,
                          self.fs_2d_scalar,
                          self.fs_3d_scalar)
        for functionspace in functionspaces:
            f = Field(functionspace)
            dim = f.f.geometric_dimension()
            probe_point = dim * (0.5,)
            if dim == 1:
                f.set(df.Expression('15.3*x[0]'))
                exact_result = 15.3*0.5
            elif dim == 2:
                f.set(df.Expression('15.3*x[0] - 2.3*x[1]'))
                exact_result = 15.3*0.5 - 2.3*0.5
            else:
                f.set(df.Expression('15.3*x[0] - 2.3*x[1] + 96.1*x[2]'))
                exact_result = 15.3*0.5 - 2.3*0.5 + 96.1*0.5
            assert abs(f.probe_field(probe_point) - exact_result) < 1e-13

    def test_mesh_dim(self):
        functionspaces = [self.fs_1d_scalar,
                          self.fs_2d_scalar,
                          self.fs_3d_scalar,
                          self.fs_1d_vector2d,
                          self.fs_2d_vector2d,
                          self.fs_3d_vector2d,
                          self.fs_1d_vector3d,
                          self.fs_2d_vector3d,
                          self.fs_3d_vector3d]

        mesh_dim = []
        for functionspace in functionspaces:
            field = Field(functionspace)
            mesh_dim.append(field.mesh_dim())

        expected_result = 3*[1, 2, 3]
        assert mesh_dim == expected_result

    def test_value_dim(self):
        functionspaces = [self.fs_1d_scalar,
                          self.fs_2d_scalar,
                          self.fs_3d_scalar,
                          self.fs_1d_vector2d,
                          self.fs_2d_vector2d,
                          self.fs_3d_vector2d,
                          self.fs_1d_vector3d,
                          self.fs_2d_vector3d,
                          self.fs_3d_vector3d]

        value_dim = []
        for functionspace in functionspaces:
            field = Field(functionspace)
            value_dim.append(field.value_dim())

        expected_result = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        assert value_dim == expected_result
