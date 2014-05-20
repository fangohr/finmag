import dolfin as df
import numpy as np
from field import Field


class TestField(object):
    def setup(self):
        # Create meshes of several dimensions.
        self.mesh1d = df.UnitIntervalMesh(10)
        self.mesh2d = df.UnitSquareMesh(7, 10)
        self.mesh3d = df.UnitCubeMesh(5, 7, 10)

        # All function spaces are CG, degree=1 unless named explicitly.

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

        # Set the tolerance used throughout all tests.
        self.tol = 1e-13

    def test_init_scalar_constant(self):
        # scalar function spaces on 1d, 2d and 3d mesh
        functionspaces = [self.fs_1d_scalar,
                          self.fs_2d_scalar,
                          self.fs_3d_scalar]

        # scalar field different expressions for constant value 42
        values = [df.Constant("42"),
                  df.Constant("42.0"),
                  df.Constant(42),
                  "42",
                  42,
                  42.0]

        for functionspace in functionspaces:
            for value in values:
                field = Field(functionspace, value)

                # check values in vector, should be exact
                assert np.all(field.f.vector().array() == 42)

                # check the result of get_coords_and_values, should be exact
                coords, field_values = field.get_coords_and_values()
                assert np.all(field_values == 42)

                # check values that are interpolated
                # dolfin is fairly inaccurate here, see field_test.ipynb
                probe_point = field.f.geometric_dimension() * (0.5,)
                assert abs(field.f(probe_point) - 42) < self.tol

    def test_init_scalar_expression(self):
        # scalar function spaces on 1d, 2d and 3d mesh
        functionspaces = [self.fs_1d_scalar,
                          self.fs_2d_scalar,
                          self.fs_3d_scalar]

        # scalar field different expressions for constant value 42
        expressions = [df.Expression("11.2*x[0]"),
                       df.Expression("11.2*x[0] - 3.1*x[1]"),
                       df.Expression("11.2*x[0] - 3.1*x[1] + 2.7*x[2]*x[2]")]

        for functionspace in functionspaces:
            mesh_dim = functionspace.mesh().topology().dim()
            coords = functionspace.mesh().coordinates()
            if mesh_dim == 1:
                # Use the first expression for 1d mesh.
                expression = expressions[0]
                # Compute expected values at mesh nodes.
                expected_values = 11.2*coords[:, 0]
                # Compute expected probed value. 
                expected_probed_value = 11.2*0.5
            elif mesh_dim == 2:
                # Use the second expression for 2d mesh.
                expression = expressions[1]
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1] 
                expected_probed_value = 11.2*0.5 - 3.1*0.5
            elif mesh_dim == 3:
                # Use the second expression for 2d mesh.
                expression = expressions[2]
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1] + \
                                  2.7*coords[:, 2]*coords[:, 2]
                expected_probed_value = 11.2*0.5 - 3.1*0.5 + 2.7*0.5**2

            field = Field(functionspace, expression)

            # Check the field value at all nodes (should be exact).
            field_values = field.get_coords_and_values()[1]
            assert np.all(field_values == expected_values)
            
            # Check the probed field value (not exact - interpolation).
            probed_value = field.probe_field(mesh_dim*(0.5,))
            assert abs(probed_value - expected_probed_value) < self.tol

    def test_get_coords_and_values_scalar_field(self):
        mesh = self.mesh3d
        f = Field(self.fs_3d_scalar)
        f.set(df.Expression('15.3*x[0] - 2.3*x[1] + 96.1*x[2]'))
        coords, values = f.get_coords_and_values()
        assert(np.allclose(coords, mesh.coordinates()))
        assert(values, 15.3*coords[:, 0] - 2.3*coords[:, 1] + 96.1*coords[:, 2])

    def test_probe_field_scalar(self):
        functionspaces = (self.fs_1d_scalar, self.fs_2d_scalar, self.fs_3d_scalar)
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

    """
    def test_get_coords_and_values_vector_field(self):
        mesh = self.mesh3d
        f = Field(self.fs_3d_vector)
        f.set(df.Expression(['x[0]', '2.3*x[1]', '-4.2*x[2]']))
        coords, values = f.get_coords_and_values()
        assert(np.allclose(coords, mesh.coordinates()))
        assert(np.allclose(values, np.array(mesh.coordinates()) * np.array([1, 2.3, -4.2])))
    """
