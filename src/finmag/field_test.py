import dolfin as df
import numpy as np
import pytest
import os
from field import Field


class TestField(object):

    def setup(self):
        # Create meshes of several dimensions.
        self.mesh1d = df.UnitIntervalMesh(10)
        self.mesh2d = df.UnitSquareMesh(11, 10)
        self.mesh3d = df.UnitCubeMesh(9, 11, 10)

        # All function spaces are CG (Lagrange)
        # with degree=1 unless named explicitly.

        # Create scalar function spaces.
        self.fs1d_scalar = df.FunctionSpace(self.mesh1d,
                                            family="CG", degree=1)
        self.fs2d_scalar = df.FunctionSpace(self.mesh2d,
                                            family="CG", degree=1)
        self.fs3d_scalar = df.FunctionSpace(self.mesh3d,
                                            family="CG", degree=1)

        # Create 2D vector function spaces.
        self.fs1d_vector2d = df.VectorFunctionSpace(self.mesh1d,
                                                    family="CG",
                                                    degree=1, dim=2)
        self.fs2d_vector2d = df.VectorFunctionSpace(self.mesh2d,
                                                    family="CG",
                                                    degree=1, dim=2)
        self.fs3d_vector2d = df.VectorFunctionSpace(self.mesh3d,
                                                    family="CG",
                                                    degree=1, dim=2)

        # Create 3D vector function spaces.
        self.fs1d_vector3d = df.VectorFunctionSpace(self.mesh1d,
                                                    family="CG",
                                                    degree=1, dim=3)
        self.fs2d_vector3d = df.VectorFunctionSpace(self.mesh2d,
                                                    family="CG",
                                                    degree=1, dim=3)
        self.fs3d_vector3d = df.VectorFunctionSpace(self.mesh3d,
                                                    family="CG",
                                                    degree=1, dim=3)

        # Create 4D vector function spaces.
        self.fs1d_vector4d = df.VectorFunctionSpace(self.mesh1d,
                                                    family="CG",
                                                    degree=1, dim=4)
        self.fs2d_vector4d = df.VectorFunctionSpace(self.mesh2d,
                                                    family="CG",
                                                    degree=1, dim=4)
        self.fs3d_vector4d = df.VectorFunctionSpace(self.mesh3d,
                                                    family="CG",
                                                    degree=1, dim=4)

        # Create lists of meshes and functionspaces
        # to avoid the repetition of code in tests.
        self.meshes = [self.mesh1d, self.mesh2d, self.mesh3d]

        self.scalar_fspaces = [self.fs1d_scalar, self.fs2d_scalar,
                               self.fs3d_scalar]
        self.vector2d_fspaces = [self.fs1d_vector2d, self.fs2d_vector2d,
                                 self.fs3d_vector2d]
        self.vector3d_fspaces = [self.fs1d_vector3d, self.fs2d_vector3d,
                                 self.fs3d_vector3d]
        self.vector4d_fspaces = [self.fs1d_vector4d, self.fs2d_vector4d,
                                 self.fs3d_vector4d]
        self.all_fspaces = self.scalar_fspaces + self.vector2d_fspaces + \
            self.vector3d_fspaces + self.vector4d_fspaces

        # x, y, or z coordinate value for probing the field.
        self.probing_coord = 0.4351  # Not at the mesh node.

        # Set the tolerances used throughout all tests
        # mainly due to interpolation errors.

        # Tolerance value at the mesh node and
        # outside the mesh node for linear functions.
        self.tol1 = 5e-13

        # Tolerance value outside the mesh node for non-linear functions.
        self.tol2 = 1e-2  # outside the mesh node

        # Tolerance value for computing average and norm.
        self.tol3 = 5e-6

    def test_init(self):
        """Test the initialisation of field parameters."""
        for functionspace in self.all_fspaces:
            # Initialisation arguments.
            value = None  # Not specified, a zero-function is expected.
            normalised = True
            name = 'name_test'
            unit = 'unit_test'

            field = Field(functionspace, value, normalised, name, unit)

            assert field.functionspace == functionspace
            assert field.name == name
            assert field.unit == unit

            # Check that both function's name and label are changed.
            assert field.f.name() == name
            assert field.f.label() == name

            # Check that the created function is a dolfin zero function.
            assert isinstance(field.f, df.Function)
            assert np.all(field.coords_and_values()[1] == 0)

    def test_set_scalar_field_with_constant(self):
        """Test setting the scalar field with a constant."""
        # Different expressions of constant value 42 for scalar field setting.
        constants = [df.Constant(42), df.Constant(42.0), df.Constant("42"),
                     df.Constant("42.0"), 42, 42.0, "42",
                     "42.0", u"42", u"42.0"]

        expected_value = 42

        # Setting the scalar field for different
        # scalar function spaces and constants.
        for functionspace in self.scalar_fspaces:
            for constant in constants:
                field = Field(functionspace, constant)

                # Check vector (numpy array) values (should be exact).
                assert np.all(field.f.vector().array() == expected_value)

                # Check the result of coords_and_values (should be exact).
                field_values = field.coords_and_values()[1]  # coords ignored
                assert np.all(field_values == expected_value)

                # Check the interpolated value outside the mesh node.
                # The expected field is constant and, because of that,
                # smaller tolerance value (tol1) is used.
                probing_point = field.mesh_dim() * (self.probing_coord,)
                probed_value = field.probe(probing_point)
                assert abs(probed_value - expected_value) < self.tol1

    # TODO: Add tests to set scalar/vector field using a string.

    def test_set_scalar_field_with_expression(self):
        """Test setting the scalar field with an expression."""
        # Different expressions for setting the scalar field,
        # depending on the mesh dimension (1D, 2D, or 3D).
        expressions = [df.Expression("11.2*x[0]"),
                       df.Expression("11.2*x[0] - 3.01*x[1]"),
                       df.Expression("11.2*x[0] - 3.01*x[1] + 2.7*x[2]")]

        # Setting the scalar field for different
        # scalar function spaces and appropriate expressions.
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)

            # Set the field and compute expected values
            # depending on the mesh dimension.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(expressions[0])
                expected_values = 11.2 * coords[:, 0]
                expected_probed_value = 11.2 * self.probing_coord
            elif field.mesh_dim() == 2:
                field.set(expressions[1])
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1]
                expected_probed_value = (11.2 - 3.01) * self.probing_coord
            elif field.mesh_dim() == 3:
                field.set(expressions[2])
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1] + \
                    2.7 * coords[:, 2]
                expected_probed_value = (
                    11.2 - 3.01 + 2.7) * self.probing_coord

            # Check the result of coords_and_values (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values[:, 0] == expected_values)

            # Check the interpolated value outside the mesh node.
            # The expected field is linear and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_scalar_field_with_dolfin_function(self):
        """Test setting the scalar field with a dolfin function."""
        # Different expressions for defining the dolfin function,
        # depending on the mesh dimension (1D, 2D, or 3D).
        expressions = [df.Expression("11.2*x[0]"),
                       df.Expression("11.2*x[0] - 3.01*x[1]"),
                       df.Expression("11.2*x[0] - 3.01*x[1] + 2.7*x[2]")]

        # Setting the scalar field for different
        # scalar function spaces and appropriate expressions.
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)
            dolfin_function = df.Function(functionspace)

            # Set the field and compute expected values
            # depending on the mesh dimension.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                dolfin_function = df.interpolate(expressions[0], functionspace)
                field.set(dolfin_function)
                expected_values = 11.2 * coords[:, 0]
                expected_probed_value = 11.2 * self.probing_coord
            elif field.mesh_dim() == 2:
                dolfin_function = df.interpolate(expressions[1], functionspace)
                field.set(dolfin_function)
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1]
                expected_probed_value = (11.2 - 3.01) * self.probing_coord
            elif field.mesh_dim() == 3:
                dolfin_function = df.interpolate(expressions[2], functionspace)
                field.set(dolfin_function)
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1] + \
                    2.7 * coords[:, 2]
                expected_probed_value = (
                    11.2 - 3.01 + 2.7) * self.probing_coord

            # Check the result of coords_and_values (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values[:, 0] == expected_values)

            # Check the interpolated value outside the mesh node.
            # The expected field is linear and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_scalar_field_with_generic_vector(self):
        """Test setting the scalar field with a generic vector."""
        # Different expressions for defining the dolfin function,
        # depending on the mesh dimension (1D, 2D, or 3D).
        expressions = [df.Expression("11.2*x[0]"),
                       df.Expression("11.2*x[0] - 3.01*x[1]"),
                       df.Expression("11.2*x[0] - 3.01*x[1] + 2.7*x[2]")]

        # Setting the scalar field for different
        # scalar function spaces and appropriate expressions.
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)
            dolfin_function = df.Function(functionspace)

            # Set the field and compute expected values
            # depending on the mesh dimension.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                dolfin_function = df.interpolate(expressions[0], functionspace)
                field.set(dolfin_function.vector())
                expected_values = 11.2 * coords[:, 0]
                expected_probed_value = 11.2 * self.probing_coord
            elif field.mesh_dim() == 2:
                dolfin_function = df.interpolate(expressions[1], functionspace)
                field.set(dolfin_function.vector())
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1]
                expected_probed_value = (11.2 - 3.01) * self.probing_coord
            elif field.mesh_dim() == 3:
                dolfin_function = df.interpolate(expressions[2], functionspace)
                field.set(dolfin_function.vector())
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1] + \
                    2.7 * coords[:, 2]
                expected_probed_value = (
                    11.2 - 3.01 + 2.7) * self.probing_coord

            # Check the result of coords_and_values (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values[:, 0] == expected_values)

            # Check the interpolated value outside the mesh node.
            # The expected field is linear and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_scalar_field_with_python_function(self):
        """Test setting the scalar field with a python function."""
        # Python functions array for setting the scalar field.
        python_functions = [lambda x:1.21 * x[0],
                            lambda x:1.21 * x[0] - 3.21 * x[1],
                            lambda x:1.21 * x[0] - 3.21 * x[1] + 2.47 * x[2]]

        # Setting the scalar field for different
        # scalar function spaces and appropriate python functions.
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)

            # Set the field and compute expected values
            # depending on the mesh dimension.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(python_functions[0])
                expected_values = 1.21 * coords[:, 0]
                expected_probed_value = 1.21 * self.probing_coord
            elif field.mesh_dim() == 2:
                field.set(python_functions[1])
                expected_values = 1.21 * coords[:, 0] - 3.21 * coords[:, 1]
                expected_probed_value = (1.21 - 3.21) * self.probing_coord
            elif field.mesh_dim() == 3:
                field.set(python_functions[2])
                expected_values = 1.21 * coords[:, 0] - 3.21 * coords[:, 1] + \
                    2.47 * coords[:, 2]
                expected_probed_value = (
                    1.21 - 3.21 + 2.47) * self.probing_coord

            # Check the result of coords_and_values (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values[:, 0] == expected_values)

            # Check the interpolated value outside the mesh node.
            # The expected field is linear and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_vector_field_with_constant(self):
        """Test setting the 3D vector field with a constant."""
        # Different expressions of constant for vector field setting.
        constants = [df.Constant((0.15, -2.3, -6.41)),
                     df.Constant([0.15, -2.3, -6.41]),
                     df.Constant(np.array([0.15, -2.3, -6.41])),
                     (0.15, -2.3, -6.41),
                     [0.15, -2.3, -6.41],
                     np.array([0.15, -2.3, -6.41])]

        expected_value = (0.15, -2.3, -6.41)

        # Test setting the vector field for different
        # vector function spaces and constants.
        for functionspace in self.vector3d_fspaces:
            for constant in constants:
                field = Field(functionspace, constant)

                # Check vector (numpy array) values (should be exact).
                f_array = field.f.vector().array()
                f_array_split = np.split(f_array, field.value_dim())
                assert np.all(f_array_split[0] == expected_value[0])
                assert np.all(f_array_split[1] == expected_value[1])
                assert np.all(f_array_split[2] == expected_value[2])

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])
                assert np.all(field_values[:, 2] == expected_value[2])

                # Check the interpolated value outside the mesh node.
                # The expected field is constant and, because of that,
                # smaller tolerance value (tol1) is used.
                probing_point = field.mesh_dim() * (self.probing_coord,)
                probed_value = field.probe(probing_point)
                assert abs(probed_value[0] - expected_value[0]) < self.tol1
                assert abs(probed_value[1] - expected_value[1]) < self.tol1
                assert abs(probed_value[2] - expected_value[2]) < self.tol1


    def test_setting_field_with_argument_of_incorrect_dimension_raises_ValueError(self):
        # Check that we get a decent error (rather than the generic
        # RuntimError thrown by dolfin) if we try to set a field with
        # a value whose dimension doesn't match the function space.

        # Try to set scalar field with a vector value
        field = Field(self.fs3d_scalar)
        with pytest.raises(ValueError):
            field.set([1, 0, 0])

        # Try to set vector field with a scalar value
        field = Field(self.fs2d_vector3d)
        with pytest.raises(ValueError):
            field.set(42.0)

        # Try to set 2D vector field with a 3D vector
        field = Field(self.fs3d_vector2d)
        with pytest.raises(ValueError):
            field.set([1, 0, 0])

        # Try to set 2D vector field with a 3D vector
        field = Field(self.fs3d_vector2d)
        with pytest.raises(ValueError):
            field.set(["x[0]", "1", "0"])

    def test_set_vector_field_with_expression(self):
        """Test setting the 3D vector field with an expression."""
        # Different expressions for 3D vector fields.
        expressions = [df.Expression(['1.1*x[0]', '-2.4*x[0]', '3*x[0]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[1]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[2]'])]

        # Test setting the vector field for different
        # vector function spaces and appropriate expressions.
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

            # Set the vector field and compute expected values.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(expressions[0])
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 0],
                                   3 * coords[:, 0])
            elif field.mesh_dim() == 2:
                field.set(expressions[1])
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 1])
            elif field.mesh_dim() == 3:
                field.set(expressions[2])
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 2])

            # Compute expected probed value.
            expected_probed_value = (1.1 * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            assert np.all(f_array_split[2] == expected_values[2])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            assert np.all(field_values[:, 2] == expected_values[2])

            # Check the interpolated value outside the mesh node.
            # The expected field is constant and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector_field_with_dolfin_function(self):
        """Test setting the 3D vector field with a dolfin function."""
        # Different expressions for 3D vector fields.
        expressions = [df.Expression(['1.1*x[0]', '-2.4*x[0]', '3*x[0]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[1]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[2]'])]

        # Test setting the vector field for different
        # vector function spaces and appropriate expressions.
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)
            dolfin_function = df.Function(functionspace)

            # Set the vector field and compute expected values.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                dolfin_function = df.interpolate(expressions[0], functionspace)
                field.set(dolfin_function)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 0],
                                   3 * coords[:, 0])
            elif field.mesh_dim() == 2:
                dolfin_function = df.interpolate(expressions[1], functionspace)
                field.set(dolfin_function)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 1])
            elif field.mesh_dim() == 3:
                dolfin_function = df.interpolate(expressions[2], functionspace)
                field.set(dolfin_function)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 2])

            # Compute expected probed value.
            expected_probed_value = (1.1 * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            assert np.all(f_array_split[2] == expected_values[2])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            assert np.all(field_values[:, 2] == expected_values[2])

            # Check the interpolated value outside the mesh node.
            # The expected field is constant and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector_field_with_generic_vector(self):
        """Test setting the 3D vector field with a generic_vector."""
        # Different expressions for 3D vector fields.
        expressions = [df.Expression(['1.1*x[0]', '-2.4*x[0]', '3*x[0]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[1]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[2]'])]

        # Test setting the vector field for different
        # vector function spaces and appropriate expressions.
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)
            dolfin_function = df.Function(functionspace)

            # Set the vector field and compute expected values.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                dolfin_function = df.interpolate(expressions[0], functionspace)
                field.set(dolfin_function.vector())
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 0],
                                   3 * coords[:, 0])
            elif field.mesh_dim() == 2:
                dolfin_function = df.interpolate(expressions[1], functionspace)
                field.set(dolfin_function.vector())
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 1])
            elif field.mesh_dim() == 3:
                dolfin_function = df.interpolate(expressions[2], functionspace)
                field.set(dolfin_function.vector())
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 2])

            # Compute expected probed value.
            expected_probed_value = (1.1 * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            assert np.all(f_array_split[2] == expected_values[2])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            assert np.all(field_values[:, 2] == expected_values[2])

            # Check the interpolated value outside the mesh node.
            # The expected field is constant and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector_field_with_python_function(self):
        """Test setting the 3D vector field with a python function."""
        # Different python functions for setting the vector field.
        python_functions = [lambda x:(1.21 * x[0], -2.47 * x[0], 3 * x[0]),
                            lambda x:(1.21 * x[0], -2.47 * x[1], 3 * x[1]),
                            lambda x:(1.21 * x[0], -2.47 * x[1], 3 * x[2])]

        # Test setting the vector field for different
        # vector function spaces and appropriate expressions.
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

            # Set the vector field and compute expected values.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(python_functions[0])
                expected_values = (1.21 * coords[:, 0], -2.47 * coords[:, 0],
                                   3 * coords[:, 0])
            elif field.mesh_dim() == 2:
                field.set(python_functions[1])
                expected_values = (1.21 * coords[:, 0], -2.47 * coords[:, 1],
                                   3 * coords[:, 1])
            elif field.mesh_dim() == 3:
                field.set(python_functions[2])
                expected_values = (1.21 * coords[:, 0], -2.47 * coords[:, 1],
                                   3 * coords[:, 2])

            # Compute expected probed value.
            expected_probed_value = (1.21 * self.probing_coord,
                                     -2.47 * self.probing_coord,
                                     3 * self.probing_coord)

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            assert np.all(f_array_split[2] == expected_values[2])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            assert np.all(field_values[:, 2] == expected_values[2])

            # Check the interpolated value outside the mesh node.
            # The expected field is constant and, because of that,
            # smaller tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector2d_field(self):
        """Test setting the 2D vector field."""
        # Different values for setting the 2D vector field.
        expressions = [df.Constant((1.1, -2.4)),
                       (1.1, -2.4),
                       [1.1, -2.4],
                       df.Expression(('1.1', '-2.4')),
                       lambda x:(1.1, -2.4)]

        expected_value = (1.1, -2.4)

        # Test setting the 2D vector field for different
        # vector function spaces and constants.
        for functionspace in self.vector2d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)

                # Check vector (numpy array) values (should be exact).
                f_array = field.f.vector().array()
                f_array_split = np.split(f_array, field.value_dim())
                assert np.all(f_array_split[0] == expected_value[0])
                assert np.all(f_array_split[1] == expected_value[1])

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])

                # Check the interpolated value outside the mesh node.
                # The expected field is constant and, because of that,
                # smaller tolerance value (tol1) is used.
                probing_point = field.mesh_dim() * (self.probing_coord,)
                probed_value = field.probe(probing_point)
                assert abs(probed_value[0] - expected_value[0]) < self.tol1
                assert abs(probed_value[1] - expected_value[1]) < self.tol1

    def test_set_vector4d_field(self):
        """Test setting the 4D vector field."""
        # Different values for setting the 4D vector field.
        expressions = [df.Constant((1.1, -2.4, 0, 0.9)),
                       (1.1, -2.4, 0, 0.9),
                       [1.1, -2.4, 0, 0.9],
                       df.Expression(('1.1', '-2.4', '0', '0.9')),
                       lambda x:(1.1, -2.4, 0, 0.9)]

        expected_value = (1.1, -2.4, 0, 0.9)

        # Test setting the 4D vector field for different
        # vector function spaces and constants.
        for functionspace in self.vector4d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)

                # Check vector (numpy array) values (should be exact).
                f_array = field.f.vector().array()
                f_array_split = np.split(f_array, field.value_dim())
                assert np.all(f_array_split[0] == expected_value[0])
                assert np.all(f_array_split[1] == expected_value[1])
                assert np.all(f_array_split[2] == expected_value[2])
                assert np.all(f_array_split[3] == expected_value[3])

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])
                assert np.all(field_values[:, 2] == expected_value[2])
                assert np.all(field_values[:, 3] == expected_value[3])

                # Check the interpolated value outside the mesh node.
                # The expected field is constant and, because of that,
                # smaller tolerance value (tol1) is used.
                probing_point = field.mesh_dim() * (self.probing_coord,)
                probed_value = field.probe(probing_point)
                assert abs(probed_value[0] - expected_value[0]) < self.tol1
                assert abs(probed_value[1] - expected_value[1]) < self.tol1
                assert abs(probed_value[2] - expected_value[2]) < self.tol1
                assert abs(probed_value[3] - expected_value[3]) < self.tol1

    def test_normalise(self):
        """Test normalisation of vector fields."""
        # 2D vector field
        value = (1.3, 3.6)
        for functionspace in self.vector2d_fspaces:
            field = Field(functionspace, value, normalised=True)

            values = field.coords_and_values()[1]  # Ignore coordinates.

            # Check the components of vector field.
            norm_exact = (value[0] ** 2 + value[1] ** 2) ** 0.5
            normalised_c1 = value[0] / norm_exact
            normalised_c2 = value[1] / norm_exact
            assert np.all(abs(values[:, 0] - normalised_c1) < self.tol3)
            assert np.all(abs(values[:, 1] - normalised_c2) < self.tol3)

            # Check the norm of normalised vector field.
            norm = (values[:, 0] ** 2 + values[:, 1] ** 2) ** 0.5
            assert np.all(abs(norm - 1) < self.tol3)

        # 3D vector field
        value = (-1.3, 3.16, 0)
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace, value, normalised=True)

            values = field.coords_and_values()[1]  # Ignore coordinates.

            # Check the components of vector field.
            norm_exact = (value[0] ** 2 + value[1] ** 2 + value[2] ** 2) ** 0.5
            normalised_c1 = value[0] / norm_exact
            normalised_c2 = value[1] / norm_exact
            normalised_c3 = value[2] / norm_exact
            assert np.all(abs(values[:, 0] - normalised_c1) < self.tol3)
            assert np.all(abs(values[:, 1] - normalised_c2) < self.tol3)
            assert np.all(abs(values[:, 2] - normalised_c3) < self.tol3)

            # Check the norm of normalised vector field.
            norm = (
                values[:, 0] ** 2 + values[:, 1] ** 2 + values[:, 2] ** 2) ** 0.5
            assert np.all(abs(norm - 1) < self.tol3)

        # 4D vector field
        value = (-1.23, -3.96, 0, 6.98)
        for functionspace in self.vector4d_fspaces:
            field = Field(functionspace, value, normalised=True)

            values = field.coords_and_values()[1]  # Ignore coordinates.

            # Check the components of vector field.
            norm_exact = (value[0] ** 2 + value[1] ** 2 + value[2] ** 2 +
                          value[3] ** 2) ** 0.5
            normalised_c1 = value[0] / norm_exact
            normalised_c2 = value[1] / norm_exact
            normalised_c3 = value[2] / norm_exact
            normalised_c4 = value[3] / norm_exact
            assert np.all(abs(values[:, 0] - normalised_c1) < self.tol3)
            assert np.all(abs(values[:, 1] - normalised_c2) < self.tol3)
            assert np.all(abs(values[:, 2] - normalised_c3) < self.tol3)
            assert np.all(abs(values[:, 3] - normalised_c4) < self.tol3)

            # Check the norm of normalised vector field.
            norm = (values[:, 0] ** 2 + values[:, 1] ** 2 + values[:, 2] ** 2 +
                    values[:, 3] ** 2) ** 0.5
            assert np.all(abs(norm - 1) < self.tol3)

        # Test normalisation if field is set using
        # dolfin expression or python function.
        expressions = [lambda x:(11.2 * x[0], -1.6 * x[1], 0.3 * x[2]),
                       df.Expression(['11.2*x[0]', '-1.6*x[1]', '0.3*x[2]'])]

        functionspace = self.fs3d_vector3d

        for expression in expressions:
            field = Field(functionspace, expression, normalised=True)
            values = field.coords_and_values()[1]  # Ignore coordinates.

            # Check the norm of normalised vector field.
            norm = (
                values[:, 0] ** 2 + values[:, 1] ** 2 + values[:, 2] ** 2) ** 0.5
            assert np.all(abs(norm - 1) < 0.1)  # Too big error!!!!

    def test_whether_field_is_scalar_field(self):
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace, 42)
            assert field.is_scalar_field()

        for functionspace in self.vector2d_fspaces:
            field = Field(functionspace, [42, 23])
            assert not field.is_scalar_field()

        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace, [42, 23, 12])
            assert not field.is_scalar_field()

        for functionspace in self.vector4d_fspaces:
            field = Field(functionspace, [42, 23, 12, 5])
            assert not field.is_scalar_field()

    def test_convert_scalar_field_to_constant_value(self):
        """
        Check that calling 'as_constant()' on a constant scalar field returns
        the unique field value. Also check that calling 'as_constant()' on a
        non-constant scalar field raises an exception.

        """
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace, 42.0)
            assert field.is_constant()
            assert field.as_constant() == 42.0

        for functionspace in self.scalar_fspaces:
            field = Field(functionspace, 'x[0]')
            assert not field.is_constant()
            with pytest.raises(RuntimeError):
                field.as_constant()

    def test_average_scalar_field(self):
        """Test computing the scalar field average."""
        # Different expressions for setting the scalar field.
        # All expressions set the field with same average value.

        # TODO: Add test for computing average on different mesh regions.
        expressions = [df.Constant(5),
                       df.Expression('10*x[0]'),
                       lambda x:10 * x[0]]

        f_av_expected = 5

        for functionspace in self.scalar_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                # Check the average value.
                assert abs(f_av - f_av_expected) < self.tol1

                # Check the type of average result.
                assert isinstance(f_av, float)

    def test_average_vector_field(self):
        """Test computing the vector field average."""
        # Different expressions for setting the 2D vector field.
        # All expressions set the field with same average value.
        expressions = [df.Constant((1, 5.1)),
                       df.Expression(['2*x[0]', '10.2*x[0]']),
                       lambda x:(2 * x[0], 10.2 * x[0])]

        f_av_expected = (1, 5.1)

        for functionspace in self.vector2d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                # Check the average values for all components.
                assert abs(f_av[0] - f_av_expected[0]) < self.tol1
                assert abs(f_av[1] - f_av_expected[1]) < self.tol1

                # Check the type and shape of average result.
                assert isinstance(f_av, np.ndarray)
                assert f_av.shape == (field.value_dim(),)

        # Different expressions for setting the 3D vector field.
        # All expressions set the field with same average value.
        expressions = [df.Constant((1, 5.1, -3.6)),
                       df.Expression(['2*x[0]', '10.2*x[0]', '-7.2*x[0]']),
                       lambda x:(2 * x[0], 10.2 * x[0], -7.2 * x[0])]

        f_av_expected = (1, 5.1, -3.6)

        for functionspace in self.vector3d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                # Check the average values for all components.
                assert abs(f_av[0] - f_av_expected[0]) < self.tol1
                assert abs(f_av[1] - f_av_expected[1]) < self.tol1
                assert abs(f_av[2] - f_av_expected[2]) < self.tol1

                # Check the type and shape of average result.
                assert isinstance(f_av, np.ndarray)
                assert f_av.shape == (field.value_dim(),)

        # Different expressions for setting the 4D vector field.
        # All expressions set the field with same average value.
        expressions = [df.Constant((1, 5.1, -3.6, 0)),
                       df.Expression(['2*x[0]', '10.2*x[0]',
                                      '-7.2*x[0]', '0']),
                       lambda x:(2 * x[0], 10.2 * x[0], -7.2 * x[0], 0)]

        f_av_expected = (1, 5.1, -3.6, 0)

        for functionspace in self.vector4d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                # Check the average values for all components.
                assert abs(f_av[0] - f_av_expected[0]) < self.tol1
                assert abs(f_av[1] - f_av_expected[1]) < self.tol1
                assert abs(f_av[2] - f_av_expected[2]) < self.tol1
                assert abs(f_av[3] - f_av_expected[3]) < self.tol1

                # Check the type and shape of average result.
                assert isinstance(f_av, np.ndarray)
                assert f_av.shape == (field.value_dim(),)

    def test_coords_and_values_scalar_field(self):
        """Test coordinates and values for scalar field."""
        # Test for scalar fields on 1D, 2D, and 3D meshes,
        # initialised with a dolfin expression.
        expression = df.Expression('1.3*x[0]')

        for functionspace in self.scalar_fspaces:
            expected_coords = functionspace.mesh().coordinates()
            num_nodes = functionspace.mesh().num_vertices()
            expected_values = 1.3 * expected_coords[:, 0]

            field = Field(functionspace, expression)
            coords, values = field.coords_and_values()

            # Type of results must be numpy array.
            assert isinstance(coords, np.ndarray)
            assert isinstance(values, np.ndarray)

            # Check the shape of results.
            assert values.shape == (num_nodes, field.value_dim())
            assert coords.shape == (num_nodes, field.mesh_dim())

            # Check values of results.
            assert np.all(coords == expected_coords)
            assert np.all(values[:, 0] == expected_values)

    def test_coords_and_values_vector_field(self):
        """Test coordinates and values for vector field."""
        # Different expressions for 3D vector fields.
        expression = df.Expression(['1.03*x[0]', '2.31*x[0]', '-1*x[0]'])

        for functionspace in self.vector3d_fspaces:
            # Initialise the field with an appropriate expression for
            # the function space and compute expected results.
            expected_coords = functionspace.mesh().coordinates()
            num_nodes = functionspace.mesh().num_vertices()

            expected_values = (1.03 * expected_coords[:, 0],
                               2.31 * expected_coords[:, 0],
                               -1 * expected_coords[:, 0])

            field = Field(functionspace, expression)
            coords, values = field.coords_and_values()

            # Type of results must be numpy array.
            assert isinstance(coords, np.ndarray)
            assert isinstance(values, np.ndarray)

            # Check the shape of results.
            assert values.shape == (num_nodes, field.value_dim())
            assert coords.shape == (num_nodes, field.mesh_dim())

            # Check values of results.
            assert np.all(coords == expected_coords)
            assert np.all(values[:, 0] == expected_values[0])
            assert np.all(values[:, 1] == expected_values[1])
            assert np.all(values[:, 2] == expected_values[2])

    def test_probe_scalar_field(self):
        """Test probing the scalar field."""
        # Test probing field at and outside the mesh node for scalar field and
        # an appropriate expression for setting the value.
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)
            mesh_dim = field.mesh_dim()

            if mesh_dim == 1:
                field.set(df.Expression('1.3*x[0]'))
                exact_result_at_node = 1.3 * 0.5
                exact_result_out_node = 1.3 * self.probing_coord
            elif mesh_dim == 2:
                field.set(df.Expression('1.3*x[0] - 2.3*x[1]'))
                exact_result_at_node = (1.3 - 2.3) * 0.5
                exact_result_out_node = (1.3 - 2.3) * self.probing_coord
            elif mesh_dim == 3:
                field.set(df.Expression('1.3*x[0] - 2.3*x[1] + 6.1*x[2]'))
                exact_result_at_node = (1.3 - 2.3 + 6.1) * 0.5
                exact_result_out_node = (1.3 - 2.3 + 6.1) * self.probing_coord

            # Probe and check the result at the mesh node.
            probe_point = mesh_dim * (0.5,)
            probed_value = field.probe(probe_point)
            assert isinstance(probed_value, float)
            assert abs(probed_value - exact_result_at_node) < self.tol1

            # Probe and check the result outside the mesh node.
            probe_point = mesh_dim * (self.probing_coord,)
            probed_value = field.probe(probe_point)
            assert isinstance(probed_value, float)
            assert abs(probed_value - exact_result_out_node) < self.tol1

    def test_probe_vector_field(self):
        """Test probing the vector field."""
        # Test probing field at and outside the mesh node for vector field and
        # an appropriate expression for setting the value.
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace,
                          df.Expression(['1.3*x[0]', '0.3*x[0]', '-6.2*x[0]']))
            mesh_dim = field.mesh_dim()

            exact_result_at_node = (1.3 * 0.5, 0.3 * 0.5, -6.2 * 0.5)
            exact_result_out_node = (1.3 * self.probing_coord,
                                     0.3 * self.probing_coord,
                                     -6.2 * self.probing_coord)

            # Probe and check the result at the mesh node.
            probe_point = mesh_dim * (0.5,)
            probed_value = field.probe(probe_point)
            assert isinstance(probed_value, np.ndarray)
            assert len(probed_value) == 3
            assert abs(probed_value[0] - exact_result_at_node[0]) < self.tol1
            assert abs(probed_value[1] - exact_result_at_node[1]) < self.tol1
            assert abs(probed_value[2] - exact_result_at_node[2]) < self.tol1

            # Probe and check the result outside the mesh node.
            probe_point = mesh_dim * (self.probing_coord,)
            probed_value = field.probe(probe_point)
            assert isinstance(probed_value, np.ndarray)
            assert len(probed_value) == 3
            assert abs(probed_value[0] - exact_result_out_node[0]) < self.tol1
            assert abs(probed_value[1] - exact_result_out_node[1]) < self.tol1
            assert abs(probed_value[2] - exact_result_out_node[2]) < self.tol1

    def test_mesh_dim(self):
        """Test mesh_dim method."""
        for functionspace in self.all_fspaces:
            field = Field(functionspace)
            mesh_dim_expected = functionspace.mesh().topology().dim()

            assert isinstance(field.mesh_dim(), int)
            assert field.mesh_dim() == mesh_dim_expected

    def test_value_dim(self):
        """Test value_dim method."""
        for functionspace in self.all_fspaces:
            field = Field(functionspace)
            value_dim_expected = functionspace.ufl_element().value_shape()
            assert isinstance(field.value_dim(), int)
            if isinstance(functionspace, df.FunctionSpace):
                assert field.value_dim() == 1
            elif isinstance(functionspace, df.VectorFunctionSpace):
                assert field.value_dim() == value_dim_expected[0]

    def test_mesh(self):
        """Test mesh method."""
        for functionspace in self.all_fspaces:
            field = Field(functionspace)

            assert isinstance(field.mesh(), df.Mesh)

    def test_set_nonlinear_scalar_field(self):
        """Test setting nonlinear scalar field."""
        # Python functions array for setting the scalar field.
        python_functions = [lambda x:1.21 * x[0] * x[0],
                            lambda x:1.21 * x[0] * x[0] - 3.21 * x[1],
                            lambda x:1.21 * x[0] * x[0] - 3.21 * x[1] + 2.47 * x[2]]

        # Setting the scalar field for different
        # scalar function spaces and appropriate python functions.
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)

            # Set the field and compute expected values
            # depending on the mesh dimension.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(python_functions[0])
                expected_values = 1.21 * coords[:, 0] * coords[:, 0]
                expected_probed_value = 1.21 * self.probing_coord * \
                    self.probing_coord
            elif field.mesh_dim() == 2:
                field.set(python_functions[1])
                expected_values = 1.21 * coords[:, 0] * coords[:, 0] - \
                    3.21 * coords[:, 1]
                expected_probed_value = (1.21 * self.probing_coord - 3.21) * \
                    self.probing_coord
            elif field.mesh_dim() == 3:
                field.set(python_functions[2])
                expected_values = 1.21 * coords[:, 0] * coords[:, 0] - \
                    3.21 * coords[:, 1] + 2.47 * coords[:, 2]
                expected_probed_value = (1.21 * self.probing_coord - 3.21 +
                                         2.47) * self.probing_coord

            # Check the result of coords_and_values (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values[:, 0] == expected_values)

            # Check the interpolated value outside the mesh node.
            # The expected field is nonlinear and, because of that,
            # greater tolerance value (tol1) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol2

    def test_set_nonlinear_vector_field(self):
        """Test setting the vector field with a nonlinear expression."""
        # Different nonlinear expressions for 2D vector fields.
        expressions = [df.Expression(['1.1*x[0]*x[0]', '-2.4*x[0]']),
                       df.Expression(['1.1*x[0]*x[0]', '-2.4*x[0]']),
                       df.Expression(['1.1*x[0]*x[0]', '-2.4*x[1]'])]

        # Test setting the vector field for different
        # vector function spaces and appropriate expressions.
        for functionspace in self.vector2d_fspaces:
            field = Field(functionspace)

            # Set the vector field and compute expected values.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(expressions[0])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 0])
            elif field.mesh_dim() == 2:
                field.set(expressions[1])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 0])
            elif field.mesh_dim() == 3:
                field.set(expressions[2])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 1])

            # Compute expected probed value.
            expected_probed_value = (1.1 * self.probing_coord * self.probing_coord,
                                     -2.4 * self.probing_coord)

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])

            # Check the interpolated value outside the mesh node.
            # The expected field is nonlinear and, because of that,
            # greater tolerance value (tol2) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol2
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol2

        # Different nonlinear expressions for 3D vector fields.
        expressions = [df.Expression(['1.1*x[0]*x[0]', '-2.4*x[0]', '3*x[0]']),
                       df.Expression(['1.1*x[0]*x[0]', '-2.4*x[1]', '3*x[1]']),
                       df.Expression(['1.1*x[0]*x[0]', '-2.4*x[1]', '3*x[2]'])]

        # Test setting the vector field for different
        # vector function spaces and appropriate expressions.
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

            # Set the vector field and compute expected values.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(expressions[0])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 0], 3 * coords[:, 0])
            elif field.mesh_dim() == 2:
                field.set(expressions[1])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 1], 3 * coords[:, 1])
            elif field.mesh_dim() == 3:
                field.set(expressions[2])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 1], 3 * coords[:, 2])

            # Compute expected probed value.
            expected_probed_value = (1.1 * self.probing_coord * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            assert np.all(f_array_split[2] == expected_values[2])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            assert np.all(field_values[:, 2] == expected_values[2])

            # Check the interpolated value outside the mesh node.
            # The expected field is nonlinear and, because of that,
            # greater tolerance value (tol2) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol2
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol2
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol2

        # Different nonlinear expressions for 4D vector fields.
        expressions = [df.Expression(['1.1*x[0]*x[0]', '-2.4*x[0]',
                                      '3*x[0]', 'x[0]']),
                       df.Expression(['1.1*x[0]*x[0]', '-2.4*x[1]',
                                      '3*x[1]', 'x[0]']),
                       df.Expression(['1.1*x[0]*x[0]', '-2.4*x[1]',
                                      '3*x[2]', 'x[0]'])]

        # Test setting the vector field for different
        # vector function spaces and appropriate expressions.
        for functionspace in self.vector4d_fspaces:
            field = Field(functionspace)

            # Set the vector field and compute expected values.
            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                field.set(expressions[0])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 0], 3 * coords[:, 0],
                                   coords[:, 0])
            elif field.mesh_dim() == 2:
                field.set(expressions[1])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 1], 3 * coords[:, 1],
                                   coords[:, 0])
            elif field.mesh_dim() == 3:
                field.set(expressions[2])
                expected_values = (1.1 * coords[:, 0] * coords[:, 0],
                                   -2.4 * coords[:, 1], 3 * coords[:, 2],
                                   coords[:, 0])

            # Compute expected probed value.
            expected_probed_value = (1.1 * self.probing_coord * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord,
                                     self.probing_coord)

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            assert np.all(f_array_split[2] == expected_values[2])
            assert np.all(f_array_split[3] == expected_values[3])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            assert np.all(field_values[:, 2] == expected_values[2])
            assert np.all(field_values[:, 3] == expected_values[3])

            # Check the interpolated value outside the mesh node.
            # The expected field is nonlinear and, because of that,
            # greater tolerance value (tol2) is used.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol2
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol2
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol2
            assert abs(probed_value[3] - expected_probed_value[3]) < self.tol2

    def test_plot_with_dolfin(self):
        """Test that we can call the plotting function of a Field object."""
        # Set environment variable DOLFIN_NOPLOT to a non-zero value in
        # order to suppress the actual plotting (because we have no way
        # to close the window non-interactively from within the test).
        os.environ['DOLFIN_NOPLOT'] = 'TRUE'

        field = Field(self.fs3d_vector3d, value=[1, 0, 0])
        field.plot_with_dolfin(interactive=False)

    def test_add_scalar_fields(self):
        for functionspace in self.scalar_fspaces:
            field1 = Field(functionspace, value=3.1)
            field2 = Field(functionspace, value=3.35)

            field3 = field1 + field2

            assert np.allclose(field3.f.vector().array(), 6.45)

    def test_add_vector_fields(self):
        for functionspace in self.vector3d_fspaces:
            field1 = Field(functionspace, value=(1, 2, 3))
            field2 = Field(functionspace, value=(5, 2.1, 6))

            field3 = field1 + field2

            coords = field3.coords_and_values()[0]
            for coord in coords:
                assert abs(field3.probe(coord)[0] - 6) < self.tol1
                assert abs(field3.probe(coord)[1] - 4.1) < self.tol1
                assert abs(field3.probe(coord)[2] - 9) < self.tol1
