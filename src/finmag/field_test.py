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

        # Create lists of meshes and functionspaces
        # to avoid the repetition of code in tests.
        self.meshes = [self.mesh1d,
                       self.mesh2d,
                       self.mesh3d]

        self.scalar_functionspaces = [self.fs_1d_scalar,
                                      self.fs_2d_scalar,
                                      self.fs_3d_scalar]

        self.vector_functionspaces = [self.fs_1d_vector2d,
                                      self.fs_2d_vector2d,
                                      self.fs_3d_vector2d,
                                      self.fs_1d_vector3d,
                                      self.fs_2d_vector3d,
                                      self.fs_3d_vector3d]

        self.all_functionspaces = self.scalar_functionspaces + \
            self.vector_functionspaces

        # Set the tolerance used throughout all tests
        # mainly due to interpolation errors.
        self.tol = 1e-13  # at the mesh node
        self.tol_outside_node = 1e-3  # outside the mesh node

    def test_init(self):
        # Initialisation arguments.
        functionspace = self.fs_3d_vector3d
        value = None  # Not specified, a zero-function is created.
        normalised = True
        name = 'name_test'
        unit = 'unit_test'

        field = Field(functionspace, value, normalised, name, unit)

        assert field.functionspace == functionspace

        # Assert that created field.f is an "empty" function.
        assert isinstance(field.f, df.Function)
        assert np.all(field.f.vector().array() == 0)

        assert field.normalised is True

        # Assert that both function's name and label are changed.
        assert field.name == name
        assert field.f.name() == name
        assert field.f.label() == name

        assert field.unit == unit

    def test_set_scalar_field_with_constant(self):
        # Different expressions for constant value 42.
        values = [df.Constant(42),
                  df.Constant(42.0),
                  df.Constant("42"),
                  df.Constant("42.0"),
                  42,
                  42.0,
                  "42",
                  "42.0",
                  u"42",
                  u"42.0"]

        # x, y, or z coordinate value for probing the field.
        probing_coord = 0.55  # Not at the mesh node.

        expected_value = 42

        # Test setting the scalar field value for
        # different scalar function spaces and constants.
        for functionspace in self.scalar_functionspaces:
            for value in values:
                field = Field(functionspace, value)

                # Check vector (numpy array) values (should be exact).
                assert np.all(field.f.vector().array() == expected_value)

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values == expected_value)

                # Check values that are interpolated,
                # dolfin is fairly inaccurate here, see field_test.ipynb.
                probing_point = field.mesh_dim() * (probing_coord,)
                probed_value = field.probe_field(probing_point)
                assert abs(probed_value - expected_value) < self.tol

    def test_set_scalar_field_with_expression(self):
        # Different expressions for scalar field value setting,
        # depending on the mesh dimension (1d, 2d, or 3d).
        expressions = [df.Expression("11.2*x[0]"),
                       df.Expression("11.2*x[0] - 3.1*x[1]"),
                       df.Expression("11.2*x[0] - 3.1*x[1] + 2.7*x[2]")]

        # x, y, or z coordinate value for probing the field.
        probing_coord = 0.55  # Not at the mesh node.

        # Test setting the scalar field value for
        # different scalar function spaces and expressions.
        for i in range(len(self.scalar_functionspaces)):
            field = Field(self.scalar_functionspaces[i], expressions[i])

            # Compute expected values.
            coords = self.scalar_functionspaces[i].mesh().coordinates()
            if i == 0:
                # Compute expected values at all mesh nodes.
                expected_values = 11.2*coords[:, 0]
                # Compute expected probed value.
                expected_probed_value = 11.2*probing_coord
            elif i == 1:
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1]
                expected_probed_value = (11.2 - 3.1)*probing_coord
            elif i == 2:
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1] + \
                    2.7*coords[:, 2]
                expected_probed_value = (11.2 - 3.1 + 2.7)*probing_coord

            # Check the field value at all nodes (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values[:, 0] == expected_values)

            # Check the probed field value (not exact - interpolation).
            probing_point = field.mesh_dim() * (probing_coord,)
            probed_value = field.probe_field(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol

    def test_set_scalar_field_with_python_function(self):
        # Python functions for setting the scalar field value.
        def python_fun1d(x):
            return 11.2*x[0]

        def python_fun2d(x):
            return 11.2*x[0] - 3.1*x[1]

        def python_fun3d(x):
            return 11.2*x[0] - 3.1*x[1] + 2.7*x[2]

        python_functions = [python_fun1d, python_fun2d, python_fun3d]

        # x, y, or z coordinate value for probing the field.
        probing_coord = 0.55  # Not at the mesh node.

        # Test setting the scalar field value for
        # different scalar function spaces and python functions.
        for i in range(len(self.scalar_functionspaces)):
            field = Field(self.scalar_functionspaces[i], python_functions[i])

            # Compute expected values.
            coords = self.scalar_functionspaces[i].mesh().coordinates()
            if i == 0:
                # Compute expected values at all mesh nodes.
                expected_values = 11.2*coords[:, 0]
                # Compute expected probed value.
                expected_probed_value = 11.2*probing_coord
            elif i == 1:
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1]
                expected_probed_value = (11.2 - 3.1)*probing_coord
            elif i == 2:
                expected_values = 11.2*coords[:, 0] - 3.1*coords[:, 1] + \
                    2.7*coords[:, 2]
                expected_probed_value = (11.2 - 3.1 + 2.7)*probing_coord

            # Check the field value at all nodes (should be exact).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            assert np.all(field_values[:, 0] == expected_values)

            # Check the probed field value (not exact - interpolation).
            probing_point = field.mesh_dim() * (probing_coord,)
            probed_value = field.probe_field(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol

    def test_set_vector_field_with_constant(self):
        # Different constant expressions for 2d vector fields.
        constants2d = [df.Constant((0.15, -2.3)),
                       df.Constant([0.15, -2.3]),
                       df.Constant(np.array([0.15, -2.3])),
                       (0.15, -2.3),
                       [0.15, -2.3],
                       np.array([0.15, -2.3])]

        # Different constant expressions for 3d vector fields.
        constants3d = [df.Constant((0.15, -2.3, -6.41)),
                       df.Constant([0.15, -2.3, -6.41]),
                       df.Constant(np.array([0.15, -2.3, -6.41])),
                       (0.15, -2.3, -6.41),
                       [0.15, -2.3, -6.41],
                       np.array([0.15, -2.3, -6.41])]

        # x, y, or z coordinate value for probing the field.
        probing_coord = 0.55  # Not at the mesh node.

        expected_value = (0.15, -2.3, -6.41)

        # Test setting the vector field value for
        # different vector function spaces and constants.
        for functionspace in self.vector_functionspaces:
            # Choose an appropriate set of constants
            if functionspace.ufl_element().value_shape()[0] == 2:
                # 2d vector
                constants = constants2d
            elif functionspace.ufl_element().value_shape()[0] == 3:
                # 3d vector
                constants = constants3d

            for constant in constants:
                field = Field(functionspace, constant)

                # Check vector (numpy array) values (should be exact).
                f_array = field.f.vector().array()
                f_array_split = np.split(f_array, field.value_dim())
                assert np.all(f_array_split[0] == expected_value[0])
                assert np.all(f_array_split[1] == expected_value[1])
                if field.value_dim() == 3:
                    # only for 3d vectors
                    assert np.all(f_array_split[2] == expected_value[2])

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])
                if field.value_dim() == 3:
                    # only for 3d vectors
                    assert np.all(field_values[:, 2] == expected_value[2])

                # Check values that are interpolated,
                # dolfin is fairly inaccurate here, see field_test.ipynb.
                probing_point = field.mesh_dim() * (probing_coord,)
                probed_value = field.probe_field(probing_point)
                assert abs(probed_value[0] - expected_value[0]) < self.tol
                assert abs(probed_value[1] - expected_value[1]) < self.tol
                if field.value_dim() == 3:
                    # only for 3d vectors
                    assert abs(probed_value[2] - expected_value[2]) < self.tol

    def test_set_vector_field_with_expression(self):
        # Different expressions for 2d and 3d vector fields.
        expressions = [df.Expression(['1.1*x[0]', '-2.4*x[0]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[2]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[0]', '3*x[0]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[1]']),
                       df.Expression(['1.1*x[0]', '-2.4*x[1]', '3*x[2]'])]

        # x, y, or z coordinate value for probing the field.
        probing_coord = 0.55  # Not at the mesh node.

        # Test setting the vector field value for
        # different vector function spaces and expressions.
        for i in xrange(len(self.vector_functionspaces)):
            functionspace = self.vector_functionspaces[i]
            coords = functionspace.mesh().coordinates()
            mesh_dim = functionspace.mesh().topology().dim()
            value_dim = functionspace.ufl_element().value_shape()[0]

            # Compute expected values.
            if value_dim == 2:
                # 2d vector
                if mesh_dim == 1:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 0])
                elif mesh_dim == 2:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 1])
                elif mesh_dim == 3:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 2])

            elif value_dim == 3:
                # 3d vector
                if mesh_dim == 1:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 0],
                                       3*coords[:, 0])
                elif mesh_dim == 2:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 1],
                                       3*coords[:, 1])
                elif mesh_dim == 3:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 1],
                                       3*coords[:, 2])

            # Compute expected probed value
            expected_probed_value = (1.1*probing_coord,
                                     -2.4*probing_coord,
                                     3*probing_coord)

            field = Field(functionspace, expressions[i])

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            if field.value_dim() == 3:
                # only for 3d vectors
                assert np.all(f_array_split[2] == expected_values[2])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            if field.value_dim() == 3:
                # only for 3d vectors
                assert np.all(field_values[:, 2] == expected_values[2])

            # Check values that are interpolated,
            # dolfin is fairly inaccurate here, see field_test.ipynb.
            probing_point = field.mesh_dim() * (probing_coord,)
            probed_value = field.probe_field(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol
            if field.value_dim() == 3:
                # only for 3d vectors
                assert abs(probed_value[2] -
                           expected_probed_value[2]) < self.tol

    def test_set_vector_field_with_python_function(self):
        # Different python functions for setting the vector field values.
        def python_fun_1d_vector2d(x):
            return (1.1*x[0], -2.4*x[0])

        def python_fun_2d_vector2d(x):
            return (1.1*x[0], -2.4*x[1])

        def python_fun_3d_vector2d(x):
            return (1.1*x[0], -2.4*x[2])

        def python_fun_1d_vector3d(x):
            return (1.1*x[0], -2.4*x[0], 3*x[0])

        def python_fun_2d_vector3d(x):
            return (1.1*x[0], -2.4*x[1], 3*x[1])

        def python_fun_3d_vector3d(x):
            return (1.1*x[0], -2.4*x[1], 3*x[2])

        python_functions = [python_fun_1d_vector2d,
                            python_fun_2d_vector2d,
                            python_fun_3d_vector2d,
                            python_fun_1d_vector3d,
                            python_fun_2d_vector3d,
                            python_fun_3d_vector3d]

        # x, y, or z coordinate value for probing the field.
        probing_coord = 0.55  # Not at the mesh node.

        # Test setting the vector field value for
        # different vector function spaces and python functions.
        for i in xrange(len(self.vector_functionspaces)):
            functionspace = self.vector_functionspaces[i]
            coords = functionspace.mesh().coordinates()
            mesh_dim = functionspace.mesh().topology().dim()
            value_dim = functionspace.ufl_element().value_shape()[0]

            # Compute expected values.
            if value_dim == 2:
                # 2d vector
                if mesh_dim == 1:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 0])
                elif mesh_dim == 2:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 1])
                elif mesh_dim == 3:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 2])

            elif value_dim == 3:
                # 3d vector
                if mesh_dim == 1:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 0],
                                       3*coords[:, 0])
                elif mesh_dim == 2:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 1],
                                       3*coords[:, 1])
                elif mesh_dim == 3:
                    expected_values = (1.1*coords[:, 0],
                                       -2.4*coords[:, 1],
                                       3*coords[:, 2])

            # Compute expected probed value
            expected_probed_value = (1.1*probing_coord,
                                     -2.4*probing_coord,
                                     3*probing_coord)

            field = Field(functionspace, python_functions[i])

            # Check vector (numpy array) values (should be exact).
            f_array = field.f.vector().array()
            f_array_split = np.split(f_array, field.value_dim())
            assert np.all(f_array_split[0] == expected_values[0])
            assert np.all(f_array_split[1] == expected_values[1])
            if field.value_dim() == 3:
                # only for 3d vectors
                assert np.all(f_array_split[2] == expected_values[2])

            # Check the result of coords_and_values (should be exact).
            coords, field_values = field.coords_and_values()
            assert np.all(field_values[:, 0] == expected_values[0])
            assert np.all(field_values[:, 1] == expected_values[1])
            if field.value_dim() == 3:
                # only for 3d vectors
                assert np.all(field_values[:, 2] == expected_values[2])

            # Check values that are interpolated,
            # dolfin is fairly inaccurate here, see field_test.ipynb.
            probing_point = field.mesh_dim() * (probing_coord,)
            probed_value = field.probe_field(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol
            if field.value_dim() == 3:
                # only for 3d vectors
                assert abs(probed_value[2] -
                           expected_probed_value[2]) < self.tol

    def test_coords_and_values_scalar_field(self):
        # Test for scalar fields on 1d, 2d, and 3d meshes,
        # initialised with a dolfin expression.
        expression = df.Expression('1.3*x[0]')

        for functionspace in self.scalar_functionspaces:
            expected_coords = functionspace.mesh().coordinates()
            num_nodes = functionspace.mesh().num_vertices()
            expected_values = 1.3*expected_coords[:, 0]

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

    def test_normalise(self):
        # 3d vector field
        functionspace = self.fs_3d_vector3d
        field = Field(functionspace, (1, 2, 3))
        field.normalise()
        coords, values = field.coords_and_values()
        norm = values[:, 0]**2 + values[:, 1]**2 + values[:,2]**2
        assert np.all(abs(norm - 1) < 1e-5)

        # 2d vector field
        functionspace = self.fs_2d_vector2d
        field = Field(functionspace, (1, 3))
        field.normalise()
        coords, values = field.coords_and_values()
        norm = values[:, 0]**2 + values[:, 1.01]**2
        assert np.all(abs(norm - 1) < 1e-5)

    def test_coords_and_values_vector_field(self):
        # Different expressions for 2d and 3d vector fields.
        expression2d = df.Expression(['1.3*x[0]', '2.3*x[0]'])
        expression3d = df.Expression(['1.3*x[0]', '2.3*x[0]', '-1*x[0]'])

        for functionspace in self.vector_functionspaces:
            # Initialise the field with an appropriate expression for
            # the function space and compute expected results.
            expected_coords = functionspace.mesh().coordinates()
            num_nodes = functionspace.mesh().num_vertices()
            value_dim = functionspace.ufl_element().value_shape()[0]

            if value_dim == 2:
                expression = expression2d
                expected_values = (1.3*expected_coords[:, 0],
                                   2.3*expected_coords[:, 0])
            elif value_dim == 3:
                expression = expression3d
                expected_values = (1.3*expected_coords[:, 0],
                                   2.3*expected_coords[:, 0],
                                   -1*expected_coords[:, 0])

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
            if field.value_dim() == 3:
                assert np.all(values[:, 2] == expected_values[2])

    def test_probe_field_scalar_field(self):
        for functionspace in self.scalar_functionspaces:
            field = Field(functionspace)
            mesh_dim = field.mesh_dim()

            if mesh_dim == 1:
                field.set(df.Expression('15.3*x[0]'))
                exact_result_at_node = 15.3*0.5
                exact_result_outside_node = 15.3*0.55
            elif mesh_dim == 2:
                field.set(df.Expression('15.3*x[0] - 2.3*x[1]'))
                exact_result_at_node = 15.3*0.5 - 2.3*0.5
                exact_result_outside_node = 15.3*0.55 - 2.3*0.55
            elif mesh_dim == 3:
                field.set(df.Expression('15.3*x[0] - 2.3*x[1] + 96.1*x[2]'))
                exact_result_at_node = 15.3*0.5 - 2.3*0.5 + 96.1*0.5
                exact_result_outside_node = 15.3*0.55 - 2.3*0.55 + 96.1*0.55

            # Probe and check the result at the mesh node.
            probe_point = mesh_dim * (0.5,)
            probed_value = field.probe_field(probe_point)
            assert abs(probed_value - exact_result_at_node) < self.tol

            # Probe and check the result outside the mesh node.
            probe_point = mesh_dim * (0.55,)
            probed_value = field.probe_field(probe_point)
            assert abs(probed_value - exact_result_outside_node) < self.tol

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
                          self.fs_1d_vector2d,
                          self.fs_1d_vector3d,
                          self.fs_2d_scalar,
                          self.fs_2d_vector2d,
                          self.fs_2d_vector3d,
                          self.fs_3d_scalar,
                          self.fs_3d_vector2d,
                          self.fs_3d_vector3d]

        value_dim = []
        for functionspace in functionspaces:
            field = Field(functionspace)
            value_dim.append(field.value_dim())

        expected_result = 3*[1, 2, 3]
        assert value_dim == expected_result
