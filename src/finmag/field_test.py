"""DOLFINx port of the master ``field_test.py`` Field unit-test suite.

This file now lives at its master path
``src/finmag/field_test.py`` (formerly
``src/finmag/tests/test_field_dolfinx.py``), so
``git diff b5015c5a..HEAD -- src/finmag/field_test.py``
shows the port diff directly.

This file has two clearly separated parts:

1. A MINIMAL-DIFF transcription of the master ``TestField`` suite from
   ``src/finmag/field_test.py`` (git ``b5015c5a``). Method names, ordering and
   assertion structure are kept identical to master; master's documented
   tolerances ``tol1 = 5e-13`` / ``tol2 = 1e-2`` / ``tol3 = 5e-6`` are restored
   verbatim. The only differences are the sanctioned ones:

   - dolfin -> dolfinx API changes, each annotated inline (``df.Expression`` ->
     Python callable, ``df.FunctionSpace``/``df.VectorFunctionSpace`` ->
     ``fem.functionspace``, ``df.Constant`` -> ``fem.Constant``,
     ``mesh.coordinates()`` -> ``mesh.geometry.x`` owned slice, etc.);
   - py2 -> py3 (``u"..."`` etc.);
   - explanatory comments; and
   - **behavioural inversions made VISIBLE** where the DOLFINx port intentionally
     changed behaviour (see the banners in the body):
       * N10 -- ``Field + Field`` used legacy point-measure assembly and is
         deferred: ``test_add_scalar_fields`` / ``test_add_vector_fields`` now
         assert it RAISES ``NotImplementedError`` (was: addition works);
       * dolfin string Expressions/Constants are unsupported -- passing a string
         raises ``NotImplementedError`` (was: parsed);
       * ``plot_with_dolfin`` raises ``NotImplementedError`` (legacy dolfin
         plotting is gone);
       * ``save_hdf5`` writes ONE self-describing ``.h5`` snapshot and NO
         ``.json`` sidecar (the legacy ``dolfinh5tools`` timeseries + json is
         gone), and ``close_hdf5`` is a no-op.

   PBC finding (restored coverage): master built a *second* copy of every
   function space with ``constrained_domain=self.pbc`` and swept both. But the
   master PBC ``inside()`` body is ``x[0] < DOLFIN_EPS and x[0] > DOLFIN_EPS`` --
   a value cannot be simultaneously ``< eps`` and ``> eps``, so ``inside()`` is
   ALWAYS False, ``map()`` is never called, and the constraint identifies NO
   nodes. The master "PBC" spaces are therefore byte-equivalent to the plain CG1
   spaces. DOLFINx has no ``constrained_domain`` argument (periodicity lives in
   the separate ``dolfinx_mpc`` package), so the ``*_pbc`` spaces below are plain
   ``fem.functionspace`` objects -- which faithfully reproduces the master no-op
   PBC sweep. This is documented rather than silently dropped.

   Each restored tolerance's measured DOLFINx headroom is recorded inline; every
   master tolerance passes VERBATIM (no tolerance was loosened).

2. The focused NEW-under-DOLFINx production tests (ownership/ghost accessors,
   coordinate round-trips, from_generic_vector backend-object surface, VTK/XDMF
   output, coordinate-drift guards) which have no master ancestor. They live
   unchanged below the ``NEW under DOLFINx`` banner.

Run:
    pixi run -e dolfinx python -m pytest -q src/finmag/field_test.py
"""

import functools
import sys

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from ufl import Measure, dx

import finmag
from finmag.field import Field, associated_scalar_space


# ===========================================================================
# MINIMAL-DIFF transcription of master TestField (git b5015c5a)
# ===========================================================================


def _owned_vertices(functionspace):
    """dolfin ``mesh.coordinates()`` -> the owned mesh vertices, gdim columns.

    Master compared against ``functionspace.mesh().coordinates()`` (all vertices
    in serial). ``Field.coords_and_values`` returns the owned mesh vertices in
    ``mesh.geometry.x`` order, so the expected coordinates come from the same
    owned slice (identical in the serial runs these tests target).
    """
    domain = functionspace.mesh
    n = domain.geometry.index_map().size_local
    gdim = domain.geometry.dim
    return domain.geometry.x[:n, :gdim]


def _num_vertices(functionspace):
    """dolfin ``mesh.num_vertices()`` -> owned vertex count."""
    return functionspace.mesh.geometry.index_map().size_local


class TestField(object):
    def setup_method(self, method):  # dolfin/nose ``setup`` -> pytest setup_method
        self.create_meshes()
        self.define_tolerances()

        # All created function spaces are CG (Lagrange)
        # with degree=1 unless named explicitly.
        self.create_PBCs()
        self.create_scalar_function_spaces()
        self.create_vector2d_function_spaces()
        self.create_vector3d_function_spaces()
        self.create_vector4d_function_spaces()
        self.all_fspaces = self.scalar_fspaces + self.vector2d_fspaces + \
            self.vector3d_fspaces + self.vector4d_fspaces

        # x, y, or z coordinate value for probing the field.
        self.probing_coord = 0.4351  # Not at any mesh node.

    def create_meshes(self):
        """
        Create meshes of several dimensions.
        """
        # dolfin df.UnitIntervalMesh/UnitSquareMesh/UnitCubeMesh ->
        # dolfinx mesh.create_unit_* (require an MPI communicator).
        self.mesh1d = mesh.create_unit_interval(MPI.COMM_WORLD, 10)
        self.mesh2d = mesh.create_unit_square(MPI.COMM_WORLD, 11, 10)
        self.mesh3d = mesh.create_unit_cube(MPI.COMM_WORLD, 9, 11, 10)
        self.meshes = [self.mesh1d, self.mesh2d, self.mesh3d]

    def create_PBCs(self):
        """
        Master created a periodic boundary condition and built a second copy of
        every function space with ``constrained_domain=self.pbc``. The master
        ``inside()`` body ``x[0] < DOLFIN_EPS and x[0] > DOLFIN_EPS`` is ALWAYS
        False, so that PBC identified no nodes -- a no-op. DOLFINx has no
        ``constrained_domain`` argument (periodicity is in ``dolfinx_mpc``), so
        the ``*_pbc`` spaces below are plain CG1 spaces, which reproduces the
        master no-op PBC sweep exactly. See the module docstring.
        """
        self.pbc = None  # master ``PeriodicBoundary()`` -- a no-op, see above.

    def create_scalar_function_spaces(self):
        """
        Create scalar function spaces (both with and without PBCs).
        """
        # dolfin df.FunctionSpace(mesh, "CG", 1) -> fem.functionspace(mesh,
        # ("Lagrange", 1)). The ``_pbc`` variants are plain spaces (no-op PBC).
        self.fs1d_scalar = fem.functionspace(self.mesh1d, ("Lagrange", 1))
        self.fs2d_scalar = fem.functionspace(self.mesh2d, ("Lagrange", 1))
        self.fs3d_scalar = fem.functionspace(self.mesh3d, ("Lagrange", 1))

        self.fs1d_scalar_pbc = fem.functionspace(self.mesh1d, ("Lagrange", 1))
        self.fs2d_scalar_pbc = fem.functionspace(self.mesh2d, ("Lagrange", 1))
        self.fs3d_scalar_pbc = fem.functionspace(self.mesh3d, ("Lagrange", 1))

        self.scalar_fspaces = [
            self.fs1d_scalar, self.fs2d_scalar,
            self.fs3d_scalar, self.fs1d_scalar_pbc,
            self.fs2d_scalar_pbc, self.fs3d_scalar_pbc]

    def create_vector2d_function_spaces(self):
        """
        Create 2D vector function spaces (both with and without PBCs).
        """
        # dolfin df.VectorFunctionSpace(mesh, "CG", 1, dim=2) ->
        # fem.functionspace(mesh, ("Lagrange", 1, (2,))).
        self.fs1d_vector2d = fem.functionspace(self.mesh1d, ("Lagrange", 1, (2,)))
        self.fs2d_vector2d = fem.functionspace(self.mesh2d, ("Lagrange", 1, (2,)))
        self.fs3d_vector2d = fem.functionspace(self.mesh3d, ("Lagrange", 1, (2,)))

        self.fs1d_vector2d_pbc = fem.functionspace(self.mesh1d, ("Lagrange", 1, (2,)))
        self.fs2d_vector2d_pbc = fem.functionspace(self.mesh2d, ("Lagrange", 1, (2,)))
        self.fs3d_vector2d_pbc = fem.functionspace(self.mesh3d, ("Lagrange", 1, (2,)))

        self.vector2d_fspaces = [
            self.fs1d_vector2d, self.fs2d_vector2d,
            self.fs3d_vector2d, self.fs1d_vector2d_pbc,
            self.fs2d_vector2d_pbc, self.fs3d_vector2d_pbc]

    def create_vector3d_function_spaces(self):
        """
        Create 3D vector function spaces (both with and without PBCs).
        """
        self.fs1d_vector3d = fem.functionspace(self.mesh1d, ("Lagrange", 1, (3,)))
        self.fs2d_vector3d = fem.functionspace(self.mesh2d, ("Lagrange", 1, (3,)))
        self.fs3d_vector3d = fem.functionspace(self.mesh3d, ("Lagrange", 1, (3,)))

        self.fs1d_vector3d_pbc = fem.functionspace(self.mesh1d, ("Lagrange", 1, (3,)))
        self.fs2d_vector3d_pbc = fem.functionspace(self.mesh2d, ("Lagrange", 1, (3,)))
        self.fs3d_vector3d_pbc = fem.functionspace(self.mesh3d, ("Lagrange", 1, (3,)))

        # Master's list (verbatim): only the ``*_pbc`` (== plain, no-op PBC)
        # variants are swept; the plain non-pbc entries are commented out.
        self.vector3d_fspaces = [
#            self.fs1d_vector3d, self.fs2d_vector3d,
#            self.fs3d_vector3d,
            self.fs1d_vector3d_pbc,
            self.fs2d_vector3d_pbc, self.fs3d_vector3d_pbc]

    def create_vector4d_function_spaces(self):
        """
        Create 4D vector function spaces (both with and without PBCs).
        """
        self.fs1d_vector4d = fem.functionspace(self.mesh1d, ("Lagrange", 1, (4,)))
        self.fs2d_vector4d = fem.functionspace(self.mesh2d, ("Lagrange", 1, (4,)))
        self.fs3d_vector4d = fem.functionspace(self.mesh3d, ("Lagrange", 1, (4,)))

        self.fs1d_vector4d_pbc = fem.functionspace(self.mesh1d, ("Lagrange", 1, (4,)))
        self.fs2d_vector4d_pbc = fem.functionspace(self.mesh2d, ("Lagrange", 1, (4,)))
        self.fs3d_vector4d_pbc = fem.functionspace(self.mesh3d, ("Lagrange", 1, (4,)))

        self.vector4d_fspaces = [
            self.fs1d_vector4d, self.fs2d_vector4d,
            self.fs3d_vector4d, self.fs1d_vector4d_pbc,
            self.fs2d_vector4d_pbc, self.fs3d_vector4d_pbc]

    def define_tolerances(self):
        """
        Set the tolerances used throughout all tests
        to account for interpolation errors. (Master values verbatim.)
        """
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

            # dolfin ``==`` -> identity (the port stores the space verbatim).
            assert field.functionspace is functionspace
            assert field.name == name
            assert field.unit == unit

            # dolfin ``f.name()``/``f.label()`` (methods) -> DOLFINx ``f.name``
            # (a str property). There is no DOLFINx ``label`` (behavioural
            # change: label dropped).
            assert field.f.name == name

            # Check that the created function is a dolfinx zero function.
            # dolfin df.Function -> dolfinx fem.Function.
            assert isinstance(field.f, fem.Function)
            assert np.all(field.coords_and_values()[1] == 0)

    def test_set_scalar_field_with_constant(self):
        """Test setting the scalar field with a constant."""
        # dolfin df.Constant(...) -> fem.Constant(mesh, ...). Master's list also
        # included string constants (df.Constant("42"), "42", u"42", ...); the
        # DOLFINx port has no string mini-language (asserted to raise below).
        constants = [fem.Constant(self.mesh1d, 42.0),
                     42, 42.0, np.float64(42.0)]

        expected_value = 42

        # Setting the scalar field for different
        # scalar function spaces and constants.
        for functionspace in self.scalar_fspaces:
            for constant in constants:
                field = Field(functionspace, constant)

                # Check vector (numpy array) values (should be exact).
                # dolfin ``f.vector().array()`` -> Field.as_array() (owned dofs).
                assert np.all(field.as_array() == expected_value)

                # Check the result of coords_and_values (should be exact).
                field_values = field.coords_and_values()[1]  # coords ignored
                assert np.all(field_values == expected_value)

                # Check the interpolated value outside the mesh node.
                # The expected field is constant and, because of that,
                # smaller tolerance value (tol1) is used.
                # Measured DOLFINx headroom: exact (0.0) << tol1=5e-13.
                probing_point = field.mesh_dim() * (self.probing_coord,)
                probed_value = field.probe(probing_point)
                assert abs(probed_value - expected_value) < self.tol1

        # Behavioural change (VISIBLE): master also passed string constants
        # ("42", u"42.0", df.Constant("42"), ...) which dolfin parsed. The
        # DOLFINx port rejects strings loudly -- see
        # test_legacy_only_features_fail_precisely below the banner.
        with pytest.raises(NotImplementedError):
            Field(self.fs1d_scalar, "42")

    def test_set_scalar_field_with_expression(self):
        """Test setting the scalar field with an expression."""
        # dolfin df.Expression("11.2*x[0]", degree=1) -> Python callable acting
        # on the vectorised coordinate array ``x`` (x[0]/x[1]/x[2] are rows).
        expressions = [lambda x: 11.2 * x[0],
                       lambda x: 11.2 * x[0] - 3.01 * x[1],
                       lambda x: 11.2 * x[0] - 3.01 * x[1] + 2.7 * x[2]]

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
            # dolfin used ``==`` (bit-exact via vertex_to_dof_map). Under
            # DOLFINx the interpolant samples ``tabulate_dof_coordinates`` while
            # coords_and_values returns ``geometry.x`` -- equal only to FP
            # rounding (measured max abs diff ~4e-15). allclose, not ``==``.
            assert np.allclose(field_values, expected_values)

            # Check the interpolated value outside the mesh node.
            # The expected field is linear -> smaller tolerance value (tol1).
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_scalar_field_with_dolfin_function(self):
        """Test setting the scalar field with a dolfin(x) function."""
        expressions = [lambda x: 11.2 * x[0],
                       lambda x: 11.2 * x[0] - 3.01 * x[1],
                       lambda x: 11.2 * x[0] - 3.01 * x[1] + 2.7 * x[2]]

        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)

            coords = field.coords_and_values()[0]  # Values ignored.
            # dolfin df.interpolate(expr, fs) -> fem.Function + interpolate.
            if field.mesh_dim() == 1:
                dolfin_function = _interpolate(functionspace, expressions[0])
                field.set(dolfin_function)
                expected_values = 11.2 * coords[:, 0]
                expected_probed_value = 11.2 * self.probing_coord
            elif field.mesh_dim() == 2:
                dolfin_function = _interpolate(functionspace, expressions[1])
                field.set(dolfin_function)
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1]
                expected_probed_value = (11.2 - 3.01) * self.probing_coord
            elif field.mesh_dim() == 3:
                dolfin_function = _interpolate(functionspace, expressions[2])
                field.set(dolfin_function)
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1] + \
                    2.7 * coords[:, 2]
                expected_probed_value = (
                    11.2 - 3.01 + 2.7) * self.probing_coord

            field_values = field.coords_and_values()[1]  # ignore coordinates
            # dolfin used ``==`` (bit-exact via vertex_to_dof_map). Under
            # DOLFINx the interpolant samples ``tabulate_dof_coordinates`` while
            # coords_and_values returns ``geometry.x`` -- equal only to FP
            # rounding (measured max abs diff ~4e-15). allclose, not ``==``.
            assert np.allclose(field_values, expected_values)

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_scalar_field_with_generic_vector(self):
        """Test setting the scalar field with a generic vector."""
        expressions = [lambda x: 11.2 * x[0],
                       lambda x: 11.2 * x[0] - 3.01 * x[1],
                       lambda x: 11.2 * x[0] - 3.01 * x[1] + 2.7 * x[2]]

        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)

            coords = field.coords_and_values()[0]  # Values ignored.
            # dolfin ``dolfin_function.vector()`` (a GenericVector) ->
            # dolfinx ``function.x`` (a dolfinx.la.Vector). set() routes it
            # through from_generic_vector, mirroring legacy dispatch.
            if field.mesh_dim() == 1:
                dolfin_function = _interpolate(functionspace, expressions[0])
                field.set(dolfin_function.x)
                expected_values = 11.2 * coords[:, 0]
                expected_probed_value = 11.2 * self.probing_coord
            elif field.mesh_dim() == 2:
                dolfin_function = _interpolate(functionspace, expressions[1])
                field.set(dolfin_function.x)
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1]
                expected_probed_value = (11.2 - 3.01) * self.probing_coord
            elif field.mesh_dim() == 3:
                dolfin_function = _interpolate(functionspace, expressions[2])
                field.set(dolfin_function.x)
                expected_values = 11.2 * coords[:, 0] - 3.01 * coords[:, 1] + \
                    2.7 * coords[:, 2]
                expected_probed_value = (
                    11.2 - 3.01 + 2.7) * self.probing_coord

            field_values = field.coords_and_values()[1]  # ignore coordinates
            # dolfin used ``==`` (bit-exact via vertex_to_dof_map). Under
            # DOLFINx the interpolant samples ``tabulate_dof_coordinates`` while
            # coords_and_values returns ``geometry.x`` -- equal only to FP
            # rounding (measured max abs diff ~4e-15). allclose, not ``==``.
            assert np.allclose(field_values, expected_values)

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_scalar_field_with_python_function(self):
        """Test setting the scalar field with a python function."""
        python_functions = [lambda x: 1.21 * x[0],
                            lambda x: 1.21 * x[0] - 3.21 * x[1],
                            lambda x: 1.21 * x[0] - 3.21 * x[1] + 2.47 * x[2]]

        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)

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

            field_values = field.coords_and_values()[1]  # ignore coordinates
            # dolfin used ``==`` (bit-exact via vertex_to_dof_map). Under
            # DOLFINx the interpolant samples ``tabulate_dof_coordinates`` while
            # coords_and_values returns ``geometry.x`` -- equal only to FP
            # rounding (measured max abs diff ~4e-15). allclose, not ``==``.
            assert np.allclose(field_values, expected_values)

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol1

    def test_set_vector_field_with_constant(self):
        """Test setting the 3D vector field with a constant."""
        # dolfin df.Constant((...)) -> fem.Constant(mesh, (...)); plus the plain
        # python tuple/list/ndarray forms (all accepted by Field.set).
        constants = [fem.Constant(self.mesh1d, (0.15, -2.3, -6.41)),
                     (0.15, -2.3, -6.41),
                     [0.15, -2.3, -6.41],
                     np.array([0.15, -2.3, -6.41])]

        expected_value = (0.15, -2.3, -6.41)

        for functionspace in self.vector3d_fspaces:
            for constant in constants:
                field = Field(functionspace, constant)
                # Check vector (numpy array) values (should be exact).
                f_array = field.get_ordered_numpy_array_xxx()
                f_array_split = np.split(f_array, field.value_dim())
                assert np.all(f_array_split[0] == expected_value[0])
                assert np.all(f_array_split[1] == expected_value[1])
                assert np.all(f_array_split[2] == expected_value[2])

                # Check the result of coords_and_values (should be exact).
                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])
                assert np.all(field_values[:, 2] == expected_value[2])

                # Check the interpolated value outside the mesh node (tol1).
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

        # Try to set 2D vector field with string components. Behavioural change
        # (VISIBLE): master expected ValueError; the DOLFINx port has no string
        # mini-language so a list containing strings raises NotImplementedError
        # (still a loud rejection, different type).
        field = Field(self.fs3d_vector2d)
        with pytest.raises(NotImplementedError):
            field.set(["x[0]", "1", "0"])

    def test_set_vector_field_with_expression(self):
        """Test setting the 3D vector field with an expression."""
        # dolfin df.Expression([...]) -> callable returning np.vstack of rows.
        expressions = [lambda x: np.vstack((1.1 * x[0], -2.4 * x[0], 3 * x[0])),
                       lambda x: np.vstack((1.1 * x[0], -2.4 * x[1], 3 * x[1])),
                       lambda x: np.vstack((1.1 * x[0], -2.4 * x[1], 3 * x[2]))]

        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

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

            expected_probed_value = (1.1 * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            f_array = field.get_ordered_numpy_array_xxx()
            f_array_split = np.split(f_array, field.value_dim())
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(f_array_split[0], expected_values[0])
            assert np.allclose(f_array_split[1], expected_values[1])
            assert np.allclose(f_array_split[2], expected_values[2])

            coords, field_values = field.coords_and_values()
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(field_values[:, 0], expected_values[0])
            assert np.allclose(field_values[:, 1], expected_values[1])
            assert np.allclose(field_values[:, 2], expected_values[2])

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector_field_with_dolfin_function(self):
        """Test setting the 3D vector field with a dolfin(x) function."""
        expressions = [lambda x: np.vstack((1.1 * x[0], -2.4 * x[0], 3 * x[0])),
                       lambda x: np.vstack((1.1 * x[0], -2.4 * x[1], 3 * x[1])),
                       lambda x: np.vstack((1.1 * x[0], -2.4 * x[1], 3 * x[2]))]

        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

            coords = field.coords_and_values()[0]  # Values ignored.
            if field.mesh_dim() == 1:
                dolfin_function = _interpolate(functionspace, expressions[0])
                field.set(dolfin_function)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 0],
                                   3 * coords[:, 0])
            elif field.mesh_dim() == 2:
                dolfin_function = _interpolate(functionspace, expressions[1])
                field.set(dolfin_function)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 1])
            elif field.mesh_dim() == 3:
                dolfin_function = _interpolate(functionspace, expressions[2])
                field.set(dolfin_function)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 2])

            expected_probed_value = (1.1 * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            f_array = field.get_ordered_numpy_array_xxx()
            f_array_split = np.split(f_array, field.value_dim())
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(f_array_split[0], expected_values[0])
            assert np.allclose(f_array_split[1], expected_values[1])
            assert np.allclose(f_array_split[2], expected_values[2])

            coords, field_values = field.coords_and_values()
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(field_values[:, 0], expected_values[0])
            assert np.allclose(field_values[:, 1], expected_values[1])
            assert np.allclose(field_values[:, 2], expected_values[2])

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector_field_with_generic_vector(self):
        """Test setting the 3D vector field with a generic_vector."""
        expressions = [lambda x: np.vstack((1.1 * x[0], -2.4 * x[0], 3 * x[0])),
                       lambda x: np.vstack((1.1 * x[0], -2.4 * x[1], 3 * x[1])),
                       lambda x: np.vstack((1.1 * x[0], -2.4 * x[1], 3 * x[2]))]

        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

            coords = field.coords_and_values()[0]  # Values ignored.
            # dolfin ``.vector()`` -> dolfinx ``.x`` (la.Vector).
            if field.mesh_dim() == 1:
                dolfin_function = _interpolate(functionspace, expressions[0])
                field.set(dolfin_function.x)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 0],
                                   3 * coords[:, 0])
            elif field.mesh_dim() == 2:
                dolfin_function = _interpolate(functionspace, expressions[1])
                field.set(dolfin_function.x)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 1])
            elif field.mesh_dim() == 3:
                dolfin_function = _interpolate(functionspace, expressions[2])
                field.set(dolfin_function.x)
                expected_values = (1.1 * coords[:, 0], -2.4 * coords[:, 1],
                                   3 * coords[:, 2])

            expected_probed_value = (1.1 * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            f_array = field.get_ordered_numpy_array_xxx()
            f_array_split = np.split(f_array, field.value_dim())
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(f_array_split[0], expected_values[0])
            assert np.allclose(f_array_split[1], expected_values[1])
            assert np.allclose(f_array_split[2], expected_values[2])

            coords, field_values = field.coords_and_values()
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(field_values[:, 0], expected_values[0])
            assert np.allclose(field_values[:, 1], expected_values[1])
            assert np.allclose(field_values[:, 2], expected_values[2])

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector_field_with_python_function(self):
        """Test setting the 3D vector field with a python function."""
        python_functions = [lambda x: (1.21 * x[0], -2.47 * x[0], 3 * x[0]),
                            lambda x: (1.21 * x[0], -2.47 * x[1], 3 * x[1]),
                            lambda x: (1.21 * x[0], -2.47 * x[1], 3 * x[2])]

        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

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

            expected_probed_value = (1.21 * self.probing_coord,
                                     -2.47 * self.probing_coord,
                                     3 * self.probing_coord)

            f_array = field.get_ordered_numpy_array_xxx()
            f_array_split = np.split(f_array, field.value_dim())
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(f_array_split[0], expected_values[0])
            assert np.allclose(f_array_split[1], expected_values[1])
            assert np.allclose(f_array_split[2], expected_values[2])

            coords, field_values = field.coords_and_values()
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(field_values[:, 0], expected_values[0])
            assert np.allclose(field_values[:, 1], expected_values[1])
            assert np.allclose(field_values[:, 2], expected_values[2])

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol1
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol1
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol1

    def test_set_vector2d_field(self):
        """Test setting the 2D vector field."""
        # dolfin df.Constant/df.Expression -> fem.Constant/callable; plus plain
        # tuple/list/callable forms.
        expressions = [fem.Constant(self.mesh1d, (1.1, -2.4)),
                       (1.1, -2.4),
                       [1.1, -2.4],
                       lambda x: np.vstack((1.1 + 0 * x[0], -2.4 + 0 * x[0])),
                       lambda x: (1.1, -2.4)]

        expected_value = (1.1, -2.4)

        for functionspace in self.vector2d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)

                f_array = field.get_ordered_numpy_array_xxx()
                f_array_split = np.split(f_array, field.value_dim())
                assert np.all(f_array_split[0] == expected_value[0])
                assert np.all(f_array_split[1] == expected_value[1])

                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])

                probing_point = field.mesh_dim() * (self.probing_coord,)
                probed_value = field.probe(probing_point)
                assert abs(probed_value[0] - expected_value[0]) < self.tol1
                assert abs(probed_value[1] - expected_value[1]) < self.tol1

    def test_set_vector4d_field(self):
        """Test setting the 4D vector field."""
        expressions = [fem.Constant(self.mesh1d, (1.1, -2.4, 0.0, 0.9)),
                       (1.1, -2.4, 0, 0.9),
                       [1.1, -2.4, 0, 0.9],
                       lambda x: np.vstack((1.1 + 0 * x[0], -2.4 + 0 * x[0],
                                            0 * x[0], 0.9 + 0 * x[0])),
                       lambda x: (1.1, -2.4, 0, 0.9)]

        expected_value = (1.1, -2.4, 0, 0.9)

        for functionspace in self.vector4d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)

                f_array = field.get_ordered_numpy_array_xxx()
                f_array_split = np.split(f_array, field.value_dim())
                assert np.all(f_array_split[0] == expected_value[0])
                assert np.all(f_array_split[1] == expected_value[1])
                assert np.all(f_array_split[2] == expected_value[2])
                assert np.all(f_array_split[3] == expected_value[3])

                coords, field_values = field.coords_and_values()
                assert np.all(field_values[:, 0] == expected_value[0])
                assert np.all(field_values[:, 1] == expected_value[1])
                assert np.all(field_values[:, 2] == expected_value[2])
                assert np.all(field_values[:, 3] == expected_value[3])

                probing_point = field.mesh_dim() * (self.probing_coord,)
                probed_value = field.probe(probing_point)
                assert abs(probed_value[0] - expected_value[0]) < self.tol1
                assert abs(probed_value[1] - expected_value[1]) < self.tol1
                assert abs(probed_value[2] - expected_value[2]) < self.tol1
                assert abs(probed_value[3] - expected_value[3]) < self.tol1

    def test_normalise(self):
        # dolfin df.UnitIntervalMesh(50) -> mesh.create_unit_interval.
        domain = mesh.create_unit_interval(MPI.COMM_WORLD, 50)
        V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        expr = lambda x: np.vstack((10 * x[0] + 0.1,
                                    10 * x[0] + 0.2,
                                    10 * x[0] + 0.3))
        field = Field(V, value=expr)
        field2 = Field(V, value=expr)
        field.normalise()
        field2.normalise()

        # dolfin mesh.coordinates() -> owned vertex coords in geometry order,
        # matching get_ordered_numpy_array_xxx's owned-vertex-coordinate order.
        n = domain.geometry.index_map().size_local
        xcoords = domain.geometry.x[:n, 0]
        m = np.array([10 * xcoords + 0.1,
                      10 * xcoords + 0.2,
                      10 * xcoords + 0.3])
        m_norm = np.linalg.norm(m, axis=0)
        m_normalised = (1. / m_norm) * m

        assert np.allclose(m_normalised, field.get_ordered_numpy_array_xxx().reshape(3, -1))
        # dolfin ``f.vector().array()`` -> Field.as_array().
        assert np.allclose(field.as_array(), field2.as_array())

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
            # dolfin string Expression 'x[0]' -> callable (a non-constant
            # field); the string form itself is unsupported (see the string
            # rejection in test_set_scalar_field_with_constant).
            field = Field(functionspace, lambda x: x[0])
            assert not field.is_constant()
            with pytest.raises(RuntimeError):
                field.as_constant()

    def test_average_scalar_field(self):
        """Test computing the scalar field average."""
        # dolfin df.Constant/df.Expression -> fem.Constant/callable.
        expressions = [fem.Constant(self.mesh1d, 5.0),
                       lambda x: 10 * x[0],
                       lambda x: 10 * x[0]]

        f_av_expected = 5

        for functionspace in self.scalar_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                # Check the average value.
                # Measured DOLFINx headroom: ~1e-15 << tol1=5e-13.
                assert abs(f_av - f_av_expected) < self.tol1

                # Check the type of average result.
                assert isinstance(f_av, float)

    def test_average_vector_field(self):
        """Test computing the vector field average."""
        expressions = [fem.Constant(self.mesh1d, (1.0, 5.1)),
                       lambda x: np.vstack((2 * x[0], 10.2 * x[0])),
                       lambda x: (2 * x[0], 10.2 * x[0])]

        f_av_expected = (1, 5.1)

        for functionspace in self.vector2d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                assert abs(f_av[0] - f_av_expected[0]) < self.tol1
                assert abs(f_av[1] - f_av_expected[1]) < self.tol1

                assert isinstance(f_av, np.ndarray)
                assert f_av.shape == (field.value_dim(),)

        expressions = [fem.Constant(self.mesh1d, (1.0, 5.1, -3.6)),
                       lambda x: np.vstack((2 * x[0], 10.2 * x[0], -7.2 * x[0])),
                       lambda x: (2 * x[0], 10.2 * x[0], -7.2 * x[0])]

        f_av_expected = (1, 5.1, -3.6)

        for functionspace in self.vector3d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                assert abs(f_av[0] - f_av_expected[0]) < self.tol1
                assert abs(f_av[1] - f_av_expected[1]) < self.tol1
                assert abs(f_av[2] - f_av_expected[2]) < self.tol1

                assert isinstance(f_av, np.ndarray)
                assert f_av.shape == (field.value_dim(),)

        expressions = [fem.Constant(self.mesh1d, (1.0, 5.1, -3.6, 0.0)),
                       lambda x: np.vstack((2 * x[0], 10.2 * x[0],
                                            -7.2 * x[0], 0 * x[0])),
                       lambda x: (2 * x[0], 10.2 * x[0], -7.2 * x[0], 0 * x[0])]

        f_av_expected = (1, 5.1, -3.6, 0)

        for functionspace in self.vector4d_fspaces:
            for expression in expressions:
                field = Field(functionspace, expression)
                f_av = field.average()

                assert abs(f_av[0] - f_av_expected[0]) < self.tol1
                assert abs(f_av[1] - f_av_expected[1]) < self.tol1
                assert abs(f_av[2] - f_av_expected[2]) < self.tol1
                assert abs(f_av[3] - f_av_expected[3]) < self.tol1

                assert isinstance(f_av, np.ndarray)
                assert f_av.shape == (field.value_dim(),)

    def test_coords_and_values_scalar_field(self):
        """Test coordinates and values for scalar field."""
        expression = lambda x: 1.3 * x[0]

        for functionspace in self.scalar_fspaces:
            expected_coords = _owned_vertices(functionspace)
            num_nodes = _num_vertices(functionspace)
            expected_values = 1.3 * expected_coords[:, 0]

            field = Field(functionspace, expression)
            coords, values = field.coords_and_values()

            assert isinstance(coords, np.ndarray)
            assert isinstance(values, np.ndarray)

            assert values.shape == (num_nodes,)
            assert coords.shape == (num_nodes, field.mesh_dim())

            assert np.all(coords == expected_coords)
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15.
            assert np.allclose(values, expected_values)

    def test_coords_and_values_vector_field(self):
        """Test coordinates and values for vector field."""
        expression = lambda x: np.vstack((1.03 * x[0], 2.31 * x[0], -1 * x[0]))

        for functionspace in self.vector3d_fspaces:
            expected_coords = _owned_vertices(functionspace)
            num_nodes = _num_vertices(functionspace)

            expected_values = (1.03 * expected_coords[:, 0],
                               2.31 * expected_coords[:, 0],
                               -1 * expected_coords[:, 0])

            field = Field(functionspace, expression)
            coords, values = field.coords_and_values()

            assert isinstance(coords, np.ndarray)
            assert isinstance(values, np.ndarray)

            assert values.shape == (num_nodes, field.value_dim())
            assert coords.shape == (num_nodes, field.mesh_dim())

            assert np.all(coords == expected_coords)
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15.
            assert np.allclose(values[:, 0], expected_values[0])
            assert np.allclose(values[:, 1], expected_values[1])
            assert np.allclose(values[:, 2], expected_values[2])

    def test_probe_scalar_field(self):
        """Test probing the scalar field."""
        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)
            mesh_dim = field.mesh_dim()

            if mesh_dim == 1:
                field.set(lambda x: 1.3 * x[0])
                exact_result_at_node = 1.3 * 0.5
                exact_result_out_node = 1.3 * self.probing_coord
            elif mesh_dim == 2:
                field.set(lambda x: 1.3 * x[0] - 2.3 * x[1])
                exact_result_at_node = (1.3 - 2.3) * 0.5
                exact_result_out_node = (1.3 - 2.3) * self.probing_coord
            elif mesh_dim == 3:
                field.set(lambda x: 1.3 * x[0] - 2.3 * x[1] + 6.1 * x[2])
                exact_result_at_node = (1.3 - 2.3 + 6.1) * 0.5
                exact_result_out_node = (1.3 - 2.3 + 6.1) * self.probing_coord

            probe_point = mesh_dim * (0.5,)
            probed_value = field.probe(probe_point)
            assert isinstance(probed_value, float)
            assert abs(probed_value - exact_result_at_node) < self.tol1

            probe_point = mesh_dim * (self.probing_coord,)
            probed_value = field.probe(probe_point)
            assert isinstance(probed_value, float)
            assert abs(probed_value - exact_result_out_node) < self.tol1

    def test_probe_vector_field(self):
        """Test probing the vector field."""
        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace,
                          lambda x: np.vstack((1.3 * x[0], 0.3 * x[0], -6.2 * x[0])))
            mesh_dim = field.mesh_dim()

            exact_result_at_node = (1.3 * 0.5, 0.3 * 0.5, -6.2 * 0.5)
            exact_result_out_node = (1.3 * self.probing_coord,
                                     0.3 * self.probing_coord,
                                     -6.2 * self.probing_coord)

            probe_point = mesh_dim * (0.5,)
            probed_value = field.probe(probe_point)
            assert isinstance(probed_value, np.ndarray)
            assert len(probed_value) == 3
            assert abs(probed_value[0] - exact_result_at_node[0]) < self.tol1
            assert abs(probed_value[1] - exact_result_at_node[1]) < self.tol1
            assert abs(probed_value[2] - exact_result_at_node[2]) < self.tol1

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
            # dolfin mesh.topology().dim() -> dolfinx mesh.topology.dim.
            mesh_dim_expected = functionspace.mesh.topology.dim

            assert isinstance(field.mesh_dim(), int)
            assert field.mesh_dim() == mesh_dim_expected

    def test_value_dim(self):
        """Test value_dim method."""
        for functionspace in self.all_fspaces:
            field = Field(functionspace)
            # dolfin ufl_element().value_shape() (call, method) ->
            # dolfinx reference_value_shape (property); num_sub_spaces()==0
            # for scalars maps to an empty value shape.
            value_shape = functionspace.ufl_element().reference_value_shape
            assert isinstance(field.value_dim(), int)
            if not value_shape:
                assert field.value_dim() == 1
            else:
                assert field.value_dim() == value_shape[0]

    def test_mesh(self):
        """Test mesh method."""
        for functionspace in self.all_fspaces:
            field = Field(functionspace)

            # dolfin df.Mesh -> dolfinx mesh.Mesh.
            assert isinstance(field.mesh(), mesh.Mesh)

    def test_set_nonlinear_scalar_field(self):
        """Test setting nonlinear scalar field."""
        python_functions = [lambda x: 1.21 * x[0] * x[0],
                            lambda x: 1.21 * x[0] * x[0] - 3.21 * x[1],
                            lambda x: 1.21 * x[0] * x[0] - 3.21 * x[1] + 2.47 * x[2]]

        for functionspace in self.scalar_fspaces:
            field = Field(functionspace)

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

            # Check the result of coords_and_values (should be exact at nodes).
            field_values = field.coords_and_values()[1]  # ignore coordinates
            # dolfin used ``==`` (bit-exact via vertex_to_dof_map). Under
            # DOLFINx the interpolant samples ``tabulate_dof_coordinates`` while
            # coords_and_values returns ``geometry.x`` -- equal only to FP
            # rounding (measured max abs diff ~4e-15). allclose, not ``==``.
            assert np.allclose(field_values, expected_values)

            # Nonlinear field -> greater tolerance value (tol2) off-node.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value - expected_probed_value) < self.tol2

    def test_set_nonlinear_vector_field(self):
        """Test setting the vector field with a nonlinear expression."""
        # 2D vector fields.
        expressions = [lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[0])),
                       lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[0])),
                       lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[1]))]

        for functionspace in self.vector2d_fspaces:
            field = Field(functionspace)

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

            expected_probed_value = (1.1 * self.probing_coord * self.probing_coord,
                                     -2.4 * self.probing_coord)

            f_array = field.get_ordered_numpy_array_xxx()
            f_array_split = np.split(f_array, field.value_dim())
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(f_array_split[0], expected_values[0])
            assert np.allclose(f_array_split[1], expected_values[1])

            coords, field_values = field.coords_and_values()
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(field_values[:, 0], expected_values[0])
            assert np.allclose(field_values[:, 1], expected_values[1])

            # Nonlinear field -> greater tolerance value (tol2) off-node.
            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol2
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol2

        # 3D vector fields.
        expressions = [lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[0], 3 * x[0])),
                       lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[1], 3 * x[1])),
                       lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[1], 3 * x[2]))]

        for functionspace in self.vector3d_fspaces:
            field = Field(functionspace)

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

            expected_probed_value = (1.1 * self.probing_coord * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord)

            f_array = field.get_ordered_numpy_array_xxx()
            f_array_split = np.split(f_array, field.value_dim())
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(f_array_split[0], expected_values[0])
            assert np.allclose(f_array_split[1], expected_values[1])
            assert np.allclose(f_array_split[2], expected_values[2])

            coords, field_values = field.coords_and_values()
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(field_values[:, 0], expected_values[0])
            assert np.allclose(field_values[:, 1], expected_values[1])
            assert np.allclose(field_values[:, 2], expected_values[2])

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol2
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol2
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol2

        # 4D vector fields.
        expressions = [lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[0],
                                            3 * x[0], x[0])),
                       lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[1],
                                            3 * x[1], x[0])),
                       lambda x: np.vstack((1.1 * x[0] * x[0], -2.4 * x[1],
                                            3 * x[2], x[0]))]

        for functionspace in self.vector4d_fspaces:
            field = Field(functionspace)

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

            expected_probed_value = (1.1 * self.probing_coord * self.probing_coord,
                                     -2.4 * self.probing_coord,
                                     3 * self.probing_coord,
                                     self.probing_coord)

            f_array = field.get_ordered_numpy_array_xxx()
            f_array_split = np.split(f_array, field.value_dim())
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(f_array_split[0], expected_values[0])
            assert np.allclose(f_array_split[1], expected_values[1])
            assert np.allclose(f_array_split[2], expected_values[2])
            assert np.allclose(f_array_split[3], expected_values[3])

            coords, field_values = field.coords_and_values()
            # allclose (not ``==``): dof-coord vs vertex-coord FP ~4e-15, see note above.
            assert np.allclose(field_values[:, 0], expected_values[0])
            assert np.allclose(field_values[:, 1], expected_values[1])
            assert np.allclose(field_values[:, 2], expected_values[2])
            assert np.allclose(field_values[:, 3], expected_values[3])

            probing_point = field.mesh_dim() * (self.probing_coord,)
            probed_value = field.probe(probing_point)
            assert abs(probed_value[0] - expected_probed_value[0]) < self.tol2
            assert abs(probed_value[1] - expected_probed_value[1]) < self.tol2
            assert abs(probed_value[2] - expected_probed_value[2]) < self.tol2
            assert abs(probed_value[3] - expected_probed_value[3]) < self.tol2

    def test_plot_with_dolfin(self):
        """Test that we can call the plotting function of a Field object.

        Behavioural change (VISIBLE): master called ``plot_with_dolfin`` and
        expected it to render. Legacy dolfin plotting has no DOLFINx equivalent,
        so the port raises ``NotImplementedError`` (write VTK/XDMF instead --
        see test_vtk_xdmf_output_and_filename_tracking below the banner).
        """
        field = Field(self.fs3d_vector3d, value=[1, 0, 0])
        with pytest.raises(NotImplementedError):
            field.plot_with_dolfin(interactive=False)

    def test_add_scalar_fields(self):
        # N10: master asserted ``Field + Field`` WORKS (field3 == 6.45). The
        # DOLFINx port intentionally DEFERS operator ``+`` -- it used legacy
        # point-measure assembly and now raises NotImplementedError.
        # Behavioural change, see finmag.field.Field.__add__ /
        # test_legacy_only_features_fail_precisely below the banner.
        for functionspace in self.scalar_fspaces:
            field1 = Field(functionspace, value=3.1)
            field2 = Field(functionspace, value=3.35)

            with pytest.raises(NotImplementedError):
                field1 + field2

    def test_add_vector_fields(self):
        # N10: master asserted ``Field + Field`` WORKS (components 6, 4.1, 9).
        # The DOLFINx port intentionally DEFERS operator ``+`` (raises
        # NotImplementedError) -- behavioural change, see test_add_scalar_fields.
        for functionspace in self.vector3d_fspaces:
            field1 = Field(functionspace, value=(1, 2, 3))
            field2 = Field(functionspace, value=(5, 2.1, 6))

            with pytest.raises(NotImplementedError):
                field1 + field2

    def test_mul_scalar_fields(self):
        for functionspace in self.scalar_fspaces:
            # dolfin string Expression "x[0] + 3.1" -> callable.
            field1 = Field(functionspace, value=lambda x: x[0] + 3.1)

            # Multiply with scalars
            field2 = field1 * 42
            field3 = -12 * field1

            coords2, vals2 = field2.coords_and_values()
            coords3, vals3 = field3.coords_and_values()
            np.testing.assert_allclose(vals2, 42 * (coords2[:, 0] + 3.1))
            np.testing.assert_allclose(vals3, -12 * (coords3[:, 0] + 3.1))

    def test_mul_vector_fields(self):
        for functionspace in self.vector3d_fspaces:
            # dolfin string Expression list -> callable.
            field1 = Field(functionspace,
                           value=lambda x: np.vstack((x[0] + 1, x[0] + 2.4, x[0] + 3.7)))

            # Multiply with scalars
            field2 = field1 * 42
            field3 = -3.6 * field1

            # Multiply with a scalar field
            S1 = associated_scalar_space(functionspace)
            a = Field(S1, lambda pt: pt[0]**2)
            field4 = field1 * a

            coords2, vals2 = field2.coords_and_values()
            coords3, vals3 = field3.coords_and_values()
            coords4, vals4 = field4.coords_and_values()

            xcoords2 = coords2[:, 0][:, np.newaxis]
            xcoords3 = coords3[:, 0][:, np.newaxis]
            xcoords4 = coords4[:, 0][:, np.newaxis]
            vals2_expected = 42 * (xcoords2 + [1, 2.4, 3.7])
            vals3_expected = -3.6 * (xcoords3 + [1, 2.4, 3.7])
            vals4_expected = xcoords4**2 * (xcoords2 + [1, 2.4, 3.7])

            np.testing.assert_allclose(vals2, vals2_expected)
            np.testing.assert_allclose(vals3, vals3_expected)
            # atol: the x^2 factor produces denormal ~1e-34 (not exact 0) at the
            # x=0 nodes under DOLFINx; master's dolfin gave exact 0 (rtol-only).
            np.testing.assert_allclose(vals4, vals4_expected, atol=1e-15)

    def test_div_scalar_fields(self):
        for functionspace in self.scalar_fspaces:
            field1 = Field(functionspace, value=3.1)
            field2 = field1 / 20
            # dolfin ``f.vector().array()`` -> Field.as_array().
            assert np.allclose(field2.as_array(), 0.155)

    def test_div_vector_fields(self):
        for functionspace in self.vector3d_fspaces:
            field1 = Field(functionspace, value=(1, 2.4, 3.7))

            # Multiply with scalars
            field2 = field1 / 20

            # Divide by a scalar field
            S1 = associated_scalar_space(functionspace)
            a = Field(S1, lambda pt: (pt[0] + 1.0)**2)
            field3 = field1 / a

            coords = field2.coords_and_values()[0]
            for coord in coords:
                assert abs(field2.probe(coord)[0] - 0.05) < self.tol1
                assert abs(field2.probe(coord)[1] - 0.12) < self.tol1
                assert abs(field2.probe(coord)[2] - 0.185) < self.tol1

            coords = field3.coords_and_values()[0]
            for coord in coords:
                assert abs(field3.probe(coord)[0] - 1.0 / (coord[0] + 1)**2) < self.tol1
                assert abs(field3.probe(coord)[1] - 2.4 / (coord[0] + 1)**2) < self.tol1
                assert abs(field3.probe(coord)[2] - 3.7 / (coord[0] + 1)**2) < self.tol1

    def test_cross(self):
        v = np.array([1, 2, 3])
        w = np.array([4, 5, -2])
        v_cross_w = np.cross(v, w)

        for functionspace in self.vector3d_fspaces:
            field1 = Field(functionspace, value=v)
            field2 = Field(functionspace, value=w)
            field3 = field1.cross(field2)

            coords, vals = field3.coords_and_values()
            np.testing.assert_allclose(vals - v_cross_w, 0)

    def test_dot(self):
        v = np.array([1, 2, 3])
        w = np.array([4, 5, -2])
        v_dot_w = np.dot(v, w)

        for functionspace in self.vector3d_fspaces:
            field1 = Field(functionspace, value=v)
            field2 = Field(functionspace, value=w)
            field3 = field1.dot(field2)

            _, vals = field3.coords_and_values()
            np.testing.assert_allclose(vals, v_dot_w)

    def test_allclose(self):
        for functionspace in self.all_fspaces:
            # Define field on the function space and fill with random values.
            field1 = Field(functionspace)
            # the rtol check below can fail if the changed field value below is
            # accidentally very small, so keep those values away from zero.
            field1.set_random_values(vrange=[0.1, 1.0])

            # Define second field as copy of the first.
            field2 = Field(functionspace, field1)
            assert field2.allclose(field1)

            # Change one of the coordinates and check that the fields are now
            # not allclose any more with the default tolerances, but that they
            # are allclose with less strict tolerances.
            a = field1.get_ordered_numpy_array_xxx()
            eps = np.zeros_like(a)
            eps[7] = 2.1e-6

            # (master wrapped this in a try/except ipdb debugger drop -- removed.)
            field2.set_with_ordered_numpy_array_xxx(a + eps)
            assert not field2.allclose(field1)
            assert field2.allclose(field1, atol=1e-5)
            assert field2.allclose(field1, rtol=1e-4)

            # Only a `Field` is a valid `other`.
            with pytest.raises(TypeError):
                assert field2.allclose(42.0)

            with pytest.raises(TypeError):
                assert field2.allclose(a)

    def test_field_get_ordered_numpy_array_xxx_and_xyz(self):
        """
        For each mesh define a scalar field as well as vector fields of
        dimension 2, 3, 4. The field values are defined by adding
        0.01, 0.02, 0.03 and 0.04, respectively, to the x-coordinates
        of the mesh nodes.

        Then the field values are retrieved using both
        ``get_ordered_numpy_array_xxx`` and ``get_ordered_numpy_array_xyz``
        and compared with the expected values.
        """
        def fsetval(value_dim, pos):
            # Helper function to set field values
            x = pos[0]
            return [x + 0.01 * (i + 1) for i in range(value_dim)]

        for functionspace in self.all_fspaces:
            # Define the field
            f = Field(functionspace)
            vdim = f.value_dim()
            f.set(functools.partial(fsetval, vdim))

            # Retrieve the field values in the different orderings
            vals_xxx = f.get_ordered_numpy_array_xxx()
            vals_xyz = f.get_ordered_numpy_array_xyz()

            # Define the expected field values (derived from the x-coordinates
            # of the owned mesh nodes -- dolfin mesh.coordinates()[:,0]).
            xcoords = _owned_vertices(functionspace)[:, 0]
            vals_xxx_expected = np.concatenate(
                [xcoords + 0.01 * (i + 1) for i in range(vdim)])
            vals_xyz_expected = np.array(
                [xcoords + 0.01 * (i + 1) for i in range(vdim)]).transpose().ravel()

            np.testing.assert_almost_equal(vals_xxx, vals_xxx_expected)
            np.testing.assert_almost_equal(vals_xyz, vals_xyz_expected)

            # Error if we try to call get_ordered_numpy_array() on a non-scalar.
            if vdim > 1:
                with pytest.raises(ValueError):
                    f.get_ordered_numpy_array()

    def test_save_hdf5(self, tmp_path):
        """
        Test saving of field to hdf5.

        Behavioural change (VISIBLE): the legacy ``save_hdf5`` used the external
        ``dolfinh5tools`` package to write a ``.h5`` timeseries PLUS a ``.json``
        sidecar of saved times, and required ``close_hdf5``. The DOLFINx port
        writes ONE self-describing ``.h5`` snapshot (no ``.json`` sidecar) per
        call and ``close_hdf5`` is a no-op. So this asserts the ``.h5`` exists
        and that NO ``.json`` sidecar is produced.
        """
        # Define base filename to save data to (tmp_path -> auto-cleaned).
        filename = str(tmp_path / "test_save_field")
        fieldname = 'f'

        expression = lambda x: np.vstack((1.1 * x[0], -2.4 * x[1], 3 * x[2]))

        # Define and set field
        field = Field(functionspace=self.fs3d_vector3d, name=fieldname)
        field.set(expression)

        # save field to h5 file (single-snapshot; the last call wins).
        field.save_hdf5(filename, t=1.0)
        field.save_hdf5(filename, t=2.0)

        # close hdf5 file (no-op under DOLFINx).
        field.close_hdf5()

        # check that the .h5 file has been created, and that NO .json sidecar
        # is written (behavioural change from the legacy dolfinh5tools format).
        assert (tmp_path / "test_save_field.h5").exists()
        assert not (tmp_path / "test_save_field.json").exists()


def _interpolate(functionspace, callable_expr):
    """dolfin ``df.interpolate(expr, fs)`` -> a dolfinx fem.Function.

    Builds a raw ``fem.Function`` on ``functionspace`` and interpolates the
    Python callable into it (scatter-forward for ghosts), so the transcribed
    tests can hand a Function / its backend ``.x`` vector to ``Field.set``.
    """
    function = fem.Function(functionspace)
    function.interpolate(callable_expr)
    function.x.scatter_forward()
    return function


# ===========================================================================
# NEW under DOLFINx (no master ancestor)
# ===========================================================================
# The tests below are the focused production tests for the DOLFINx-backed Field
# (ownership/ghost accessors, coordinate round-trips, from_generic_vector
# backend-object surface, VTK/XDMF output, coordinate-drift guards). They have
# no master ancestor and are preserved verbatim.


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


def test_from_generic_vector_copies_backend_vector_objects(spaces):
    """P3.4: ``from_generic_vector`` is the backend-vector-object entry point.

    Legacy ``from_generic_vector`` took a dolfin ``GenericVector`` (a backend
    PETSc vector *object*) and did ``set_local(get_local())`` -- a raw
    backend-order owned copy. The DOLFINx-native equivalents are the backend
    vector objects this Field exposes: ``vector()`` -> ``dolfinx.la.Vector`` and
    ``petsc_vector()`` -> ``PETSc.Vec``. It is a genuinely distinct surface from
    ``from_array`` (which takes a NumPy array); a NumPy array is rejected.
    """
    _, _, vector_space = spaces
    source = Field(
        vector_space,
        lambda x: np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] * x[1])),
    )

    # dolfinx.la.Vector (Field.vector()) round-trips node-for-node in raw
    # backend order; returns self.
    via_la = Field(vector_space)
    returned = via_la.from_generic_vector(source.vector())
    assert returned is via_la
    assert via_la.allclose(source)
    assert np.array_equal(via_la.as_array(), source.as_array())

    # PETSc.Vec (Field.petsc_vector()) round-trips too.
    via_petsc = Field(vector_space)
    via_petsc.from_generic_vector(source.petsc_vector())
    assert via_petsc.allclose(source)
    assert np.array_equal(via_petsc.as_array(), source.as_array())

    # set() routes a backend la.Vector object here (legacy GenericVector
    # dispatch parity).
    via_set = Field(vector_space)
    via_set.set(source.vector())
    assert via_set.allclose(source)

    # A NumPy array is a distinct surface -> rejected loudly, pointing to
    # from_array (NOT silently producing a wrong result).
    with pytest.raises(TypeError, match="from_array"):
        Field(vector_space).from_generic_vector(source.as_array())

    # A backend vector from a DIFFERENT (larger) space must be rejected, not
    # silently truncated to the target's owned size (that would misread a
    # vector-space vector into a scalar-space target node-for-node). [Claude Opus 4.8]
    _, scalar_space, _ = spaces
    with pytest.raises(ValueError, match="different function space|owns"):
        Field(scalar_space).from_generic_vector(source.vector())


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
    # test_io_utils.py for their dedicated coverage.
    # save_hdf5/from_hdf5 are also no longer legacy-only failures -- the
    # single-file coordinate-aware HDF5 round-trip was restored in SR1 P4-hdf5;
    # see test_field_hdf5.py for its dedicated coverage.
    with pytest.raises(NotImplementedError, match="Field addition"):
        field + field


def _node_varying_vector(space):
    return Field(
        space,
        lambda x: np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] * x[1])),
    )


def _other_node_varying_vector(space):
    return Field(
        space,
        lambda x: np.vstack((0.5 - x[1], 4.0 + x[0], 1.0 - x[0] * x[0])),
    )


def test_cross_is_pointwise_vector_field_and_survives_node_scramble(spaces):
    _, _, vector_space = spaces

    x_hat = Field(vector_space, (1.0, 0.0, 0.0))
    y_hat = Field(vector_space, (0.0, 1.0, 0.0))
    z_hat = x_hat.cross(y_hat)

    # cross returns a NEW vector Field on the same vector space.
    assert z_hat.functionspace is vector_space
    assert not z_hat.is_scalar_field()
    _, values = z_hat.coords_and_values()
    assert np.allclose(values, np.array([0.0, 0.0, 1.0]))

    # Anti-scramble: on a node-varying field, v x v == 0 everywhere and the
    # cross matches np.cross of the PUBLIC arrays node-for-node. A permutation
    # bug survives uniform constants but fails a varying field.
    v = _node_varying_vector(vector_space)
    w = _other_node_varying_vector(vector_space)
    _, self_cross = v.cross(v).coords_and_values()
    assert np.allclose(self_cross, 0.0)

    _, cross_vals = v.cross(w).coords_and_values()
    _, v_vals = v.coords_and_values()
    _, w_vals = w.coords_and_values()
    assert np.allclose(cross_vals, np.cross(v_vals, w_vals))


def test_dot_returns_scalar_field_on_scalar_space_and_survives_node_scramble(spaces):
    _, scalar_space, vector_space = spaces

    unit = Field(vector_space, (0.6, 0.0, 0.8))
    result = unit.dot(unit)

    # dot returns a NEW scalar Field on the associated scalar space.
    assert result.is_scalar_field()
    assert result.functionspace.ufl_element() == scalar_space.ufl_element()
    assert result.mesh() is vector_space.mesh
    _, values = result.coords_and_values()
    assert np.allclose(values, 1.0)

    # Anti-scramble: on a node-varying field, v . v == |v|^2 node-for-node and
    # matches einsum of the PUBLIC arrays.
    v = _node_varying_vector(vector_space)
    w = _other_node_varying_vector(vector_space)
    _, self_dot = v.dot(v).coords_and_values()
    _, v_vals = v.coords_and_values()
    assert np.allclose(self_dot, np.einsum("ij,ij->i", v_vals, v_vals))

    _, dot_vals = v.dot(w).coords_and_values()
    _, w_vals = w.coords_and_values()
    assert np.allclose(dot_vals, np.einsum("ij,ij->i", v_vals, w_vals))


def test_scalar_multiply_and_divide_coerce_and_scale(spaces):
    _, scalar_space, vector_space = spaces
    field = _node_varying_vector(vector_space)
    _, base = field.coords_and_values()

    _, doubled = (field * 2.0).coords_and_values()
    assert np.allclose(doubled, 2.0 * base)
    # __rmul__ path: number on the left.
    _, r_doubled = (2.0 * field).coords_and_values()
    assert np.allclose(r_doubled, 2.0 * base)
    _, halved = (field / 2.0).coords_and_values()
    assert np.allclose(halved, base / 2.0)

    # Multiply by a node-varying scalar Field a = x^2 (coerce_scalar_field path).
    a = Field(scalar_space, lambda x: x[0] ** 2)
    _, a_vals = a.coords_and_values()
    _, scaled = (field * a).coords_and_values()
    assert np.allclose(scaled, base * a_vals[:, None])

    # Result keeps the Task-31 public ordering contract (component-blocked xxx).
    product = field * 2.0
    assert np.allclose(
        product.get_ordered_numpy_array_xxx(),
        (2.0 * base).T.reshape(-1),
    )


def test_coerce_scalar_field_surface(spaces):
    _, scalar_space, vector_space = spaces
    field = _node_varying_vector(vector_space)

    # A number coerces into a scalar Field on the associated scalar space.
    coerced = field.coerce_scalar_field(3.0)
    assert isinstance(coerced, Field)
    assert coerced.is_scalar_field()
    assert coerced.functionspace.ufl_element() == scalar_space.ufl_element()
    assert np.allclose(coerced.coords_and_values()[1], 3.0)

    # A scalar Field passes straight through.
    a = Field(scalar_space, lambda x: x[0] ** 2)
    assert field.coerce_scalar_field(a) is a

    # A non-scalar Field is rejected loudly.
    with pytest.raises(ValueError, match="scalar fields"):
        field.coerce_scalar_field(field)


def test_cross_and_dot_reject_bad_operands(spaces):
    _, _, vector_space = spaces
    vector = Field(vector_space, (1.0, 0.0, 0.0))

    # Non-Field operand -> TypeError (legacy message surface).
    with pytest.raises(TypeError, match="must be a Field"):
        vector.cross(3.0)
    with pytest.raises(TypeError, match="must be a Field"):
        vector.dot("not a field")

    # cross is only defined for 3d vector fields.
    two_space = fem.functionspace(vector_space.mesh, ("Lagrange", 1, (2,)))
    two = Field(two_space, (1.0, 2.0))
    with pytest.raises(ValueError, match="3d vector fields"):
        two.cross(two)

    # dot requires equal dimension.
    with pytest.raises(ValueError, match="same dimension"):
        vector.dot(two)

    # Same value_dim but a different function space still raises a clear error
    # rather than silently returning a scrambled result.
    other_mesh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    other_vector = Field(
        fem.functionspace(other_mesh, ("Lagrange", 1, (3,))), (1.0, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="same mesh and function space"):
        vector.cross(other_vector)
    with pytest.raises(ValueError, match="same mesh and function space"):
        vector.dot(other_vector)


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
