"""
Representation of scalar and vector fields, as well as
operations on them backed by a dolfin function.

This module exists because things like per-node operations or exporting of
field values to convenient formats are awkward to do in dolfin currently.
Additionally, some things are non-trivial to get right, especially in parallel.
This class therefore acts as a "single point of contant", so that we don't
duplicate code all over the FinMag code base.

"""
import logging
import dolfin as df
import numpy as np
from finmag.util.helpers import expression_from_python_function

log = logging.getLogger(name="finmag")


class Field(object):
    """
    Representation of scalar and vector fields using a dolfin function.

    You can set the field values using a wide range of object types:
        - tuples, lists, ints, floats, basestrings, numpy arrrays
        - dolfin constants, expressions and functions
        - callables
        - files in hdf5

    The Field class provides raw access to the field at some particular point
    or all nodes. It also computes derived entities of the field, such as
    spatially averaged energy. It outputs data suited for visualisation
    or storage.

    """
    def __init__(self, functionspace, value=None, normalised=False, name=None, unit=None):
        self.functionspace = functionspace
        self.f = df.Function(self.functionspace)
        self.name = name

        if value is not None:
            self.value = value
            self.set(value, normalised=normalised)

        if name is not None:
            self.f.rename(name, name)  # set function's name and label

        self.unit = unit

        functionspace_family = self.f.ufl_element().family()
        if functionspace_family == 'Lagrange':
            dim = self.value_dim()
            self.v2d_xyz = df.vertex_to_dof_map(self.functionspace)
            n1 = len(self.v2d_xyz)
            self.v2d_xxx = ((self.v2d_xyz.reshape(n1/dim, dim)).transpose()).reshape(-1,)

            self.d2v_xyz = df.dof_to_vertex_map(self.functionspace)
            n2 = len(self.d2v_xyz)
            self.d2v_xxx = self.d2v_xyz.copy()
            for i in xrange(n2):
                j = self.d2v_xyz[i]
                self.d2v_xxx[i] = (j%dim)*n1/dim + (j/dim)
            self.d2v_xxx.shape=(-1,)

    def __call__(self, x):
        """
        Shorthand so user can do field(x) instead of field.f(x) to interpolate.

        """
        return self.f(x)

    def assert_is_scalar_field(self):
        if self.value_dim() != 1:
            raise ValueError(
                "This function is only defined for scalar fields.")

    def from_array(self, arr):
        assert isinstance(arr, np.ndarray)
        if arr.shape == (3,) and isinstance(self.functionspace, df.VectorFunctionSpace):
            self.from_constant(df.Constant(arr))
        else:
            if arr.shape[0] == self.f.vector().local_size():
                self.f.vector().set_local(arr)
            else:
                # in serial, local_size == size, so this will only warn in parallel
                log.warning("Global setting of field values by overwriting with np.array.")
                self.f.vector()[:] = arr

    def from_callable(self, func):
        assert hasattr(func, "__call__") and not isinstance(func, df.Function)
        expr = expression_from_python_function(func, self.functionspace)
        self.from_expression(expr)

    def from_constant(self, constant):
        assert isinstance(constant, df.Constant)
        self.f.assign(constant)

    def from_expression(self, expr, **kwargs):
        """
        Set field values using dolfin expression or the ingredients for one,
        in which case it will build the dolfin expression for you.

        """
        if not isinstance(expr, df.Expression):
            if isinstance(self.functionspace, df.FunctionSpace):
                assert (isinstance(expr, basestring) or
                        isinstance(expr, (tuple, list)) and len(expr) == 1)
                expr = str(expr)  # dolfin does not like unicode in the expression
            if isinstance(self.functionspace, df.VectorFunctionSpace):
                assert isinstance(expr, (tuple, list)) and len(expr) == 3
                assert all(isinstance(item, basestring) for item in expr)
                map(str, expr)  # dolfin does not like unicode in the expression
            expr = df.Expression(expr, **kwargs)
        temp_function = df.interpolate(expr, self.functionspace)
        self.f.vector().set_local(temp_function.vector().get_local())

    def from_field(self, field):
        assert isinstance(field, Field)
        if self.functionspace == field.functionspace:
            self.f.vector().set_local(field.f.vector().get_local())
        else:
            temp_function = df.interpolate(field.f, self.functionspace)
            self.f.vector().set_local(temp_function.vector().get_local())

    def from_function(self, function):
        assert isinstance(function, df.Function)
        self.f.vector().set_local(function.vector().get_local())

    def from_generic_vector(self, vector):
        assert isinstance(vector, df.GenericVector)
        self.f.vector().set_local(vector.get_local())

    def from_sequence(self, seq):
        assert isinstance(seq, (tuple, list))
        self._check_can_set_vector_value(seq)
        self.from_constant(df.Constant(seq))

    def _check_can_set_scalar_value(self):
        if not isinstance(self.functionspace, df.FunctionSpace):
            raise ValueError("Cannot set vector field with scalar value.")

    def _check_can_set_vector_value(self, seq):
        if not isinstance(self.functionspace, df.VectorFunctionSpace):
            raise ValueError("Cannot set scalar field with vector value.")
        if len(seq) != self.functionspace.num_sub_spaces():
            raise ValueError(
                "Cannot set vector field with value of non-matching dimension "
                "({} != {})", len(seq), self.functionspace.num_sub_spaces())

    def set(self, value, normalised=False, **kwargs):
        """
        Set field values using `value` and normalise if `normalised` is True.

        The parameter `value` can be one of many different types,
        as described in the class docstring. This method avoids the user
        having to find the correct `from_*` method to call.

        """
        if isinstance(value, df.Constant):
            self.from_constant(value)
        elif isinstance(value, df.Expression):
            self.from_expression(value)
        elif isinstance(value, df.Function):
            self.from_function(value)
        elif isinstance(value, Field):
            self.from_field(value)
        elif isinstance(value, df.GenericVector):
            self.from_generic_vector(value)
        elif isinstance(value, (int, float)):
            self._check_can_set_scalar_value()
            self.from_constant(df.Constant(value))
        elif isinstance(value, basestring):
            self._check_can_set_scalar_value()
            self.from_expression(value, **kwargs)
        elif (isinstance(value, (tuple, list)) and
              all(isinstance(item, basestring) for item in value)):
            self._check_can_set_vector_value(value)
            self.from_expression(value, **kwargs)
        elif isinstance(value, (tuple, list)):
            self.from_sequence(value)
        elif isinstance(value, np.ndarray):
            self.from_array(value)
        elif hasattr(value, '__call__'):
            # this matches df.Function as well, so this clause needs to
            # be after the one checking for df.Function
            self.from_callable(value)
        else:
            raise TypeError("Can't set field values using {}.".format(type(value)))

        if normalised:
            self.normalise()

    def set_with_numpy_array_debug(self, value, normalised=False):
        """ONLY for debugging"""
        self.f.vector().set_local(value)

        if normalised:
            self.normalise()

    def get_ordered_numpy_array(self):
        """
        For a scalar field, return the dolfin function as an ordered
        numpy array, such that the field values are in the same order
        as the vertices of the underlying mesh (as returned by
        `mesh.coordinates`).

        Note:

        This function is only defined for scalar fields and raises an
        error if it is applied to a vector field. For the latter, use
        either

            get_ordered_numpy_array_xxx

        or

            get_ordered_numpy_array_xyz

        depending on the order in which you want the values to be returned.

        """
        self.assert_is_scalar_field()
        return self.get_ordered_numpy_array_xxx()

    def get_ordered_numpy_array_xyz(self):
        """
        Returns the dolfin function as an ordered numpy array, so that
        all components at the same node are grouped together. For example,
        for a 3d vector field the values are returned in the following order:

          [f_1x, f_1y, f_1z,  f_2x, f_2y, f_2z,  f_3x, f_3y, f_3z,  ...]

        Note: In the case of a scalar field this function is equivalent to
        `get_ordered_numpy_array_xxx` (but for vector fields they yield
        different results).
        """
        return self.get_numpy_array_debug()[self.v2d_xyz]

    def get_ordered_numpy_array_xxx(self):
        """
        Returns the dolfin function as an ordered numpy array, so that
        all x-components at different nodes are grouped together, and
        similarly for the other components. For example, for a 3d
        vector field the values are returned in the following order:

          [f_1x, f_2x, f_3x, ...,  f_1y, f_2y, f_3y, ...,  f_1z, f_2z, f_3z, ...]

        Note: In the case of a scalar field this function is equivalent to
        `get_ordered_numpy_array_xyz` (but for vector fields they yield
        different results).
        """
        return self.get_numpy_array_debug()[self.v2d_xxx]

    # def order2_to_order1(self, order2):
    #     """Returns the dolfin function as an ordered numpy array, so that
    #     in the case of vector fields all components of different nodes
    #     are grouped together."""
    #     n = len(order2)
    #     return ((order2.reshape(3, n/3)).transpose()).reshape(n)
    #
    # def order1_to_order2(self, order1):
    #     """Returns the dolfin function as an ordered numpy array, so that
    #     in the case of vector fields all components of different nodes
    #     are grouped together."""
    #     n = len(order1)
    #     return ((order1.reshape(n/3, 3)).transpose()).reshape(n)

    def set_with_ordered_numpy_array(self, ordered_array):
        """
        Set the scalar field using an ordered numpy array (where the field
        values have the same ordering as the vertices in the underlying
        mesh).

        This function raises an error if the field is not a scalar field.
        """
        self.assert_is_scalar_field()
        self.set_with_ordered_numpy_array_xxx(ordered_array)

    def set_with_ordered_numpy_array_xyz(self, ordered_array):
        """
        Set the field using an ordered numpy array in "xyz" order.
        For example, for a 3d vector field the values should be
        arranged as follows:

          [f_1x, f_1y, f_1z, f_2x, f_2y, f_2z, f_3x, f_3y, f_3z, ...]

        For a scalar field this function is equivalent to
        `set_with_ordered_numpy_array_xxx`.
        """
        self.set(ordered_array[self.d2v_xyz])

    def set_with_ordered_numpy_array_xxx(self, ordered_array):
        """
        Set the field using an ordered numpy array in "xxx" order.
        For example, for a 3d vector field the values should be
        arranged as follows:

          [f_1x, f_2x, f_3x, ..., f_1y, f_2y, f_3y, ..., f_1z, f_2z, f_3z, ...]

        For a scalar field this function is equivalent to
        `set_with_ordered_numpy_array_xyz`.
        """
        self.set(ordered_array[self.d2v_xxx])

    def as_array(self):
        return self.f.vector().array()

    def as_vector(self):
        return self.f.vector()

    def get_numpy_array_debug(self):
        """ONLY for debugging"""
        return self.f.vector().array()

    def is_scalar_field(self):
        """
        Return `True` if the Field is a scalar field and `False` otherwise.
        """
        return isinstance(self.functionspace, df.FunctionSpace)

    def is_constant(self, eps=1e-14):
        """
        Return `True` if the Field has a unique constant value across the mesh
        and `False` otherwise.

        """
        # Scalar field
        if self.is_scalar_field():
            maxval = self.f.vector().max()  # global (!) maximum value
            minval = self.f.vector().min()  # global (!) minimum value
            return (maxval - minval) < eps
        # Vector field
        else:
            raise NotImplementedError()

    def as_constant(self, eps=1e-14):
        """
        If the Field has a unique constant value across the mesh, return this value.
        Otherwise a RuntimeError is raised.
        """
        if self.is_scalar_field():
            maxval = self.f.vector().max()  # global (!) maximum value
            minval = self.f.vector().min()  # global (!) minimum value
            if (maxval - minval) < eps:
                return maxval
            else:
                raise RuntimeError("Field does not have a unique constant value.")
        else:
            raise NotImplementedError()

    def average(self, dx=df.dx):
        """
        Return the spatial field average.

        Returns:
          f_average (float for scalar and np.ndarray for vector field)

        """
        # Compute the mesh "volume". For 1D mesh "volume" is the length and
        # for 2D mesh is the area of the mesh.
        volume = df.assemble(df.Constant(1) * dx(self.mesh()))

        # Scalar field.
        if self.is_scalar_field():
            return df.assemble(self.f * dx) / volume

        # Vector field.
        else:
            f_average = []
            # Compute the average for every vector component independently.
            for i in xrange(self.value_dim()):
                f_average.append(df.assemble(self.f[i] * dx))

            return np.array(f_average) / volume

    def coords_and_values(self, t=None):
        # The function values are defined at mesh nodes only for
        # specific function space families. In finmag, the only families
        # of interest are Lagrange (CG) and Discontinuous Lagrange (DG).
        # Therefore, if the function space is not CG-family-type,
        # values cannot be associated to mesh nodes.
        functionspace_family = self.f.ufl_element().family()
        if functionspace_family == 'Discontinuous Lagrange':
            # Function values are not defined at nodes.
            raise TypeError('The function space is Discontinuous Lagrange '
                            '(DG) family type, for which the function values '
                            'are not defined at mesh nodes.')

        elif functionspace_family == 'Lagrange':
            # Function values are defined at nodes.
            coords = self.functionspace.mesh().coordinates()
            num_nodes = self.functionspace.mesh().num_vertices()
            f_array = self.f.vector().array()  # numpy array
            vtd_map = df.vertex_to_dof_map(self.functionspace)

            value_dim = self.value_dim()
            values = np.empty((num_nodes, value_dim))
            for i in xrange(num_nodes):
                try:
                    values[i, :] = f_array[vtd_map[value_dim * i:
                                                   value_dim * (i + 1)]]
                except IndexError:
                    # This only occurs in parallel and is probably related
                    # to ghost nodes. I thought we could ignore those, but
                    # this doesn't seem to be true since the resulting
                    # array of function values has the wrong size. Need to
                    # investigate.  (Max, 15/05/2014)
                    raise NotImplementedError("TODO")

            return coords, values

        else:
            raise NotImplementedError('This method is not implemented '
                                      'for {} family type function '
                                      'spaces.'.format(functionspace_family))

    def __add__(self, other):
        result = Field(self.functionspace)
        result.set(self.f.vector() + other.f.vector())
        return result

    def probe(self, coord):
        return self.f(coord)

    def mesh(self):
        return self.functionspace.mesh()

    def mesh_dim(self):
        return self.functionspace.mesh().topology().dim()

    def mesh_dofmap(self):
        return self.functionspace.dofmap()

    def value_dim(self):
        if self.is_scalar_field():
            # Scalar field.
            return 1
        else:
            # value_shape() returns a tuple (N,) and int is required.
            return self.functionspace.ufl_element().value_shape()[0]

    def vector(self):
        return self.f.vector()

    def petsc_vector(self):
        return df.as_backend_type(self.f.vector()).vec()

    def save_pvd(self, filename):
        """Save to pvd file using dolfin code"""
        if filename[-4:] != '.pvd':
            filename += '.pvd'
        pvd_file = df.File(filename)
        pvd_file << self.f

    def save(self, filename):
        """Dispatches to specialists"""
        raise NotImplementedError

    def save_hdf5(self, filename):
        """Save to hdf5 file using dolfin code"""
        raise NotImplementedError

    def load_hdf5(self, filename):
        """Load field from hdf5 file using dolfin code"""
        raise NotImplementedError

    def plot_with_dolfin(self, interactive=True):
        df.plot(self.f, interactive=True)

    def normalise(self):
        """
        Overwrite own field values with normalised ones.

        """
        dofmap = df.vertex_to_dof_map(self.functionspace)
        reordered = self.f.vector().array()[dofmap]  # [x1, y1, z1, ..., xn, yn, zn]
        vectors = reordered.reshape((3, -1))  # [[x1, y1, z1], ..., [xn, yn, zn]]
        lengths = np.sqrt(np.add.reduce(vectors * vectors, axis=1))
        normalised = np.dot(vectors.T, np.diag(1 / lengths)).T.ravel()
        vertexmap = df.dof_to_vertex_map(self.functionspace)
        normalised_original_order = normalised[vertexmap]
        self.from_array(normalised_original_order)
