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

    def __call__(self, x):
        """
        Shorthand so user can do field(x) instead of field.f(x) to interpolate.

        """
        return self.f(x)

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
        assert isinstance(self.functionspace, df.VectorFunctionSpace)
        assert isinstance(seq, (tuple, list))
        assert len(seq) == self.functionspace.num_sub_spaces()
        self.from_constant(df.Constant(seq))

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
            self.from_constant(df.Constant(value))
        elif (isinstance(value, basestring) or
              isinstance(value, (tuple, list)) and
                all(isinstance(item, basestring) for item in value)):
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

    def as_array(self):
        return self.f.vector().array()

    def as_vector(self):
        return self.f.vector()

    def get_numpy_array_debug(self):
        """ONLY for debugging"""
        return self.f.vector().array()

    def normalise(self):
        """
        Normalise the vector field so that the norm=1.

        Note:
          This method is not implemented for scalar fields.

        """
        # Normalisation is implemented only for vector fields.
        if isinstance(self.functionspace, df.VectorFunctionSpace):
            # Vector field is normalised so that norm=1 at all mesh nodes.
            norm_squared = 0
            for i in range(self.value_dim()):
                norm_squared += self.f[i] ** 2
            norm = norm_squared ** 0.5

            self.f = df.project(self.f / norm, self.functionspace)

        else:
            # Scalar field normalisation is not required. Normalisation
            # can be implemented so that the whole field is divided by
            # its maximum value. This might cause some problems if the
            # code is run in parallel.
            raise NotImplementedError('The normalisation of scalar field '
                                      'values is not implemented.')

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

    def compute_pointwise_norm(self, target=None, method=1):
        """
        Compute the norm of the function pointwise, i.e. for every vertex.

        Arguments:

        ``target`` is a scalar dolfin function to accommodate the norm

        ``method`` is an integer to choose the method. We are not sure
        what method is best at the moment.

        This method is not implemented only for vector fields
        with 3 compoents at every vertex.

        """

        if not target:
            raise NotImplementedError("This is missing - could cerate a "
                                      "df.Function(V) here")

        assert self.value_dim() in [3], "Only implemented for 3d vector field"

        if method == 1:
            wx, wy, wz = self.f.split(deepcopy=True)
            wnorm = np.sqrt(wx.vector() * wx.vector() +
                            wy.vector() * wy.vector() +
                            wz.vector() * wz.vector())
            target.vector().set_local(wnorm)

        elif method == 2:
            raise NotImplementedError("this code doesn't compile in Cython "
                                      "- deactivate for now")
            # V_vec = self.f.function_space()
            # dofs0 = V_vec.sub(0).dofmap().dofs()    # indices of x-components
            # dofs1 = V_vec.sub(1).dofmap().dofs()    # indices of y-components
            # dofs2 = V_vec.sub(2).dofmap().dofs()    # indices of z-components

            # target.vector()[:] = np.sqrt(w.vector()[dofs0]*w.vector()[dofs0]+
            #                            w.vector()[dofs1]*w.vector()[dofs1]+
            #                             w.vector()[dofs2]*w.vector()[dofs2])

        elif method == 3:
            try:
                import finmag.native.clib as clib
            except ImportError:
                print "please go to the finmag/native/src/clib and " + \
                    "run 'make' to install clib"
            f = df.as_backend_type(target.vector()).vec()
            w = df.as_backend_type(self.f.vector()).vec()
            clib.norm(w, f)

        else:
            raise NotImplementedError("method {} unknown".format(method))
