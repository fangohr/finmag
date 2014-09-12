import dolfin as df
import numpy as np


class Field(object):
    """
    A thin wrapper around the dolfin function for unified and convenient
    operations on scalar and vector fields.

    The Field class represents all scalar and vector fields that need to
    be represented as dolfin functions (i.e. discretized fields on a mesh).
    It is always tied to a specific mesh and choice of function space.

    There are two reasons Field class exists:

      - Certain things are awkward (or currently impossible) when using
        dolfin.Functions directly (for example, per-node operations on
        vector fields or storing a dolfin.Function in a format convenient
        for use in Finmag).

      - Even if things are possible, sometimes they are non-trivial to
        get right, especially in parallel. Therefore this class acts as
        a "single point of contact" so that we don't duplicate
        functionality all over the Finmag code base.

    What does the field class do?

      - Set field values:
        - from constant (dolfin constant, tuple, list, int, float, basestring)
        - from dolfin expression
        - from python function
        - from files (hdf5)

      - Retrieve field values:
        - derived entities such as spatially averaged energy.
        - raw access to field at some particular point or raw data for
          all nodes

      - Output data:
        - for visualisation
        - for data storage

    """
    def __init__(self, functionspace, value=None, normalised=False,
                 name=None, unit=None):
        self.functionspace = functionspace
        self.f = df.Function(self.functionspace)  # Create a zero-function.

        if value is not None:
            self.set(value, normalised=normalised)

        self.name = name
        if name is not None:
            self.f.rename(name, name)  # Rename function's name and label.

        self.unit = unit

    def set(self, value, normalised=False):
        """
        Set the field value f and normalise the field if specified in __init__.

        Args:
          value: The value for setting the field.
                 The type of value argument can be:
                   - Scalar field: int, float, basestring, df.Constant,
                                   df.Expression, python function
                   - Vector field: tuple, list, numpy array, df.Constant,
                                   df.Expression, python function

        """
        # Dolfin Constant and Expression type values
        # appropriate for both scalar and vector fields.
        if isinstance(value, (df.Constant, df.Expression)):
            self.f = df.interpolate(value, self.functionspace)

        # Dolfin function type value
        # appropriate for both scalar and vector field.
        elif isinstance(value, df.Function):
            if value.function_space() == self.functionspace:
                self.f = value
            else:
                raise TypeError('Function and field functionspaces '
                                'do not match')

        # Generic vector type value
        # appropriate for both scalar and vector fields.
        elif isinstance(value, df.GenericVector):
            self.f.vector()[:] = value

        # Int, float, and basestring (str and unicode) type values
        # appropriate only for scalar fields.
        elif isinstance(value, (int, float, basestring)):
            if isinstance(self.functionspace, df.FunctionSpace):
                self.f = df.interpolate(df.Constant(value), self.functionspace)
            else:
                raise TypeError('{} inappropriate for setting the vector '
                                'field value.'.format(type(value)))

        # Tuple, list, and numpy array type values
        # appropriate only for vector fields.
        elif isinstance(value, (tuple, list, np.ndarray)):
            # Dimensions of value and vector field must be equal.
            if isinstance(self.functionspace, df.VectorFunctionSpace) and \
                    len(value) == self.value_dim():
                self.f = df.interpolate(df.Constant(value), self.functionspace)

            elif len(value) != self.value_dim():
                raise ValueError('Vector function space value dimension ({}) '
                                 'and value dimension ({}) are not '
                                 'equal.'.format(len(value), self.value_dim()))

            else:
                raise TypeError('{} inappropriate for setting the scalar '
                                'field  value.'.format(type(value)))

        # Python function type values
        # appropriate for both vector and scalar fields.
        elif hasattr(value, '__call__'):
            # Functionspace is made visible to WrappedExpression class.
            fspace_for_wexp = self.functionspace

            class WrappedExpression(df.Expression):
                def __init__(self, value):
                    self.fun = value

                def eval(self, eval_result, x):
                    eval_result[:] = self.fun(x)

                def value_shape(self):
                    # Return the dimension of field value as a tuple.
                    # For instance:
                    # () for scalar field and
                    # (N,) for N dimensional vector field
                    return fspace_for_wexp.ufl_element().value_shape()

            wrapped_expression = WrappedExpression(value)
            self.f = df.interpolate(wrapped_expression, self.functionspace)

        # The value type cannot be used for neither scalar
        # nor vector field setting.
        else:
            raise TypeError('{} inappropriate for setting the field '
                            'value.'.format(type(value)))

        # Normalise the function if required.
        if normalised:
            self.normalise()

    def set_with_numpy_array_debug(self, value, normalised=False):
        """ONLY for debugging"""
        self.f.vector().set_local(value)

        if normalised:
            self.normalise()

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
                norm_squared += self.f[i]**2
            norm = norm_squared**0.5

            self.f = df.project(self.f/norm, self.functionspace)

        else:
            # Scalar field normalisation is not required. Normalisation
            # can be implemented so that the whole field is divided by
            # its maximum value. This might cause some problems if the
            # code is run in parallel.
            raise NotImplementedError('The normalisation of scalar field '
                                      'values is not implemented.')

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
        if isinstance(self.functionspace, df.FunctionSpace):
            return df.assemble(self.f * dx) / volume

        # Vector field.
        elif isinstance(self.functionspace, df.VectorFunctionSpace):
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
                    values[i, :] = f_array[vtd_map[value_dim*i:
                                                   value_dim*(i+1)]]
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
        if isinstance(self.functionspace, df.FunctionSpace):
            # Scalar field.
            return 1
        else:
            # value_shape() returns a tuple (N,) and int is required.
            return self.functionspace.ufl_element().value_shape()[0]
        
    def vector(self):
        return self.f.vector()

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

        ``method`` is an integer to choose the method. We are not sure what method is best at the moment.

        This method is not implemented only for vector fields with 3 compoents at every vertex.

        """

        if not target:
            raise NotImplementedError("This is missing - could cerate a df.Function(V) here")

        assert self.value_dim() in [3], "Only implemented for 3d vector field"

        if method == 1:
            wx, wy, wz = self.f.split(deepcopy=True)
            wnorm = np.sqrt(wx.vector() * wx.vector()  + wy.vector() * wy.vector() + wz.vector() * wz.vector())
            target.vector().set_local(wnorm)

        elif method == 2:
            raise NotImplementedError("this code doesn't compile in Cython - deactivate for now")
            #V_vec = self.f.function_space()
            #dofs0 = V_vec.sub(0).dofmap().dofs()    # indices of x-components
            #dofs1 = V_vec.sub(1).dofmap().dofs()    # indices of y-components
            #dofs2 = V_vec.sub(2).dofmap().dofs()    # indices of z-components

            #target.vector()[:] = np.sqrt(w.vector()[dofs0] * w.vector()[dofs0] +\
            #                            w.vector()[dofs1] * w.vector()[dofs1] +\
            #                             w.vector()[dofs2] * w.vector()[dofs2])

        elif method == 3:
            try:
                import finmag.native.clib as clib
            except ImportError:
                print "please go to the finmag/native/src/clib and run 'make' to install clib"
            f = df.as_backend_type(target.vector()).vec()
            w = df.as_backend_type(self.f.vector()).vec()
            clib.norm(w, f)

        else:
            raise NotImplementedError("method {} unknown".format(method))
