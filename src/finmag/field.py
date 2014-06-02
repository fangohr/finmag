# Field class: a thin wrapper around the dolfin functions
# for unified and convenient access to them.
#
# There are two reasons this exists at all:
#
#    - Certain things are awkward (or currently impossible) when using
#      dolfin.Functions directly (for example, per-node operations on
#      vector fields or storing a dolfin.Function in a format convenient
#      for use in Finmag).
#
#    - Even if things are possible, sometimes they are non-trivial to
#      get right, especially in parallel. Therefore this class acts as
#      a "single point of contact" to that we don't duplicate
#      functionality all over the Finmag code base.
#
# The `Field` class represents all scalar and vector fields we need to
# represent as dolfin functions (i.e. discretized fields on a
# mesh). It is always tied to a specific mesh and choice of function
# space (e.g. linear CG).
#
# It does *not* directly represent the abstract concept of a
# (continuous) physical field.
#
#
# What does the field class do?
#
# - Set field values (initialisation, changes at some point during the
#   simulation) for primary fields M, H_Zeeman, current, etc.
#
#   - from constant DONE
#   - from dolfin expression DONE
#   - from python function DONE
#   - from files TODO
#
# - Retrieve field values
#
#   - derived entities such as spatially averaged energy.
#     Can express dolfin code -> quick
#
#   - raw access to field at some particular point or raw
#     data for all nodes (debugging?)
#
# - output data
#
#   - for visualisation (use dolfin tools)
#   - for data storage (use dolfin tools)
#
# Maybe to be added later:
#
# def nodal_volume(self):
#
#     return nodal_volume

import dolfin as df
import numpy as np


class Field(object):
    def __init__(self, functionspace, value=None, normalised=False,
                 name=None, unit=None):
        self.functionspace = functionspace
        self.f = df.Function(self.functionspace)  # Create a zero-function.

        # Set the function (f) value if specified. Field is normalised
        # in the set method if normalised=True.
        self.normalised = normalised
        if value is not None:
            self.set(value)

        self.name = name
        if name is not None:
            self.f.rename(name, name)  # Rename function's name and label.

        self.unit = unit

    def set(self, value):
        if isinstance(value, (df.Constant, df.Expression)):
            # Dolfin Constant and Expression type values
            # appropriate for both scalar and vector fields.
            self.f = df.interpolate(value, self.functionspace)

        elif isinstance(value, (int, float, basestring)):
            # Int, float, and basestring (str and unicode) type values
            # appropriate only for scalar fields.
            if isinstance(self.functionspace, df.FunctionSpace):
                self.f = df.interpolate(df.Constant(value), self.functionspace)
            else:
                raise TypeError('{} inappropriate for setting the vector '
                                'field value.'.format(type(value)))

        elif isinstance(value, (tuple, list, np.ndarray)):
            # Tuple, list, and numpy array type values
            # appropriate only for vector fields.
            if isinstance(self.functionspace, df.VectorFunctionSpace) and \
                    len(value) == self.value_dim():
                # Value and vector field dimensions must be equal.
                self.f = df.interpolate(df.Constant(value), self.functionspace)

            elif len(value) != self.value_dim():
                raise ValueError('Vector function space value dimension ({}) '
                                 'and value dimension ({}) are not '
                                 'equal.'.format(len(value), self.value_dim()))

            else:
                raise TypeError('{} inappropriate for setting the scalar '
                                'field  value.'.format(type(value)))

        elif hasattr(value, '__call__'):
            # Python function type values
            # appropriate for both vector and scalar fields.

            # Wrapped dolfin expression class which incorporates the python
            # function. For the value_shape method, functionspace is required.
            # However, it is impossible to pass it to __init__ method
            # (Marijan, Max 22/05/2014) since value_shape is called first.
            # Therefore, functionspace is made "global".
            fspace_for_wexp = self.functionspace  # functionspace made global

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

        else:
            # The value type cannot be used for neither scalar
            # nor vector field setting.
            raise TypeError('{} inappropriate for setting the field '
                            'value.'.format(type(value)))

        # Normalise the function if required.
        if self.normalised:
            self.normalise()

    def normalise(self):
        if isinstance(self.functionspace, df.VectorFunctionSpace):
            # Vector field is normalised so that
            # the vector norm is 1 at all mesh nodes.
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

    def average(self):
        # Compute the mesh "volume". For 1D mesh "volume" is the length and
        # for 2D mesh is the area of the mesh.
        volume = df.assemble(df.Constant(1) * df.dx, mesh=self.mesh())

        # Scalar field.
        if isinstance(self.functionspace, df.FunctionSpace):
            return df.assemble(self.f * df.dx) / volume

        # Vector field.
        elif isinstance(self.functionspace, df.VectorFunctionSpace):
            f_average = []
            # Compute the average for every vector component independently.
            for i in xrange(self.value_dim()):
                f_average.append(df.assemble(self.f[i] * df.dx))

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

    def probe_field(self, coord):
        return self.f(coord)

    def mesh(self):
        return self.functionspace.mesh()

    def mesh_dim(self):
        return self.functionspace.mesh().topology().dim()

    def value_dim(self):
        if isinstance(self.functionspace, df.FunctionSpace):
            # Scalar field.
            return 1
        else:
            # value_shape() returns a tuple (N,) and int is required.
            return self.functionspace.ufl_element().value_shape()[0]

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
