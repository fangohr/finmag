# Field class: a thin wrapper around the dolfin functions
# for unified and convent access to them.
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
    def __init__(self, functionspace, value=None, name=None, unit=None):
        self.functionspace = functionspace

        self.f = df.Function(self.functionspace)  # Create a zero-function.
        # Set the function value f if specified.
        if value is not None:
            self.set(value)

        self.name = name
        if name is not None:
            self.f.rename(name, name)  # Rename both function's name and label.

        self.unit = unit

    def set(self, value):
        if isinstance(value, (df.Constant, df.Expression)):
            # Dolfin Constant and Expression type values
            # suitable for both scalar and vector fields.
            self.f = df.interpolate(value, self.functionspace)

        elif isinstance(value, (basestring, int, float)):
            # Int, float, and basestring (str and unicode) type values
            # suitable only for scalar fields.
            if isinstance(self.functionspace, df.FunctionSpace):
                self.f = df.interpolate(df.Constant(value), self.functionspace)
            else:
                raise TypeError('{} type inappropriate for setting '
                                'the vector field value.'.format(type(value)))

        elif isinstance(value, (tuple, list, np.ndarray)):
            # Tuple, list, and numpy array type values
            # suitable only for vector fields.
            if isinstance(self.functionspace, df.VectorFunctionSpace):
                # The dimensions of value and vector field value must be equal.
                if len(value) == self.value_dim():
                    self.f = df.interpolate(df.Constant(value),
                                            self.functionspace)
                else:
                    raise ValueError('Vector function space value dimension '
                                     '({}) and value dimension ({}) '
                                     'are not equal.'.format(len(value),
                                                             self.value_dim()))
            else:
                raise TypeError('{} type inappropriate for setting '
                                'the scalar field value.'.format(type(value)))

        elif hasattr(value, '__call__'):
            # Python function type values
            # suitable for both vector and scalar fields.

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
            # nor vector field initialsiation.
            raise TypeError('Type {} inappropriate for setting '
                            'the field value.'.format(type(value)))

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
            f_array = self.f.vector().array()
            vtd_map = df.vertex_to_dof_map(self.functionspace)
            num_nodes = len(coords)

            # scalar field
            if isinstance(self.functionspace, df.FunctionSpace):
                values = np.empty(num_nodes)
                for i in xrange(num_nodes):
                    try:
                        values[i] = f_array[vtd_map[i]]
                    except IndexError:
                        raise NotImplementedError

            # vector field
            elif isinstance(self.functionspace, df.VectorFunctionSpace):
                value_dim = self.functionspace.ufl_element().value_shape()[0]
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
                        # investigate.  (Max, 15.5.2014)
                        raise NotImplementedError("TODO: How to deal with this?"
                                                  " What does it even mean?!?")

            return coords, values
        
        else:
            raise NotImplementedError('This method is implemented only for '
                                      'Lagrange (CG) and Discontinuous '
                                      'Lagrange (DG) function space families.')

    def probe_field(self, coord):
        return self.f(coord)

    def mesh_dim(self):
        return self.f.geometric_dimension()

    def value_dim(self):
        if isinstance(self.functionspace, df.FunctionSpace):
            return 1
        else:
            # value_shape() returns a tuple (N,) and int is required
            return self.functionspace.ufl_element().value_shape()[0]

    def save(self, filename):
        """Dispatches to specialists"""
        raise NotImplementedError

    def save_pvd(self, filename):
        """Save to pvd file using dolfin code"""
        raise NotImplementedError

    def save_hdf5(self, filename):
        """Save to hdf5 file using dolfin code"""
        raise NotImplementedError

    def load_hdf5(self, filename):
        """Load field from hdf5 file using dolfin code"""
        raise NotImplementedError
