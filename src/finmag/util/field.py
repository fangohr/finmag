# Field class: a thin wrapper around the dolfin functions for unified and convent access to them.
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
#   - from constant
#   - from dolfin expression
#   - from python function (ideally)
#   - from files
#
# - Retrieve field values
#
#   - derived entities such as spatially averaged energy. Can express dolfin code -> quick
#
#   - raw access to field at some particular point or raw data for all nodes (debugging?)
#
# - output data
#
#   - for visualisation (use dolfin tools)
#   - for data storage (use dolfin tools)

from __future__ import division
import dolfin as df
import numpy as np


class Field(object):
    def __init__(self, mesh, family, degree, dim):
        """
        Create a discretized field on the given mesh
        """
        self.mesh = mesh
        self.family = family
        self.degree = degree
        self.dim = dim
        self.u = None
        if self.dim == 1:
            #raise NotImplementedError()
            print("Warning: should we create a df.FunctionSpace for dim=1 instead of a df.VectorFunctionSpace?")
        else:
            self.V = df.VectorFunctionSpace(mesh, family, degree, dim=dim)

    def set(self, value):
        if not isinstance(value, (df.Constant, df.Expression)):
            raise NotImplementedError("Currently `value` must be a dolfin Constant or Expression.")
        self.u = df.interpolate(value, self.V)

    def get_coords_and_values(self):
        """
        Return a list of mesh vertex coordinates and associated field values.
        In parallel, this only returns the coordinates and values owned by
        the current process.

        This function should only be used for debugging!
        """
        if self.family != 'CG':
            raise NotImplementedError(
                "This function is only implemented for finite element families where "
                "the degrees of freedoms are not defined at the mesh vertices.")

        a = self.u.vector().array()
        coords = self.mesh.coordinates()
        vtd = df.vertex_to_dof_map(self.V)

        values = list()
        for i, dum in enumerate(coords):
            try:
                values.append([a[vtd[self.dim*i+k]] for k in xrange(self.dim)])
            except IndexError:
                # This only occurs in parallel and is probably related
                # to ghost nodes. I thought we could ignore those, but
                # this doesn't seem to be true since the resulting
                # array of function values has the wrong size. Need to
                # investigate.  (Max, 15.5.2014)
                raise NotImplementedError("XXX TODO: How to deal with this? What does it even mean?!?")
        values = np.array(values)

        return coords, values
