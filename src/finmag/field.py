# Field class: a thin wrapper around the dolfin functions for unified and convent access to them.
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
#      a "single point of contact" to that we don't duplicate functionality
#      all over the Finmag code base.
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
#   - from python function
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

import dolfin as df
import numpy as np


class Field(object):
    def __init__(self, functionspace, value=None, name=None, unit=None): 
        self.functionspace = functionspace
        self.f = df.Function(self.functionspace)
        if value is not None:
            self.set(value)
        self.name = name
        if name is not None:
            self.f.rename(name, name)
        self.unit = unit

    def set(self, value):
        """
        Set the field to value.
        Value can be constant, dolfin expression, python function, file.
        """
        # works for both scalar and vector
        if isinstance(value, (df.Constant, df.Expression)):
            self.f = df.interpolate(value, self.functionspace)
        # works only for scalar
        elif isinstance(value, (basestring, int, float)):
            self.f = df.interpolate(df.Constant(value), self.functionspace)
    
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

    def get_coords_and_values(self, t=None):
        """
        Return a list of mesh vertex coordinates and associated field values.
        In parallel, this only returns the coordinates and values owned by
        the current process.

        This function should only be used for debugging!
        """
        if self.f.ufl_element().family() != 'Lagrange':
            raise NotImplementedError(
                "This function is only implemented for finite element families where "
                "the degrees of freedoms are not defined at the mesh vertices.")

        f_array = self.f.vector().array()
        coords = self.functionspace.mesh().coordinates()
        vtd_map = df.vertex_to_dof_map(self.functionspace)

        values = list()
        for i in xrange(len(coords)):
            try:
                values.append(f_array[vtd_map[i]])
            except IndexError:
                # This only occurs in parallel and is probably related
                # to ghost nodes. I thought we could ignore those, but
                # this doesn't seem to be true since the resulting
                # array of function values has the wrong size. Need to
                # investigate.  (Max, 15.5.2014)
                raise NotImplementedError("XXX TODO: How to deal with this? What does it even mean?!?")
        values = np.array(values)

        return coords, values

    def probe_field(self, coord):
        """
        Probe and return the value of a field at point with coordinates coord.
        Coord can be a tuple, list or numpy array.
        """
        return self.f(coord)


# Maybe to be added later
#
#
#def nodal_volume(self):
#
#    return nodal_volume
