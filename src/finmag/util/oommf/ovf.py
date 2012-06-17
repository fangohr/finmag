# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Matteo Franchin

"""
Generic library to read/write files using the OOMMF OVF file format.
We support:
 - all major OVF versions (OVF 1.0 and 2.0)
 - all data modes (binary8, binary4 and text)
 - only rectangular mesh types (irregular mesh type is not supported, yet).
Here are few examples illustrating how to use the library:

EXAMPLE 1: reading an OVF file and retrieving the data

  from ovf import OVFFile
  ovf_file = OVFFile("filename.ovf")
  fl = ovf_file.get_field()
  # fl is a FieldLattice object, see module lattice.py
  # fl.lattice is a Lattice object, describing the mesh (lattice.py)
  # fl.field_data is the numpy array containing the data

EXAMPLE 2: creating a new OVF file

    from ovf import OVFFile, OVF10, OVF20

    # Create the data
    fl = FieldLattice("2.5e-9,97.5e-9,20/2.5e-9,47.5e-9,10/2.5e-9,7.5e-9,1")
    def setter_function(position):
        return [1, 0, 0]
    fl.set(setter_function)

    # Save it to file
    ovf = OVFFile()
    ovf.new(fl, version=OVF20, data_type="binary8")
    ovf.write("newfile.ovf")

  Note that after the 'ovf.new' method has been called, you can customize some
  of the fields of the OVF header by accessing it directly under
  'ovf.content'. 'ovf.content' gives you access to the content of the header.
  For example, if the header contains an entry 'xmin', then you can access it
  as 'ovf.content.a_segment.a_header.a_xmin'. However, there are some header
  entries which appear only in OVF version 1.0, but not in version 2.0.
  One of such entries is 'valueunit', which is valid in v 1.0 but is not valid
  in v 2.0 (where it is replaced by the entry 'valueunits'). In practice that
  means that the OVF 1.0 files have the entry

    ovf.content.a_segment.a_header.a_valueunit

  but do not have the entry

    ovf.content.a_segment.a_header.a_valueunits

  while for OVF 2.0 it is exactly the other way round.
  'ovf.content' then provides wrapper entries which work for both version
  1.0 and version 2.0. You should then try to use the properties of
  'ovf.content' when they are available, rather than using
  'ovf.content.a_segment...'.
  For example:

    h = ovf.content.a_segment.a_header
    h.a_title.value = "The title of my OVF file"
    h.a_meshunit.value = "m"
    ovf.content.valueunits = ["A/m", "A/m", "A/m"]
    ovf.write("newfile.ovf")

  Here we use:

    ovf.content.valueunits = ["A/m", "A/m", "A/m"]

  rather than:

    h.a_valueunits.value.units = ["A/m", "A/m", "A/m"]

  The last line is indeed valid only for OVF 2.0 but fails for OVF 1.0.
  The former one, in contrast, works both for version 1.0 and 2.0.

NOTE: The OVF file defines fields over a grid of cubes while the FieldLattice
  defines fields over points (the centers of the cubes, actually).
  For example:

    fl = FieldLattice("1,9,5/1,9,5/1,9,5")

  The FieldLattice above defines a cubic lattice of 5x5x5 points: all the ones
  obtained varying x, y and z among the values 1, 3, 5, 7, 9.
  For example (1, 1, 1) is the first point of the lattice. It lies exactly
  at the center of the first cube of the mesh, which occupies the space
  between 0-2 in each dimension.
  That should be enough to understand the relation between the OVF mesh and
  the corresponding FieldLattice. Since the cubes are all equal and the points
  are at the center of the cubes, the x-size of the cubes is equal to the
  spacing between the nodes along the x-axis. Similarly for the other
  directions of space y and z.
  If your mesh has just one cubes along one or more directions, then you have
  to put some care in specifying the FieldLattice. For example,

    fl = FieldLattice("1,3,1/1,3,1/1,3,1")

  defines a lattice with just one point (1, 1, 1) corresponding to a cube
  with size (2, 2, 2) and that occupies the region of space 0-2 in each
  dimension of space. The values (3, 3, 3) are given just to define what
  the spacing is, but is not used as a point in the mesh.
"""

__all__ = ["OVF10", "OVF20", "OVFFile", "OVFValueUnits", "OVFValueLabels"]

import struct
from numpy import array, ndarray

from lattice import FieldLattice

# Abbreviations for OVF versions
OVF10 = (1, 0)
OVF20 = (2, 0)
ANY_OVF = [OVF10, OVF20]

def name_normalise(name):
    for c in [" ", "\t", "\n"]:
        name = name.replace(c, "")
    return name.lower()

class OVFReadError(Exception): pass

class OVFVersionError(Exception): pass

class OVFNode(object):
    def __init__(self, subnodes=[], data=None):
        #print "Creating node %s:%s" % (type(self), data)
        self._subnodes = list(subnodes)
        self._data = data

    def _to_str(self, indent=0):
        s = " "*indent + "Node %s: data=%s\n" % (type(self), self._data)
        for subnode in self._subnodes:
            s += subnode._to_str(indent=indent+2)
        return s

    def __str__(self):
        return self._to_str()

    def _get_name(self):
        return self._data[0]

    def _set_name(self, n):
        self._data = (n, self._data[1])

    def _get_identity(self):
        return name_normalise(self._get_name())

    def _get_value(self):
        return self._data[1]

    def _set_value(self, v):
        self._data = (self._data[0], v)

    name = property(_get_name, _set_name)
    identity = property(_get_identity)
    value = property(_get_value, _set_value)

    def _add_as_attr(self, obj=None, prefix="a_"):
        if obj != None:
            assert obj != self
            setattr(obj, prefix + self.identity, self)

        for subnode in self._subnodes:
            subnode._add_as_attr(self, prefix=prefix)

    def read(self, stream, root=None):
        raise NotImplementedError("OVFNode.read not implemented!")

    def write(self, stream, root=None):
        raise NotImplementedError("OVFNode.write not implemented!")


class OVFSectionNode(OVFNode):
    required = None

    def __init__(self, value=[], data=None):
        OVFNode.__init__(self, value, data)

        name, value = data
        self.section_action = lvalue = value.lower()
        assert lvalue in ["begin", "end"], "lvalue is %s" % lvalue
        self.received = {}

    def read(self, stream, root=None):
        while True:
            node = read_node(stream)
            if node == None:
                return

            node_name = node.name
            self._subnodes.append(node)

            self.received[node_name] = node
            setattr(self, "a_" + node.identity, node)
            assert self != node

            if isinstance(node, OVFSectionNode):
                sa = node.section_action
                if sa == "begin":
                    node.read(stream, root=root)
                elif sa == "end":
                    self._end_section(node_name)
                    return
                else:
                    raise OVFReadError("Expected section end, but got '%s'."
                                       % node_name)

    def _end_section(self, name):
        expected = self.name
        if name != expected:
            raise OVFReadError("Expected end of section %s, but got end "
                               "of section %s." % (expected, name))

        # We check wether we got all we needed
        missing_value = []
        if self.required != None:
            for required_value in self.required:
                if not self.received.has_key(required_value):
                    missing_value.append(required_value)

        if missing_value:
            raise OVFReadError("Missing entries from %s section: %s."
                               % (name, ", ".join(missing_value)))

    def write(self, stream, root=None):
        if self.section_action == "begin":
            stream.write_line("# Begin: %s" % self.name)
        else:
            stream.write_line("# End: %s" % self.name)

        for n in self._subnodes:
            n.write(stream, root=root)

class OVFValueNode(OVFNode):
    def write(self, stream, root=None):
        v = self.value
        if v != None:
            stream.write_line("# %s: %s" % (self.name, self.value))

class OVFType:
    def __init__(self, s):
        pieces = s.lower().split()
        assert pieces[0] in ["oommf", "oommf:"], \
          "Unrecognised OVF version string (%s)" % s

        mesh_type = None
        if pieces[1] == "ovf":
            version_str = pieces[2]

        elif pieces[2] == "mesh":
            mesh_type = pieces[1]
            version_str = pieces[3]
            if version_str.startswith("v"):
                version_str = version_str[1:]

        if version_str in ["0.0a0", "0.99", "1.0"]:
            version = OVF10

        elif version_str in ["2.0"]:
            version = OVF20

        else:
            print ("Unknown OVF version '%s'. Assuming version 2.0.")
            version = OVF20
            version_str = "2.0"

        self.version_str = version_str
        self.version = version
        self.mesh_type = mesh_type

    def __str__(self):
        if self.version == OVF10:
            return "OOMMF: %s mesh v%s" % (self.mesh_type, self.version_str)
        else:
            return "OOMMF OVF %s" % self.version_str

def split_strings(s, delim='"'):
    """Similar to s.split() but do not split whatherver is included between
    two delimiters."""
    OUTSIDE, INSIDE, SPACE = range(3)
    state = SPACE

    n = len(s)
    i = 0
    begin = 0
    pieces = []
    while i < n:
        c = s[i]
        inc = 1
        if state == SPACE:
            if not c.isspace():
                begin = i
                state = OUTSIDE

        elif state == OUTSIDE:
            if c.isspace():
                pieces.append(s[begin:i])
                state = SPACE
            elif c == delim:
                state = INSIDE

        else: # state == INSIDE
            if c == delim:
                state = OUTSIDE

        i += inc

    if state != SPACE:
        pieces.append(s[begin:])
    return pieces

class OVFValueUnits:
    def __init__(self, s):
        self.units = s.split() if type(s) == str else s

    def __str__(self):
        return " ".join(self.units)

class OVFValueLabels:
    def __init__(self, s):
        self.labels = split_strings(s) if type(s) == str else s

    def __str__(self):
        return " ".join(['"%s"' % l for l in self.labels])

# List of known values in OOMMF file. Each entry is a tuple specifying:
# (the name of the field, the type of the field, a description,
#  supported version of OVF, default value)
# Version and context are potional. If the version is omitted,
# ANY_OVF is assumed, if the context is omitted, the value is assumed to
# belong to the header.
known_values_list = [
  ("Segment count", int, "Number of segments in data file", ANY_OVF, "root"),
  ("Title", str, "Title/long filename of the file"),
  ("meshtype", str, "Mesh type"),
  ("meshunit", str, "Fundamental mesh measurement unit"),
  ("xmin", float, "Minimum x coordinate of the mesh"),
  ("ymin", float, "Minimum y coordinate of the mesh"),
  ("zmin", float, "Minimum z coordinate of the mesh"),
  ("xmax", float, "Maximum x coordinate of the mesh"),
  ("ymax", float, "Maximum y coordinate of the mesh"),
  ("zmax", float, "Maximum z coordinate of the mesh"),
  ("valuedim", int, "Dimension of the data", OVF20),
  ("valuelabels", OVFValueLabels,
   "Labels for each dimension of the field", OVF20),
  ("valueunit", str, "Units for data values", OVF10),
  ("valueunits", OVFValueUnits,
   "Units for each dimension of the field.", OVF20),
  ("valuemultiplier", float,
   "Multiply data values by this to get true value in valueunit-s", OVF10),
  ("ValueRangeMaxMag", float, "Maximum value of data (used as hint)", OVF10),
  ("ValueRangeMinMag", float, "Minimum value of data (used as hint)", OVF10),
  ("Desc", str, "Extra lines used by postprocessing programs"),
  ("xbase", float, "x coordinate of first point in data section"),
  ("ybase", float, "y coordinate of first point in data section"),
  ("zbase", float, "z coordinate of first point in data section"),
  ("xnodes", int, "Number of cells along x dimension in the mesh"),
  ("ynodes", int, "Number of cells along y dimension in the mesh"),
  ("znodes", int, "Number of cells along z dimension in the mesh"),
  ("xstepsize", float, "Distance between adjacent grid points."),
  ("ystepsize", float, "Distance between adjacent grid points."),
  ("zstepsize", float, "Distance between adjacent grid points."),
  ("boundary", str, "List of (x, y, z) triples specifying the vertices of a "
                    "boundary frame. Optional.", OVF10)
  # ^^^ I didn't find any examples of what this looks like. I then use str
  #     for the type.
]

# Build a dictionary corresponding to known_values_list
known_values = {}
for known_value_tuple in known_values_list:
    value_name = name_normalise(known_value_tuple[0])
    known_values[value_name] = known_value_tuple

class OVFVersionNode(OVFNode):
    def write(self, stream, root=None):
        v = self.value
        if v != None:
            stream.write_line("# %s" % self.value)

class OVFSegmentSectionNode(OVFSectionNode):
    required = ["Header"]

class OVFHeaderSectionNode(OVFSectionNode):
    pass

def _info_binary(oommf_version, data_size):
    endianness = '!' if oommf_version == OVF10 else '<'
    if data_size == 8:
        float_type = 'd'
        expected_tag = 123456789012345.0

    else:
        assert data_size == 4
        float_type = 'f'
        expected_tag = 1234567.0
    return endianness, float_type, expected_tag

class OVFDataSectionNode(OVFSectionNode):
    def __init__(self, value=[], data=None):
        OVFSectionNode.__init__(self, value, data)
        self.mesh_type = None
        self.data_type = None
        self.nodes = None
        self.num_nodes = None
        self.num_stored_nodes = None
        self.floats_per_node = None
        self.field = None

    def _get_identity(self):
        return "data"

    identity = property(_get_identity)

    def _retrieve_info_from_root(self, root):
        h = root.a_segment.a_header
        xn, yn, zn = self.nodes = \
          (h.a_xnodes.value, h.a_ynodes.value, h.a_znodes.value)
        self.num_nodes = xn*yn*zn

        field_dim = root.field_dim
        self.mesh_type = root.mesh_type
        if self.mesh_type == "rectangular":
            self.floats_per_node = field_dim
            self.num_stored_nodes = self.num_nodes

        else:
            assert self.mesh_type == "irregular"
            self.floats_per_node = 3 + field_dim
            self.num_stored_nodes = h.a_pointcount

        self.data_type = name_normalise(self.name)

    def read(self, stream, root=None):
        self._retrieve_info_from_root(root)

        if self.data_type == 'databinary8':
            self._read_binary(stream, root=root, data_size=8)
        elif self.data_type == 'databinary4':
            self._read_binary(stream, root=root, data_size=4)
        elif self.data_type == 'datatext':
            self._read_ascii(stream, root=root)
        else:
            raise OVFReadError("Unknown data type '%s' in OVF file."
                               % self.name)

        # Get to end of section
        while True:
            l = stream.next_line()
            if l.startswith("# End:"):
                return

    def _read_binary(self, stream, root=None, data_size=8):
        endianness, float_type, expected_tag = \
          _info_binary(root.a_oommf.value.version, data_size)

        fmt = endianness + float_type
        verification_tag, = struct.unpack(fmt, stream.read_bytes(data_size))
        if verification_tag != expected_tag:
            raise OVFReadError("Data carries wrong signature: got '%s' but "
                               "'%s' was expected. This usually means that "
                               "the file is corrupted or is not being red "
                               "correctly."
                               % (verification_tag, expected_tag))

        num_floats = self.num_stored_nodes*self.floats_per_node
        fmt = endianness + float_type*num_floats
        data = stream.read_bytes(num_floats*data_size)
        big_float_tuple = struct.unpack(fmt, data)

        # Reshape the data
        xn, yn, zn = self.nodes
        fn = self.floats_per_node
        self.field = array(big_float_tuple).reshape((fn, xn, yn, zn),
                                                    order="F")

    def _read_ascii(self, stream, root=None):
        semiflat_array = ndarray(dtype='float', order="F",
                                 shape=(self.floats_per_node, self.num_nodes))

        for i in range(self.num_nodes):
            l = stream.next_line()
            v = [float(vi) for vi in l.split()]
            semiflat_array[:, i] = v

        # Reshape the data
        xn, yn, zn = self.nodes
        fn = self.floats_per_node
        self.field = semiflat_array.reshape((fn, xn, yn, zn))

    def write(self, stream, root=None):
        self._retrieve_info_from_root(root)

        stream.write_line("# Begin: %s" % self.name)
        if self.data_type == "databinary8":
            self._write_binary(stream, root=root, data_size=8)
        elif self.data_type == "databinary4":
            self._write_binary(stream, root=root, data_size=4)
        elif self.data_type == "datatext":
            self._write_ascii(stream, root=root)
        else:
            raise ValueError("Unrecognised data type '%s'"
                             % self.data_type)
        stream.write_line("# End: %s" % self.name)

    def _write_binary(self, stream, root=None, data_size=8):
        endianness, float_type, expected_tag = \
          _info_binary(root.a_oommf.value.version, data_size)

        fmt = endianness + float_type
        out_data = struct.pack(fmt, expected_tag)

        num_floats = self.num_stored_nodes*self.floats_per_node
        fmt = endianness + float_type*num_floats
        flat_array = self.field.ravel('F')
        out_data += struct.pack(fmt, *flat_array) + "\n"
        stream.write(out_data)

    def _write_ascii(self, stream, root=None):
        semiflat_array = \
          self.field.reshape((self.floats_per_node, self.num_nodes))
        for i in range(self.num_nodes):
            v = semiflat_array[:, i]
            stream.write_line(" ".join([repr(vi) for vi in v]))

def remove_comment(line, marker="##"):
    """Return the given line, without the part which follows the comment
    marker ## (and without the marker itself)."""
    i = line.find(marker)
    if i < 0:
        return line
    else:
        return line[:i]

def version_node(ver_str):
    return OVFVersionNode(data=("oommf", OVFType(ver_str)))

def known_value_node(name, value):
    lname = name_normalise(name)
    if known_values.has_key(lname):
        val_type = known_values[lname][1]
        value = val_type(value)

    else:
        print "Unknown value '%s' while reading OVF file." % name

    return OVFValueNode(data=(name, value))

def known_section_node(action, name):
    lname = name_normalise(name)

    cls = None
    if lname == "segment":
        cls = OVFSegmentSectionNode
    elif lname == "header":
        cls = OVFHeaderSectionNode
    elif lname.startswith("data"):
        cls = OVFDataSectionNode
    else:
        print "Unknown section '%s' while reading OVF file." % name
        cls = OVFSectionNode

    return cls(data=(name, action))

def read_node(stream):
    l = None
    while l in ["", "#", None]:
        l = stream.next_line()
        if l == None:
            return None
        else:
            l = remove_comment(l).lstrip()

    if not l.startswith("#"):
        raise OVFReadError("Error reading OVF header. "
                           "Expected #, but got '%s'" % l)

    l = l[1:].lstrip()
    ll = l.lower()
    if ll.startswith("oommf"):
        return version_node(ll)

    else:
        piece = l.split(":", 1)
        name = piece[0].strip()
        lname = name_normalise(name)

        value = None
        if len(piece) > 1:
            value = piece[1].strip()

        if lname in ["begin", "end"]:
            return known_section_node(name, value)
        else:
            return known_value_node(name, value)

class OVFRootNode(OVFSectionNode):
    required = ["Segment count"]

    def __init__(self):
        OVFSectionNode.__init__(self, data=("main", "begin"))

    def _get_version(self):
        return self.a_oommf.value.version

    ovf_version = property(_get_version, None, None, "Version of OVF file.")

    def _get_mesh_type(self):
        v = self.ovf_version
        if v == OVF10:
            return self.a_oommf.value.mesh_type
        else:
            return self.a_segment.a_header.a_meshtype.value

    mesh_type = property(_get_mesh_type, None, None,
                         "Mesh type of the OVF file "
                         "(a string = rectangular/irregular)")

    def _get_field_dim(self):
        if self.ovf_version == OVF10:
            return 3
        else:
            return self.a_segment.a_header.a_valuedim.value

    field_dim = property(_get_field_dim, None, None, "The size of the field.")

    def _get_valueunits(self):
        v = self.ovf_version
        if v == OVF10:
            return self.a_segment.a_header.a_valueunit.value
        else:
            return self.a_segment.a_header.a_valueunits.value

    def _set_valueunits(self, units):
        units = [units] if type(units) == str else units
        v = self.ovf_version
        if v == OVF10:
            units_are_all_the_same = units.count(units[0]) == len(units)
            assert units_are_all_the_same, \
              ("OVF 1.0 does not support fields having components with "
               "different units.")
            self.a_segment.a_header.a_valueunit.value = str(units[0])
        else:
            assert v == OVF20
            def unit_setter(idx):
                return units[idx] if idx < len(units) else units[-1]
            us = [unit_setter(idx) for idx in range(self.field_dim)]
            self.a_segment.a_header.a_valueunits.value = OVFValueUnits(us)

    valueunits = property(_get_valueunits, _set_valueunits, None,
                          "The units of the components of the field "
                          "(one string or a list of strings).")

    def _get_valuelabels(self):
        v = self.ovf_version
        if v == OVF20:
            return self.a_segment.a_header.a_valuelabels.value
        else:
            raise OVFVersionError("valuelabels is only available in OVF 2.0.")

    def _set_valuelabels(self, labels):
        v = self.ovf_version
        if v == OVF20:
            if type(labels) == str:
                labels = ["%s_%d" % (labels, i)
                          for i in range(self.field_dim)]
            self.a_segment.a_header.a_valuelabels.value = \
              OVFValueLabels(labels)

    valuelabels = property(_get_valuelabels, _set_valuelabels, None,
                           "The labels for the components of the field "
                           "(a list of names, one for each component. when "
                           "setting you can provide also a string, used as "
                           "the basename of the field: _0, _1, _2, ... are "
                           "appended to each component).")

    def write(self, stream, root=None):
        for n in self._subnodes:
            n.write(stream, root=root)

class OVFStream(object):
    def __init__(self, filename, mode="r"):
        if type(filename) == str:
            self.filename = filename
            self.f = open(filename, mode)
        else:
            self.filename = None
            self.f = filename
        self.no_line = 0
        self.lines = []

    def __del__(self):
        self.f.close()

    def next_line(self):
        if self.no_line < len(self.lines):
            l = self.lines[self.no_line]

        else:
            n = self.no_line - len(self.lines)
            for _ in range(n + 1):
                l = self.f.readline()
                if len(l) == 0:
                    return None
                l = l[:-1]
                self.lines.append(l)

        self.no_line += 1
        return l

    def read_bytes(self, num_bytes):
        l = self.f.read(num_bytes)
        self.lines.append(l)
        self.no_line = len(self.lines)
        return l

    def read_lines_ahead(self):
        self.lines += self.f.readlines()

    def write(self, data):
        self.f.write(data)

    def write_line(self, line):
        self.f.write(line + "\n")

class OVFFile:
    def __init__(self, filename=None):
        self.content = OVFRootNode()

        if filename != None:
            self.read(filename)

    def new(self, fieldlattice, version=OVF10, mesh_type="rectangular",
            data_type="binary8"):

        available_data_types = {"text":"Data Text",
                                "binary4": "Data Binary 4",
                                "binary8": "Data Binary 8"}
        if available_data_types.has_key(data_type):
            data_type = available_data_types[data_type]

        else:
            available_choices = ", ".join(available_data_types.keys())
            raise ValueError("Wrong choice of data_type. Available choices "
                             "are: %s." % available_choices)

        assert mesh_type == "rectangular", "Irregular meshes are not " \
                                           "supported, yet!"

        assert fieldlattice.lattice.order == "F", \
          "FieldLattice should have Fortran ordering!"

        assert fieldlattice.lattice.dim == 3, \
          "The FieldLattice should be defined over a 3D mesh."

        # Generate the root node
        root_node = OVFRootNode()

        # Append version info
        if version == OVF10:
            t = OVFType("OOMMF: %s mesh v1.0" % mesh_type)
            assert fieldlattice.field_dim == 3, \
              ("OVF 1.0 only supports fields with dimension 3 (such as "
               "the magnetisation, for example)")
        else:
            t = OVFType("OOMMF OVF 2.0")
            assert fieldlattice.field_dim >= 1, \
              ("You are trying to write a field with dimension 0.")
        root_node._subnodes.append(OVFVersionNode(data=("OOMMF", t)))

        # Append segment count and segment section
        root_node._subnodes.append(OVFValueNode(data=("Segment count", "1")))
        segment_node = OVFSectionNode(data=("Segment", "Begin"))
        root_node._subnodes.append(segment_node)

        # Generate the header
        header_node = OVFSectionNode(data=("Header", "Begin"))
        segment_node._subnodes.append(header_node)
        for known_v in known_values_list:
            v_name = known_v[0]
            v_type = known_v[1]
            v_ver = known_v[3] if len(known_v) > 3 else ANY_OVF
            v_context = known_v[4] if len(known_v) > 4 else "header"
            if v_context == "header" and \
               (v_ver == ANY_OVF or v_ver == version):
                v_node = OVFValueNode(data=(v_name, None))
                header_node._subnodes.append(v_node)
        header_node._subnodes.append(OVFSectionNode(data=("Header", "End")))

        # Generate the data segment
        fl = fieldlattice
        l = fieldlattice.lattice
        data_node = OVFDataSectionNode(data=(data_type, "Begin"))
        segment_node._subnodes.append(data_node)
        data_node._subnodes.append(OVFSectionNode(data=(data_type, "End")))
        data_node.field = fl.field_data

        segment_node._subnodes.append(OVFSectionNode(data=("Segment", "End")))

        # Add subnodes as attributes for better accessibility
        root_node._add_as_attr()

        # Now write proper values in the header fields
        h = root_node.a_segment.a_header
        h.a_xnodes.value, h.a_ynodes.value, h.a_znodes.value = l.nodes
        ss = l.stepsizes
        hss = [0.5*ssi for ssi in ss]
        h.a_xstepsize.value, h.a_ystepsize.value, h.a_zstepsize.value = ss
        h.a_xbase.value, h.a_ybase.value, h.a_zbase.value = hss
        min_mesh_pos = [nmn - d for nmn, d in zip(l.min_node_pos, hss)]
        max_mesh_pos = [nmx + d for nmx, d in zip(l.max_node_pos, hss)]
        h.a_xmin.value, h.a_ymin.value, h.a_zmin.value = min_mesh_pos
        h.a_xmax.value, h.a_ymax.value, h.a_zmax.value = max_mesh_pos

        # Final "decorations"
        h.a_title.value = "Title"
        h.a_meshtype.value = mesh_type
        h.a_meshunit.value = "1.0"
        if version == OVF10:
            h.a_valueunit.value = "1.0"
            h.a_valuemultiplier.value = 1.0
            h.a_valuerangeminmag.value = min(fl.field_data.flat)
            h.a_valuerangemaxmag.value = max(fl.field_data.flat)

        else:
            h.a_valuedim.value = fl.field_dim
            h.a_valueunits.value = OVFValueUnits(" 1.0"*fl.field_dim)
            h.a_valuelabels.value = OVFValueLabels(' "1.0"'*fl.field_dim)

        # Finally replace self.content
        self.content = root_node

    def get_field(self):
        root_node = self.content
        segment_node = root_node.a_segment
        h = segment_node.a_header
        ss = [h.a_xstepsize, h.a_ystepsize, h.a_zstepsize]
        dx, dy, dz = [0.5*ssi.value for ssi in ss]

        min_max_ndim = \
          [(h.a_xmin.value - dx, h.a_xmax.value + dx, h.a_xnodes.value),
           (h.a_ymin.value - dy, h.a_ymax.value + dy, h.a_ynodes.value),
           (h.a_zmin.value - dz, h.a_zmax.value + dz, h.a_znodes.value)]

        field_data = segment_node.a_data.field
        field_dim = root_node.field_dim
        return FieldLattice(min_max_ndim, dim=field_dim,
                            data=field_data, order='F')

    def read(self, stream):
        if not isinstance(stream, OVFStream):
            stream = OVFStream(stream)
        self.content.read(stream, root=self.content)
        self.content._end_section("main")

    def write(self, stream):
        if not isinstance(stream, OVFStream):
            stream = OVFStream(stream, mode="w")
        self.content.write(stream, root=self.content)

if __name__ == "__main__no":
    import sys
    print "Reading"
    ovf = OVFFile(sys.argv[1])
    print "Writing"
    #ovf.content.a_segment.a_databinary8.name = "Data Binary 4"
    ovf.write(sys.argv[2])
    print "Done"

elif __name__ == "__main__":
    # Here is how to create an OVF file from a FieldLattice object
    fl = FieldLattice("2.5e-9,97.5e-9,20/2.5e-9,47.5e-9,10/2.5e-9,7.5e-9,1",
                      order="F")
    fl.set(lambda pos: [1, 0, 0])
    ovf = OVFFile()
    ovf.new(fl, version=OVF20, data_type="binary8")
    ovf.content.a_segment.a_header.a_title = "MyFile"
    ovf.write("new-v1.ovf")

