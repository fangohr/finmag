import re


def convert(nmesh_file, xml_file):
    """
    Convert a mesh from nmesh ASCII format to dolfin xml format.

    Note that region information is read from nmesh, but then discarded.
    You would need to extend the Emitter class to output a second xml file
    if you wanted to preserve the region information.
    """
    p = Parser(nmesh_file)
    e = Emitter(xml_file)

    nb_vertices = p.vertices()
    e.vertices(nb_vertices)

    for i in range(int(nb_vertices)):
        e.vertex(p.vertex())

    nb_simplices = p.simplices()
    e.simplices(nb_simplices)

    for i in range(int(nb_simplices)):
        e.simplex(p.simplex())

    p.done()
    e.done()


class Parser(object):
    """
    Encapsulates what we know about nmesh ASCII files.

    """
    def __init__(self, nmesh_file):
        """
        Create a parser for the nmesh ASCII file in `nmesh_file`.

        """
        print "Reading from '{}'.".format(nmesh_file)
        self._fi = open(nmesh_file, 'r')

        self._version_string(self._next())
        dim, self._vertices, self._simplices = self._summary(self._next())
        print "Found a mesh with {} dimensions, {} nodes and {} tetrahedra.".format(
            dim, self._vertices, self._simplices)

    def vertices(self):
        """
        Returns the number of vertices in the mesh.

        """
        self._vertices_heading(self._next())
        return self._vertices

    def vertex(self):
        """
        Returns the x, y and z coordinates of the next vertex in the file.

        """
        return self._vertex(self._next())

    def simplices(self):
        """
        Returns the number of simplices (tetrahedra) in the mesh.

        """
        self._simplices_heading(self._next())
        return self._simplices

    def simplex(self):
        """
        Returns the tuple (region_number, v0, v1, v2, v3), where the vi
        are the indices of the vertices that make up the simplex.

        """
        return self._simplex(self._next())

    def done(self):
        """
        Allow the parser to perform necessary clean-up operations like
        closing its handle on the nmesh ASCII file.

        """
        self._fi.close()

    def _next(self):
        return self._fi.readline().rstrip()

    def _version_string(self, s):
        VERSION_STRING = "# PYFEM mesh file version 1.0"
        if not s == VERSION_STRING:
            raise ValueError("Version string should be '{}', is '{}'.".format(
                VERSION_STRING, s))

    def _summary(self, s):
        pattern = '^# dim = (\d)\s+nodes = (\d+)\s+simplices = (\d+)'
        match = re.compile(pattern).match(s)
        return match.groups()

    def _vertices_heading(self, s):
        if not s == self._vertices:
            raise ValueError("Info 'nodes = {}' in summary doesn't match header '{}'.".format(
                self._vertices, s))

    def _vertex(self, s):
        coords = s.split()
        if not len(coords) == 3:
            raise ValueError("Can't parse coordinates from string '{}'.".format(s))
        return coords

    def _simplices_heading(self, s):
        if not s == self._simplices:
            raise ValueError("Info 'simplices = {}' in summary doesn't match header '{}'.".format(
                self._simplices, s))

    def _simplex(self, s):
        try:
            region, v0, v1, v2, v3 = s.split()
        except ValueError as e:
            print "Expected line with region number followed by indices of four nodes, got '{}'.".format(s)
            raise
        return region, v0, v1, v2, v3


class Emitter(object):
    """
    Write a dolfin xml mesh file.

    """
    def __init__(self, xml_file):
        """
        Create the emitter.

        """
        self.vi = 0  # index of vertex
        self.si = 0  # index of simplex
        self._vertices = 0  # number of vertices in total
        self._simplices = 0  # number of simplices in total
        self._done_with_vertices = False
        self._done_with_simplices = False

        print "Writing to '{}'.".format(xml_file)
        self._fo = open(xml_file, 'w')

        # None of the python xml libraries I checked supports emitting chunks
        # of XML. They all wanted to have the full document in memory before
        # comitting it to disk. Since the meshfiles can be potentially very,
        # very long and the format is simple I'm using custom code to emit the
        # XML. I promise to offer a round of beer if this causes issues down
        # the line. (Offer expires 31.12.2019).

        self._write('<?xml version="1.0"?>')
        self._write('<dolfin xmlns:dolfin="http://fenicsproject.org">')
        self._write('  <mesh celltype="tetrahedron" dim="3">')

    def vertices(self, n):
        """
        Save the number `n` of vertices in the mesh.

        """
        self._vertices = int(n)
        s = '    <vertices size="{}">'
        self._write(s.format(n))

    def vertex(self, (x, y, z)):
        """
        Save a single vertex with the coordinates `x`, `y` and `z`.
        Indexed automatically.

        """
        if not self._done_with_vertices:
            s = '      <vertex index="{}" x="{}" y="{}" z="{}" />'
            self._write(s.format(self.vi, x, y, z))
            self.vi += 1
            if self.vi == self._vertices:
                self._close_vertices()
                self._done_with_vertices = True
        else:
            raise ValueError("Reached number of {} vertices already. Aborting.".format(self._vertices))

    def simplices(self, n):
        """
        Save the number of simplices (tetrahedra) in the mesh.

        """
        self._simplices = int(n)
        s = '    <cells size="{}">'
        self._write(s.format(n))

    def simplex(self, (region, v0, v1, v2, v3)):
        """
        Save a single simplex, identified by its vertices `v0`, `v1`, `v2` and `v3`.
        Region number is discarded as of now.
        Indexed automatically.

        """
        if not self._done_with_simplices:
            s = '      <tetrahedron index="{}" v0="{}" v1="{}" v2="{}" v3="{}" />'
            self._write(s.format(self.si, v0, v1, v2, v3))
            self.si += 1
            if self.si == self._simplices:
                self._close_simplices()
                self._close_all()
                self._done_with_simplices = True
        else:
            raise ValueError("Reached number of {} simplices already. Aborting.".format(self._simplices))

    def done(self):
        """
        Allow the parser to perform necessary clean-up operations like
        closing its handle on the xml file.

        """
        self._fo.close()

    def _write(self, s):
        self._fo.write(s + "\n")

    def _close_vertices(self):
        self._write('    </vertices>')

    def _close_simplices(self):
        self._write('    </cells>')

    def _close_all(self):
        self._write('  </mesh>')
        self._write('</dolfin>')
