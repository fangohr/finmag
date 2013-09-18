#!/usr/bin/env python

import textwrap
from finmag.util.meshes import from_csg
from finmag.util.helpers import vec2str


class MeshPrototype(object):
    def __init__(self, name=None, csg_string=None):
        self._csg_stub = csg_string
        if isinstance(name, list):
            self.names = name
        else:
            self.names = name or []

    def __add__(self, other):
        csg_string_combined = self._csg_stub + other._csg_stub
        duplicate_names = [n for n in self.names if n in other.names]
        if duplicate_names != []:
            raise ValueError(
                "Mesh prototypes to be added must not contain duplicate names. "
                "Please name the individual prototypes explicitly (using the "
                "'name' argument in the constructor). "
                "Duplicate names found: {}".format(duplicate_names))
        names = self.names + other.names
        return MeshPrototype(name=names, csg_string=csg_string_combined)

    def generic_filename(self, maxh):
        filename = "generic_mesh"
        for n in self.names:
            filename += ("_" + n)
        return filename

    def csg_string(self):
        if not self._csg_stub:
            raise NotImplementedError

        csg = textwrap.dedent("""\
            algebraic3d
            {}
            """).format(self._csg_stub)
        for name in self.names:
            csg += "tlo {};\n".format(name)

        return csg

    def create_mesh(self, maxh, save_result=True, filename='', directory=''):
        if save_result == True and filename == '':
            filename = self.generic_filename(maxh)

        return from_csg(self.csg_string().format(maxh=maxh), save_result=save_result, filename=filename, directory=directory)


class Sphere(MeshPrototype):
    def __init__(self, r, center=(0, 0, 0), name='Sphere'):
        self.r = r
        self.center = center
        self.names = [name]
        self._csg_stub = "solid {name} = sphere ( {center}; {r} ) -maxh = {{maxh}};\n".format(name=name, center=vec2str(center, delims=''), r=r)

    def generic_filename(self, maxh):
        return "sphere__center_{}__r_{:.1f}__maxh_{:.1f}".format(
            vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.r, maxh).replace(".", "_")
