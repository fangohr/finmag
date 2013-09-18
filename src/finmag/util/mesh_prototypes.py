#!/usr/bin/env python

import textwrap
from finmag.util.meshes import from_csg
from finmag.util.helpers import vec2str


class MeshPrototype(object):
    def __init__(self):
        self.csg_string = None

    def generic_filename(self):
        raise NotImplementedError("MeshPrototype does not provide a generic filename. Please use one of the specific mesh prototype classes instead (e.g. Sphere, Nanodisk)")

    def create_mesh(self, maxh, save_result=True, filename='', directory=''):
        if not self.csg_string:
            raise NotImplementedError

        if save_result == True and filename == '':
            filename = self.generic_filename(maxh)

        return from_csg(self.csg_string.format(maxh=maxh), save_result=save_result, filename=filename, directory=directory)


class Sphere(MeshPrototype):
    def __init__(self, r, center=(0, 0, 0)):
        self.r = r
        self.center = center
        self.csg_string = textwrap.dedent("""\
            algebraic3d
            solid main = sphere ( {center}; {r} ) -maxh = {{maxh}};
            tlo main;
            """).format(center=vec2str(center, delims=''), r=r)

    def generic_filename(self, maxh):
        return "sphere__center_{}__r_{:.1f}__maxh_{:.1f}".format(
                   vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.r, maxh).replace(".", "_")
