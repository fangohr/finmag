#!/usr/bin/env python

import textwrap
from finmag.util.meshes import from_csg
from finmag.util.helpers import vec2str


class MeshPrototype(object):
    def __init__(self, name=None, csg_string=None):
        self.name = name
        self._csg_stub = csg_string

    def __add__(self, other):
        return MeshSum(self, other)

    def generic_filename(self, maxh, **kwargs):
        raise NotImplementedError("Generic mesh prototyp does not provide a filename. Please build a mesh by combining mesh primitives.")

    def csg_string(self, maxh=None, **kwargs):
        csg_string = textwrap.dedent("""\
            algebraic3d
            {}
            """).format(self.csg_stub(maxh, **kwargs))
        return csg_string

    def create_mesh(self, maxh=None, save_result=True, filename='', directory='', **kwargs):
        if save_result == True and filename == '':
            filename = self.generic_filename(maxh, **kwargs)

        csg_string = self.csg_string(maxh, **kwargs)
        return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)


class MeshSum(MeshPrototype):
    def __init__(self, mesh1, mesh2, name=None):
        if mesh1.name == mesh2.name:
            raise ValueError(
                "Cannot add mesh prototypes with the same name ('{}'). Please explicitly "
                "rename one or both of them (either by using the 'name' argument in the "
                "constructor or by setting their 'name' attribute).".format(mesh1.name))
        if name is None:
            name = 'mesh_sum__{}__{}'.format(mesh1.name, mesh2.name)
        self.name = name
        self.mesh1 = mesh1
        self.mesh2 = mesh2

    def csg_stub(self, maxh=None, **kwargs):
        csg_stub = self.mesh1.csg_stub(maxh, **kwargs) + self.mesh2.csg_stub(maxh, **kwargs)
        return csg_stub

    def generic_filename(self, maxh, **kwargs):
        filename = "mesh_sum__{}__{}".format(self.mesh1.generic_filename(maxh, **kwargs), self.mesh2.generic_filename(maxh, **kwargs))
        return filename


class MeshPrimitive(MeshPrototype):
    def _get_maxh(self, maxh=None, **kwargs):
        """
        If `kwargs` contains an item with key 'maxh_NAME' (where NAME
        is equal to self.name), returns this value and the associated key.
        Otherwise returns the value associated with the key 'maxh'.

        """
        try:
            key = 'maxh_' + self.name
            maxh = kwargs[key]
        except KeyError:
            if maxh == None:
                raise ValueError("Please provide a valid value for 'maxh' (got: None).")
        return maxh

    def csg_stub(self, maxh=None, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        key = 'maxh_{}'.format(self.name)
        fmtdict = {key: maxh}
        return self._csg_stub.format(**fmtdict)


class Sphere(MeshPrimitive):
    def __init__(self, r, center=(0, 0, 0), name='Sphere'):
        self.r = r
        self.center = center
        self.name = name
        self._csg_stub = textwrap.dedent("""\
            solid {name} = sphere ( {center}; {r} ) -maxh = {{maxh_{name}}};
            tlo {name};
            """.format(name=name, center=vec2str(center, delims=''), r=r))

    def generic_filename(self, maxh, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        return "sphere__center_{}__r_{:.1f}__maxh_{:.1f}".format(
            vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.r, maxh).replace(".", "_")
