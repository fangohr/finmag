#!/usr/bin/env python

import textwrap
from finmag.util.meshes import from_csg
from finmag.util.helpers import vec2str


class MeshTemplate(object):
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


class MeshSum(MeshTemplate):
    def __init__(self, mesh1, mesh2, name=None):
        if mesh1.name == mesh2.name:
            raise ValueError(
                "Cannot combine mesh templates with the same name ('{}'). Please explicitly "
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


class MeshPrimitive(MeshTemplate):
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


class EllipticalNanodisk(MeshPrimitive):
    def __init__(self, d1, d2, h, center=(0, 0, 0), valign='bottom', name='EllipticalNanodisk'):
        self.d1 = d1
        self.d2 = d2
        self.h = h
        self.center = center
        self.valign = valign
        self.name = name

        r1 = 0.5 * d1
        r2 = 0.5 * d2
        try:
            h_bottom = {'bottom': center[2],
                        'center': center[2] - 0.5 * h,
                        'top': center[2] - h,
                       }[valign]
        except KeyError:
            raise ValueError("Argument 'valign' must be one of 'center', 'top', 'bottom'. Got: '{}'".format(valign))
        h_top = h_bottom + h

        self._csg_stub = textwrap.dedent("""\
            solid {name} = ellipticcylinder ({center}; {r1}, 0, 0; 0, {r2}, 0 )
              and plane (0, 0, {h_bottom}; 0, 0, -1)
              and plane (0, 0, {h_top}; 0, 0, 1) -maxh = {{maxh_{name}}};
            tlo {name};
            """.format(center=vec2str(self.center, delims=''), r1=r1, r2=r2, h_bottom=h_bottom, h_top=h_top, name=name))

    def generic_filename(self, maxh, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        return "elliptical_nanodisk__d1_{:.1f}__d2_{:.1f}__h_{:.1f}__center_{}__valign_{}__maxh_{:.1f}".format(
            self.d1, self.d2, self.h, vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.valign, maxh).replace(".", "_")


class Nanodisk(EllipticalNanodisk):
    def __init__(self, d, h, center=(0, 0, 0), valign='bottom', name='Nanodisk'):
        super(Nanodisk, self).__init__(d, d, h, center=center, valign=valign, name=name)

    def generic_filename(self, maxh, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        return "nanodisk__d_{:.1f}__h_{:.1f}__center_{}__valign_{}__maxh_{:.1f}".format(
            self.d1, self.h, vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.valign, maxh).replace(".", "_")
