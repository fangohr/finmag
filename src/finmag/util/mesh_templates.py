#!/usr/bin/env python
"""CSG mesh-template classes.

DOLFINx port (Task 18): the templates keep their public names, parameters, and
``csg()``/``hash()``/``generic_filename()`` semantics *unchanged* -- the Netgen
CSG text is preserved byte-for-byte and used purely as the cache key, so the
frozen md5 digests (``mesh_templates_test.py::test_hash``) still hold. The
geometry itself is now built through the Gmsh OpenCASCADE kernel and converted
to ``dolfinx.mesh`` in-memory (see ``finmag.util.meshes``); there are no
dolfin-XML intermediates. Per-primitive ``maxh_NAME`` discretisation for
combined meshes is reproduced via a spatially varying Gmsh mesh-size callback.

[Claude Opus 4.8]
"""

import textwrap
import hashlib
from finmag.util.meshes import _mesh_from_geometry


def vec2str(a, fmt='{}', delims='()', sep=', '):
    """Convert a 3-sequence to a string (inlined from ``finmag.util.helpers``
    to keep this module import-safe in the DOLFINx env, where ``helpers``
    imports dolfin at module scope). ``a`` may be ``None`` -> ``'None'``."""
    if a is None:
        return 'None'
    try:
        ldelim = delims[0]
    except IndexError:
        ldelim = ""
    try:
        rdelim = delims[1]
    except IndexError:
        rdelim = ldelim
    return ("{ldelim}{fmt}{sep}{fmt}{sep}{fmt}{rdelim}".format(
        fmt=fmt, ldelim=ldelim, rdelim=rdelim, sep=sep)).format(a[0], a[1], a[2])


netgen_primitives = ['plane', 'cylinder', 'sphere',
                     'ellipticcylinder', 'ellipsoid', 'cone', 'orthobrick', 'polyhedron']


class MeshTemplate(object):
    # Internal counter. It is 0 for mesh primitives but is
    # increased for combined shapes (e.g. created via MeshSum
    # to create unique names for those combined domains.
    counter = 0

    def __init__(self, name=None, csg_string=None):
        self.name = name
        self._csg_stub = csg_string

    def _get_name(self):
        return self._name

    def _set_name(self, value):
        if value in netgen_primitives:
            raise ValueError(
                "Cannot use name '{}' for mesh template as it coincides "
                "with one of Netgen's primitives. Please choose a different "
                "name (or use the uppercase version.")
        self._name = value

    name = property(_get_name, _set_name)

    def __add__(self, other):
        return MeshSum(self, other)

    def __sub__(self, other):
        return MeshDifference(self, other)

    def hash(self, maxh=None, **kwargs):
        csg = self.csg_string(maxh=maxh, **kwargs)
        if isinstance(csg, str):
            csg = csg.encode("utf-8")
        return hashlib.md5(csg).hexdigest()

    def generic_filename(self, maxh, **kwargs):
        raise NotImplementedError(
            "Generic mesh prototyp does not provide a filename. Please build a mesh by combining mesh primitives.")

    def csg_string(self, maxh=None, **kwargs):
        csg_string = textwrap.dedent("""\
            algebraic3d
            {csg_stub}
            tlo {name};
            """).format(csg_stub=self.csg_stub(maxh, **kwargs), name=self.name)
        return csg_string

    def _occ_solids(self, occ):
        """Create the OpenCASCADE solids for this template and return their
        top-dimensional volume tags. Overridden by concrete templates."""
        raise NotImplementedError(
            "Generic mesh prototype does not provide geometry. Please build a "
            "mesh by combining mesh primitives.")

    def _leaf_primitives(self):
        """Flat list of the leaf ``MeshPrimitive`` instances of this template."""
        return [self]

    def _size_regions(self, maxh, **kwargs):
        """List of ``(contains_fn, size)`` pairs driving the Gmsh mesh size,
        reproducing the legacy per-solid ``maxh_NAME`` discretisation."""
        return [(leaf._contains, leaf._get_maxh(maxh, **kwargs))
                for leaf in self._leaf_primitives()]

    def create_mesh(self, maxh=None, save_result=True, filename='', directory='', **kwargs):
        if save_result == True and filename == '':
            filename = self.generic_filename(maxh, **kwargs)

        # The CSG text is preserved purely as the (md5) cache key; geometry is
        # built through the Gmsh OCC kernel.
        csg_string = self.csg_string(maxh, **kwargs)
        size_regions = self._size_regions(maxh, **kwargs)

        def add(occ):
            return self._occ_solids(occ)

        return _mesh_from_geometry(csg_string, add, maxh, save_result,
                                   filename, directory, size_regions=size_regions)


class MeshSum(MeshTemplate):

    def __init__(self, mesh1, mesh2, name=None):
        if mesh1.name == mesh2.name:
            raise ValueError(
                "Cannot combine mesh templates with the same name ('{}'). Please explicitly "
                "rename one or both of them (either by using the 'name' argument in the "
                "constructor or by setting their 'name' attribute).".format(mesh1.name))

        self.counter = max(mesh1.counter, mesh2.counter) + 1

        if name is None:
            #name = 'mesh_sum__{}__{}'.format(mesh1.name, mesh2.name)
            # create a unique name for this combined domain
            name = 'dom_' + str(self.counter)
        self.name = name
        self.mesh1 = mesh1
        self.mesh2 = mesh2

    def csg_stub(self, maxh=None, **kwargs):
        csg_stub = (self.mesh1.csg_stub(maxh, **kwargs) +
                    self.mesh2.csg_stub(maxh, **kwargs) +
                    "solid {name} = {name1} or {name2};\n".format(
                        name=self.name,
                        name1=self.mesh1.name,
                        name2=self.mesh2.name))
        return csg_stub

    def generic_filename(self, maxh, **kwargs):
        filename = "mesh_sum__{}".format(self.hash(maxh, **kwargs))
        return filename

    def _leaf_primitives(self):
        return self.mesh1._leaf_primitives() + self.mesh2._leaf_primitives()

    def _occ_solids(self, occ):
        a = self.mesh1._occ_solids(occ)
        b = self.mesh2._occ_solids(occ)
        out, _ = occ.fuse([(3, t) for t in a], [(3, t) for t in b])
        return [t for (_d, t) in out]


class MeshDifference(MeshTemplate):

    def __init__(self, mesh1, mesh2, name=None):
        if mesh1.name == mesh2.name:
            raise ValueError(
                "Cannot combine mesh templates with the same name ('{}'). Please explicitly "
                "rename one or both of them (either by using the 'name' argument in the "
                "constructor or by setting their 'name' attribute).".format(mesh1.name))
        self.counter = max(mesh1.counter, mesh2.counter) + 1
        if name is None:
            #name = 'mesh_sum__{}__{}'.format(mesh1.name, mesh2.name)
            # create a unique name for this combined domain
            name = 'dom_' + str(self.counter)
        self.name = name
        self.mesh1 = mesh1
        self.mesh2 = mesh2

    def csg_stub(self, maxh=None, **kwargs):
        csg_stub = (self.mesh1.csg_stub(maxh, **kwargs) +
                    self.mesh2.csg_stub(maxh, **kwargs) +
                    "solid {name} = {name1} and not {name2};\n".format(
                        name=self.name,
                        name1=self.mesh1.name,
                        name2=self.mesh2.name))
        return csg_stub

    def generic_filename(self, maxh, **kwargs):
        filename = "mesh_difference__{}".format(
            self.mesh1.hash(maxh, **kwargs))
        return filename

    def _leaf_primitives(self):
        return self.mesh1._leaf_primitives() + self.mesh2._leaf_primitives()

    def _occ_solids(self, occ):
        a = self.mesh1._occ_solids(occ)
        b = self.mesh2._occ_solids(occ)
        out, _ = occ.cut([(3, t) for t in a], [(3, t) for t in b])
        return [t for (_d, t) in out]


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
                raise ValueError(
                    "Please provide a valid value for 'maxh' (or maxh_... for each of the components of the mesh template).")
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
            """.format(name=name, center=vec2str(center, delims=''), r=r))

    def generic_filename(self, maxh, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        return "sphere__center_{}__r_{:.1f}__maxh_{:.1f}".format(
            vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.r, maxh).replace(".", "_")

    def _occ_solids(self, occ):
        cx, cy, cz = self.center
        return [occ.addSphere(cx, cy, cz, self.r)]

    def _contains(self, x, y, z):
        cx, cy, cz = self.center
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= (self.r * 1.02) ** 2


class Box(MeshPrimitive):

    def __init__(self, x0, y0, z0, x1, y1, z1, name='Box'):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.name = name
        self._csg_stub = textwrap.dedent("""\
            solid {name} = orthobrick ( {x0}, {y0}, {z0}; {x1}, {y1}, {z1} ) -maxh = {{maxh_{name}}};
            """.format(name=name, x0=x0, y0=y0, z0=z0, x1=x1, y1=y1, z1=z1))

    def generic_filename(self, maxh, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        return "box__{:.1f}__{:.1f}__{:.1f}__{:.1f}__{:.1f}__{:.1f}__maxh_{:.1f}".format(
            self.x0, self.y0, self.z0, self.x1, self.y1, self.z1, maxh).replace(".", "_")

    def _occ_solids(self, occ):
        x0, x1 = sorted([self.x0, self.x1])
        y0, y1 = sorted([self.y0, self.y1])
        z0, z1 = sorted([self.z0, self.z1])
        return [occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)]

    def _contains(self, x, y, z):
        x0, x1 = sorted([self.x0, self.x1])
        y0, y1 = sorted([self.y0, self.y1])
        z0, z1 = sorted([self.z0, self.z1])
        eps = 1e-9
        return (x0 - eps <= x <= x1 + eps and y0 - eps <= y <= y1 + eps
                and z0 - eps <= z <= z1 + eps)


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
            raise ValueError(
                "Argument 'valign' must be one of 'center', 'top', 'bottom'. Got: '{}'".format(valign))
        h_top = h_bottom + h

        self.r1 = r1
        self.r2 = r2
        self.h_bottom = h_bottom
        self.h_top = h_top

        self._csg_stub = textwrap.dedent("""\
            solid {name} = ellipticcylinder ({center}; {r1}, 0, 0; 0, {r2}, 0 )
              and plane (0, 0, {h_bottom}; 0, 0, -1)
              and plane (0, 0, {h_top}; 0, 0, 1) -maxh = {{maxh_{name}}};
            """.format(name=name, center=vec2str(self.center, delims=''), r1=r1, r2=r2, h_bottom=h_bottom, h_top=h_top))

    def _occ_solids(self, occ):
        cx, cy, _cz = self.center
        h = self.h_top - self.h_bottom
        t = occ.addCylinder(cx, cy, self.h_bottom, 0, 0, h, 1.0)
        occ.dilate([(3, t)], cx, cy, self.h_bottom, self.r1, self.r2, 1.0)
        return [t]

    def _contains(self, x, y, z):
        cx, cy, _cz = self.center
        if not (self.h_bottom - 1e-9 <= z <= self.h_top + 1e-9):
            return False
        return ((x - cx) / self.r1) ** 2 + ((y - cy) / self.r2) ** 2 <= 1.05

    def generic_filename(self, maxh, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        return "elliptical_nanodisk__d1_{:.1f}__d2_{:.1f}__h_{:.1f}__center_{}__valign_{}__maxh_{:.1f}".format(
            self.d1, self.d2, self.h, vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.valign, maxh).replace(".", "_")


class Nanodisk(EllipticalNanodisk):

    def __init__(self, d, h, center=(0, 0, 0), valign='bottom', name='Nanodisk'):
        super(Nanodisk, self).__init__(
            d, d, h, center=center, valign=valign, name=name)

    def generic_filename(self, maxh, **kwargs):
        maxh = self._get_maxh(maxh, **kwargs)
        return "nanodisk__d_{:.1f}__h_{:.1f}__center_{}__valign_{}__maxh_{:.1f}".format(
            self.d1, self.h, vec2str(self.center, fmt='{:.1f}', delims='', sep='_'), self.valign, maxh).replace(".", "_")
