"""
Convenience functions to create common types of meshes.

DOLFINx port (Task 18): the mesh generators are now driven through the Gmsh
Python API (OpenCASCADE kernel) and converted to ``dolfinx.mesh`` in-memory via
``dolfinx.io.gmsh.model_to_mesh`` -- there are no dolfin-XML intermediates and
no shelling out to the ``gmsh`` CLI. The public generator names/signatures are
preserved (``box``, ``sphere``, ``cylinder``, ``nanodisk``,
``elliptic_cylinder``, ``elliptical_nanodisk``, ``ellipsoid``,
``truncated_cone``, ``ring``, ``pair_of_disks``).

The md5-keyed caching *contract* of the legacy ``from_csg`` path is preserved
with a DOLFINx-native store: generated meshes are cached as XDMF (``.xdmf`` +
``.h5``). Same inputs -> cache hit (no gmsh regeneration); distinct inputs ->
distinct files; the ``directory`` override is honoured. The XDMF read-back is
kept strictly internal to this cache (it is NOT a general XDMF-read capability;
that remains Task 26).

``gmsh`` and the dolfinx runtime are imported *lazily* inside the generators so
that ``import finmag`` never pulls in gmsh (plain-import boundary).

Deferred by name in this slice (documented, Task 29 review items):
- the Netgen backend (``netgen_is_usable`` returns ``False``; conda-forge
  ``netgen`` is not part of the dolfinx env and the ``.geo`` -> netgen ->
  DIFFPACK -> dolfin-XML path does not port cleanly);
- the textual Netgen-CSG / dolfin-XML entry points ``from_geofile`` /
  ``from_csg`` (the Gmsh-API route replaces textual CSG);
- the multi-region "airbox" generators ``sphere_inside_box`` /
  ``elliptical_nanodisk_with_cuboid_shell`` (subdomain-marked meshes);
- the 2D gmsh-script helpers ``regular_polygon`` /
  ``regular_polygon_extruded`` / ``disk_with_internal_layers``;
- the dolfin-based mesh analysis/plotting utilities (``mesh_info``,
  ``mesh_quality``, ``nodal_volume``, ``longest_edges``, ``mesh_size``,
  ``mesh_size_plausible``, ``describe_mesh_size``, ``print_mesh_info``,
  ``build_mesh``, ``embed3d``, ``line_mesh``, ``mesh_is_periodic``,
  ``plot_mesh``, ``plot_mesh_with_paraview``, ``plot_mesh_regions``).

Caveat (inherited): mesh coordinates are stored at reduced precision, so build
"macroscopic" meshes and use ``unit_length`` to set the physical scale.

[Claude Opus 4.8]
"""

import os
import re
import math
import hashlib
import textwrap
import logging

import numpy as np
from math import sin, cos, pi

logger = logging.getLogger(name='finmag')


# --------------------------------------------------------------------------
# by-name deferrals (Task 29 review items)
# --------------------------------------------------------------------------

def _deferred(name, note):
    def _stub(*args, **kwargs):
        raise NotImplementedError(
            "finmag.util.meshes.{} is deferred by name in the DOLFINx mesh "
            "bridge port (Task 18); {} It is tracked as a Task 29 review "
            "item.".format(name, note))
    _stub.__name__ = name
    _stub.__qualname__ = name
    return _stub


# --------------------------------------------------------------------------
# Gmsh -> dolfinx bridge (serial only)
# --------------------------------------------------------------------------

def _generate_mesh_via_gmsh(build, gdim=3):
    """Run ``build(gmsh)`` inside a fresh Gmsh session and convert the model to
    a ``dolfinx.mesh``.

    ``build`` receives the imported ``gmsh`` module and must: create the OCC
    geometry, ``occ.synchronize()``, add the physical group(s) of dimension
    ``gdim``, set the mesh sizes, and call ``gmsh.model.mesh.generate(gdim)``.

    ``gmsh`` and the dolfinx runtime are imported lazily here so the plain
    ``import finmag`` never pulls them in.
    """
    import gmsh
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("finmag")
        build(gmsh)
        mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim)
    finally:
        gmsh.finalize()
    return mesh_data.mesh


def _make_build(add_solids, maxh, gdim=3, size_regions=None):
    """Build the ``build(gmsh)`` closure for ``_generate_mesh_via_gmsh``.

    ``add_solids(occ)`` creates the OCC solids and returns the list of
    top-dimensional volume tags. ``size_regions``, if given, is a list of
    ``(contains_fn, size)`` pairs used to drive a spatially varying mesh size
    (this reproduces the legacy per-solid ``maxh_NAME`` discretisation for
    combined meshes); otherwise a uniform ``maxh`` is used.
    """
    def build(gmsh):
        occ = gmsh.model.occ
        vol_tags = add_solids(occ)
        occ.synchronize()
        gmsh.model.addPhysicalGroup(gdim, vol_tags, 1)

        if size_regions:
            regions = list(size_regions)
            sizes = [s for (_c, s) in regions]
            default = max(sizes)
            gmsh.option.setNumber("Mesh.MeshSizeMax", default)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

            def _size(dim, tag, x, y, z, lc):
                best = None
                for contains, s in regions:
                    if contains(x, y, z):
                        best = s if best is None else min(best, s)
                return default if best is None else best

            gmsh.model.mesh.setSizeCallback(_size)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)

        gmsh.model.mesh.generate(gdim)

    return build


def _write_cached_mesh(path, mesh):
    from dolfinx.io import XDMFFile
    with XDMFFile(mesh.comm, path, "w") as f:
        f.write_mesh(mesh)


def _read_cached_mesh(path, gdim=3):
    # Internal cache read-back only -- not a general XDMF-read capability.
    from dolfinx.io import XDMFFile
    from mpi4py import MPI
    with XDMFFile(MPI.COMM_WORLD, path, "r") as f:
        return f.read_mesh()


def _mesh_from_geometry(csg_key, add_solids, maxh, save_result, filename,
                        directory, gdim=3, size_regions=None):
    """Generate (or load from cache) a dolfinx mesh for the given geometry.

    ``csg_key`` is the legacy Netgen-CSG text; it is used only as the md5 cache
    key when no explicit ``filename`` is supplied, preserving the legacy
    hashing contract.
    """
    build = _make_build(add_solids, maxh, gdim=gdim, size_regions=size_regions)

    if not save_result:
        return _generate_mesh_via_gmsh(build, gdim=gdim)

    if filename == '':
        key = csg_key.encode("utf-8") if isinstance(csg_key, str) else csg_key
        filename = hashlib.md5(key).hexdigest()

    # Honour a legacy '.xml.gz' (or '.xdmf') filename by stripping the suffix.
    filename = re.sub(r'\.xml\.gz$', '', filename)
    filename = re.sub(r'\.xdmf$', '', filename)

    if directory == '':
        directory = os.curdir

    path = os.path.abspath(os.path.join(directory, filename) + ".xdmf")
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    if os.path.exists(path):
        logger.debug("Mesh cache hit: %s", path)
        return _read_cached_mesh(path, gdim=gdim)

    mesh = _generate_mesh_via_gmsh(build, gdim=gdim)
    _write_cached_mesh(path, mesh)
    return mesh


# --------------------------------------------------------------------------
# generators (public surface preserved)
# --------------------------------------------------------------------------

def box(x0, x1, x2, y0, y1, y2, maxh, save_result=True, filename='', directory=''):
    """
    Returns a dolfinx mesh describing an axis-parallel box.

    The two points (x0, x1, x2) and (y0, y1, y2) are interpreted as two
    diagonally opposite corners of the box.
    """
    # Make sure that each min-coordinate < max-coordinate.
    [x0, y0] = sorted([x0, y0])
    [x1, y1] = sorted([x1, y1])
    [x2, y2] = sorted([x2, y2])

    csg = textwrap.dedent("""\
        algebraic3d
        solid main = orthobrick ( {}, {}, {}; {}, {}, {} ) -maxh = {maxh};
        tlo main;
        """).format(x0, x1, x2, y0, y1, y2, maxh=maxh)
    if save_result == True and filename == '':
        filename = "box-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(
            x0, x1, x2, y0, y1, y2, maxh).replace(".", "_")

    def add(occ):
        return [occ.addBox(x0, x1, x2, y0 - x0, y1 - x1, y2 - x2)]

    return _mesh_from_geometry(csg, add, maxh, save_result, filename, directory)


def sphere(r, maxh, save_result=True, filename='', directory=''):
    """
    Returns a dolfinx mesh describing a sphere of radius ``r`` centred at the
    origin with mesh coarseness ``maxh``.
    """
    csg = textwrap.dedent("""\
        algebraic3d
        solid main = sphere ( 0, 0, 0; {r} ) -maxh = {maxh};
        tlo main;
        """).format(r=r, maxh=maxh)
    if save_result == True and filename == '':
        filename = "sphere-{:g}-{:g}".format(r, maxh).replace(".", "_")

    def add(occ):
        return [occ.addSphere(0, 0, 0, r)]

    return _mesh_from_geometry(csg, add, maxh, save_result, filename, directory)


def cylinder(r, h, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfinx mesh representing a cylinder of radius ``r`` and height
    ``h`` (axis along z, base at z=0). ``maxh`` controls the element size.
    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = cylinder (0, 0, 1; 0, 0, -1; {r} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;
        """).format(r=r, h=h, maxh=maxh)
    if save_result == True and filename == '':
        filename = "cyl-{:.1f}-{:.1f}-{:.1f}".format(r, h, maxh).replace(".", "_")

    def add(occ):
        return [occ.addCylinder(0, 0, 0, 0, 0, h, r)]

    return _mesh_from_geometry(csg_string, add, maxh, save_result, filename, directory)


def nanodisk(d, h, maxh, save_result=True, filename='', directory=''):
    """
    Almost exactly the same as ``cylinder``, but the first argument is the
    *diameter* of the disk, not the radius.
    """
    return cylinder(0.5 * d, h, maxh, save_result=save_result,
                    filename=filename, directory=directory)


def elliptic_cylinder(r1, r2, h, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfinx mesh representing an elliptic cylinder with semi-major
    axis ``r1``, semi-minor axis ``r2`` and height ``h`` (axis along z).
    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = ellipticcylinder (0, 0, 0; {r1}, 0, 0; 0, {r2}, 0 )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;
        """).format(r1=r1, r2=r2, h=h, maxh=maxh)
    if save_result == True and filename == '':
        filename = "ellcyl-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(
            r1, r2, h, maxh).replace(".", "_")

    def add(occ):
        t = occ.addCylinder(0, 0, 0, 0, 0, h, 1.0)
        occ.dilate([(3, t)], 0, 0, 0, r1, r2, 1.0)
        return [t]

    return _mesh_from_geometry(csg_string, add, maxh, save_result, filename, directory)


def elliptical_nanodisk(d1, d2, h, maxh, save_result=True, filename='', directory=''):
    """
    Almost exactly the same as ``elliptic_cylinder``, except that the axes are
    given by the *diameters*, not the radii.
    """
    return elliptic_cylinder(0.5 * d1, 0.5 * d2, h, maxh, save_result=save_result,
                             filename=filename, directory=directory)


def ellipsoid(r1, r2, r3, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfinx mesh representing an ellipsoid with main axes lengths
    ``r1``, ``r2``, ``r3``.
    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid ell = ellipsoid (0, 0, 0; {r1}, 0, 0; 0, {r2}, 0; 0, 0, {r3}) -maxh = {maxh};
        tlo ell;
        """).format(r1=r1, r2=r2, r3=r3, maxh=maxh)
    if save_result == True and filename == '':
        filename = "ellipsoid-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(
            r1, r2, r3, maxh).replace(".", "_")

    def add(occ):
        t = occ.addSphere(0, 0, 0, 1.0)
        occ.dilate([(3, t)], 0, 0, 0, r1, r2, r3)
        return [t]

    return _mesh_from_geometry(csg_string, add, maxh, save_result, filename, directory)


def ring(r1, r2, h, maxh, save_result=True, filename='', directory='',
         with_middle_plane=False):
    """
    Return a dolfinx mesh representing a ring with inner radius ``r1``, outer
    radius ``r2`` and height ``h`` (centred on z=0).
    """
    if with_middle_plane:
        raise NotImplementedError(
            "finmag.util.meshes.ring(with_middle_plane=True) is deferred by "
            "name in the DOLFINx mesh bridge port (Task 18); the legacy "
            "three-solid CSG variant (an extra half-height solid split at "
            "the middle plane) is not reproduced by the Gmsh OCC bridge in "
            "this slice -- with_middle_plane=False (the default) is "
            "unaffected. It is tracked as a Task 29 review item (flagged in "
            "review: this kwarg was previously accepted but silently "
            "ignored).")

    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = cylinder (0, 0, -{h}; 0, 0, {h}; {r1} )
              and plane (0, 0, -{h}; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1);
        solid fincyl2 = cylinder (0, 0, -{h}; 0, 0, 0; {r2} )
              and plane (0, 0, -{h}; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1);
        solid ring = fincyl2 and not fincyl -maxh = {maxh};
        tlo ring;
        """).format(r1=r1, r2=r2, h=h / 2.0, maxh=maxh)
    if save_result == True and filename == '':
        filename = "ring-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(
            r1, r2, h, maxh).replace(".", "_")

    def add(occ):
        outer = occ.addCylinder(0, 0, -0.5 * h, 0, 0, h, r2)
        inner = occ.addCylinder(0, 0, -0.5 * h, 0, 0, h, r1)
        out, _ = occ.cut([(3, outer)], [(3, inner)])
        return [t for (_d, t) in out]

    return _mesh_from_geometry(csg_string, add, maxh, save_result, filename, directory)


def truncated_cone(r_base, r_top, h, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfinx mesh representing a truncated cone of base-radius
    ``r_base`` and top-radius ``r_top`` with height ``h`` (axis along z).
    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid cutcone = cone ( 0, 0, 0; {r_base}; 0, 0, {h}; {r_top})
            and plane (0, 0, 0; 0, 0, -1)
            and plane (0, 0, {h}; 0, 0, 1) -maxh = {maxh};
        tlo cutcone;
        """).format(r_base=r_base, r_top=r_top, h=h, maxh=maxh)
    if save_result == True and filename == '':
        filename = "cutcone-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(
            r_base, r_top, h, maxh).replace(".", "_")

    def add(occ):
        return [occ.addCone(0, 0, 0, 0, 0, h, r_base, r_top)]

    return _mesh_from_geometry(csg_string, add, maxh, save_result, filename, directory)


def pair_of_disks(d1, d2, h1, h2, sep, theta, maxh, save_result=True,
                  filename='', directory=''):
    """
    Return a dolfinx mesh representing a pair of (disjoint) disks. The first
    disk is centred at the origin; the second is placed so that the
    edge-to-edge separation equals ``sep`` at angle ``theta`` (degrees).
    """
    theta_rad = theta * pi / 180.0
    r1 = 0.5 * d1
    r2 = 0.5 * d2
    sep_centers = r1 + sep + r2
    x2 = sep_centers * cos(theta_rad)
    y2 = sep_centers * sin(theta_rad)
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid disk1 = cylinder (0, 0, 1; 0, 0, -1; {r1} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {h1}; 0, 0, 1) -maxh = {maxh};
        solid disk2 = cylinder ({x2}, {y2}, 1; {x2}, {y2}, -1; {r2} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {h2}; 0, 0, 1) -maxh = {maxh};
        tlo disk1;
        tlo disk2;
        """).format(r1=r1, h1=h1, x2=x2, y2=y2, r2=r2, h2=h2, maxh=maxh)
    if save_result == True and filename == '':
        filename = "diskpair-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(
            r1, r2, h1, h2, sep, theta, maxh).replace(".", "_")

    def add(occ):
        disk1 = occ.addCylinder(0, 0, 0, 0, 0, h1, r1)
        disk2 = occ.addCylinder(x2, y2, 0, 0, 0, h2, r2)
        return [disk1, disk2]

    return _mesh_from_geometry(csg_string, add, maxh, save_result, filename, directory)


# --------------------------------------------------------------------------
# mesh metrics
# --------------------------------------------------------------------------

def mesh_volume(mesh):
    """Total volume of all cells in a dolfinx mesh (in mesh units)."""
    import ufl
    from mpi4py import MPI
    from dolfinx import fem
    local = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx))
    return mesh.comm.allreduce(local, op=MPI.SUM)


def num_vertices(mesh):
    """Global number of vertices in a dolfinx mesh."""
    return mesh.topology.index_map(0).size_global


def order_of_magnitude(value):
    return int(math.floor(math.log10(value)))


# --------------------------------------------------------------------------
# deferred backends / surfaces (documented, Task 29 review items)
# --------------------------------------------------------------------------

def netgen_is_usable():
    """The Netgen backend is deferred in the DOLFINx port (Task 18); the Gmsh
    bridge covers the ported geometry set. Returns ``False`` so any legacy
    Netgen-gated coverage is treated as unavailable. Tracked for Task 29."""
    return False


from_geofile = _deferred(
    "from_geofile",
    "the textual Netgen '.geo' -> dolfin-XML path is replaced by the Gmsh "
    "Python-API generators.")
from_csg = _deferred(
    "from_csg",
    "the textual Netgen-CSG -> dolfin-XML path is replaced by the Gmsh "
    "Python-API generators (build geometry through the named generators or "
    "mesh_templates classes instead).")
sphere_inside_box = _deferred(
    "sphere_inside_box",
    "multi-region 'airbox' meshes with subdomain markers are not part of this "
    "single-material generator slice.")
elliptical_nanodisk_with_cuboid_shell = _deferred(
    "elliptical_nanodisk_with_cuboid_shell",
    "multi-region 'airbox' meshes with subdomain markers are not part of this "
    "single-material generator slice.")
regular_polygon = _deferred(
    "regular_polygon",
    "the 2D gmsh-script polygon helper is not ported in this slice.")
regular_polygon_extruded = _deferred(
    "regular_polygon_extruded",
    "the 2D gmsh-script polygon helper is not ported in this slice.")
disk_with_internal_layers = _deferred(
    "disk_with_internal_layers",
    "the layered gmsh-script disk helper is not ported in this slice.")
mesh_info = _deferred(
    "mesh_info",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
mesh_quality = _deferred(
    "mesh_quality",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
nodal_volume = _deferred(
    "nodal_volume",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
longest_edges = _deferred(
    "longest_edges",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
mesh_size = _deferred(
    "mesh_size",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
mesh_size_plausible = _deferred(
    "mesh_size_plausible",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
describe_mesh_size = _deferred(
    "describe_mesh_size",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
print_mesh_info = _deferred(
    "print_mesh_info",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
mesh_is_periodic = _deferred(
    "mesh_is_periodic",
    "dolfin-based mesh analysis utilities are not ported in this slice.")
build_mesh = _deferred(
    "build_mesh",
    "the dolfin MeshEditor-based builder is not ported in this slice.")
embed3d = _deferred(
    "embed3d",
    "the dolfin MeshEditor-based builder is not ported in this slice.")
line_mesh = _deferred(
    "line_mesh",
    "the dolfin MeshEditor-based builder is not ported in this slice.")
plot_mesh = _deferred(
    "plot_mesh",
    "matplotlib/dolfin mesh plotting is not ported in this slice.")
plot_mesh_with_paraview = _deferred(
    "plot_mesh_with_paraview",
    "Paraview-based mesh rendering is not ported in this slice.")
plot_mesh_regions = _deferred(
    "plot_mesh_regions",
    "matplotlib/dolfin mesh-region plotting is not ported in this slice.")
