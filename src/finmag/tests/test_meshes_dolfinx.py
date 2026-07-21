"""Direct DOLFINx mesh-tooling bridge port (Task 18).

Ports the trusted invariants of the legacy Netgen/Gmsh mesh generators
(``src/finmag/util/meshes.py``) and CSG template classes
(``src/finmag/util/mesh_templates.py``) onto a Gmsh-Python-API bridge that
converts geometry to ``dolfinx.mesh`` in-memory via
``dolfinx.io.gmsh.model_to_mesh`` (no dolfin-XML intermediates).

What is validated here (mirroring the frozen legacy oracle tests
``meshes_test.py`` / ``mesh_templates_test.py`` at
``ba9280934e188d7f3800e7b9865e70a9422f7687``):

- analytic volume checks per generator/template within the legacy tolerances
  (loose ``TOL1`` for curved meshes, exact ``TOL3`` for planar boxes);
- the *preserved* Netgen-CSG md5 hashing / naming contract (``test_hash``'s
  exact digests still hold, because the CSG text the templates emit is kept
  byte-for-byte and only used as the cache key -- the geometry itself is now
  built through the Gmsh OCC kernel);
- the caching contract on a DOLFINx-native store (XDMF): same inputs -> cache
  hit (no gmsh regeneration), distinct inputs -> distinct files, ``directory``
  override honoured;
- one FK-demag-on-a-generated-sphere smoke proving the bridge feeds real
  physics (uniform sphere demag factor ~1/3).

[Claude Opus 4.8]
"""

import os
from math import pi

import numpy as np
import pytest
from mpi4py import MPI

import dolfinx.fem as fem

from finmag.util import meshes
from finmag.util.meshes import (
    box, sphere, cylinder, nanodisk, elliptic_cylinder,
    elliptical_nanodisk, ellipsoid, truncated_cone, ring, pair_of_disks,
    mesh_volume, num_vertices, netgen_is_usable,
)
from finmag.util.mesh_templates import (
    MeshTemplate, MeshPrimitive, Sphere, Box, EllipticalNanodisk, Nanodisk,
    MeshSum, MeshDifference, netgen_primitives,
)

# Legacy tolerances (mesh_templates_test.py): loose for curved boundaries,
# exact for planar boxes, intermediate for combined meshes.
TOL1 = 1e-2
TOL2 = 1e-5
TOL3 = 1e-14


def _check_volume(mesh, expected, rtol, atol=0.0):
    got = mesh_volume(mesh)
    assert np.allclose(got, expected, rtol=rtol, atol=atol), \
        "volume {} not within rtol={} of expected {}".format(got, rtol, expected)


# --------------------------------------------------------------------------
# generator volume invariants
# --------------------------------------------------------------------------

def test_box_volume_exact(tmp_path):
    os.chdir(str(tmp_path))
    m = box(0, 0, 0, 10, 20, 30, maxh=8.0, save_result=False)
    _check_volume(m, 10 * 20 * 30, TOL3)


def test_sphere_volume(tmp_path):
    os.chdir(str(tmp_path))
    m = sphere(r=20.0, maxh=2.5, save_result=False)
    _check_volume(m, 4.0 / 3 * pi * 20.0 ** 3, TOL1)


def test_cylinder_and_nanodisk_volume(tmp_path):
    os.chdir(str(tmp_path))
    mc = cylinder(r=10.0, h=5.0, maxh=2.0, save_result=False)
    _check_volume(mc, pi * 10.0 ** 2 * 5.0, TOL1)
    # nanodisk(d, h) == cylinder(d/2, h)
    md = nanodisk(d=20.0, h=5.0, maxh=2.0, save_result=False)
    _check_volume(md, pi * 10.0 ** 2 * 5.0, TOL1)


def test_elliptic_cylinder_and_nanodisk_volume(tmp_path):
    os.chdir(str(tmp_path))
    me = elliptic_cylinder(r1=15.0, r2=10.0, h=5.0, maxh=2.0, save_result=False)
    _check_volume(me, pi * 15.0 * 10.0 * 5.0, TOL1)
    men = elliptical_nanodisk(d1=30.0, d2=20.0, h=5.0, maxh=2.0, save_result=False)
    _check_volume(men, pi * 15.0 * 10.0 * 5.0, TOL1)


def test_ellipsoid_volume(tmp_path):
    os.chdir(str(tmp_path))
    m = ellipsoid(r1=20.0, r2=15.0, r3=10.0, maxh=2.0, save_result=False)
    _check_volume(m, 4.0 / 3 * pi * 20.0 * 15.0 * 10.0, TOL1)


def test_ring_volume(tmp_path):
    os.chdir(str(tmp_path))
    r1, r2, h = 10.0, 18.0, 6.0
    m = ring(r1=r1, r2=r2, h=h, maxh=2.0, save_result=False)
    _check_volume(m, pi * (r2 ** 2 - r1 ** 2) * h, TOL1)


def test_truncated_cone_volume(tmp_path):
    os.chdir(str(tmp_path))
    rb, rt, h = 20.0, 10.0, 15.0
    m = truncated_cone(r_base=rb, r_top=rt, h=h, maxh=2.0, save_result=False)
    expected = pi * h / 3.0 * (rb ** 2 + rb * rt + rt ** 2)
    _check_volume(m, expected, TOL1)


def test_pair_of_disks_volume(tmp_path):
    os.chdir(str(tmp_path))
    m = pair_of_disks(d1=20.0, d2=20.0, h1=5.0, h2=5.0, sep=10.0, theta=0.0,
                      maxh=2.0, save_result=False)
    _check_volume(m, 2 * pi * 10.0 ** 2 * 5.0, TOL1)


# --------------------------------------------------------------------------
# template surface: names, hashing, disallowed names, maxh dispatch
# --------------------------------------------------------------------------

def test_generic_template_cannot_create_mesh():
    proto = MeshTemplate()
    with pytest.raises(NotImplementedError):
        proto.create_mesh(maxh=1.0)


def test_disallowed_names():
    for name in netgen_primitives:
        with pytest.raises(ValueError):
            Sphere(r=10, name=name)


def test_hash_is_preserved_from_legacy_csg():
    """The Netgen-CSG md5 digests are preserved byte-for-byte -- these exact
    values are pinned in the frozen legacy ``mesh_templates_test.py``."""
    s = Sphere(r=10, name='MySphere')
    h1 = s.hash(maxh=3.0)
    h2 = s.hash(maxh_MySphere=3.0)
    h3 = s.hash(maxh=4.0)
    assert h1 == '50f3b55770e40ba7a5f8e62d7ff7d327'
    assert h1 == h2
    assert h3 == '1ee55186811cfc21f22e17fbad35bfed'


def test_maxh_dispatch_with_mesh_primitive():
    prim = MeshPrimitive(name='foo')
    assert prim._get_maxh(maxh=2.0, maxh_foo=5.0) == 5.0
    assert prim._get_maxh(maxh=2.0, maxh_bar=5.0) == 2.0
    with pytest.raises(ValueError):
        prim._get_maxh(random_arg=42)

    prim = MeshPrimitive(name='foo', csg_string='-maxh = {maxh_foo}')
    assert prim.csg_stub(maxh=2.0) == '-maxh = 2.0'
    assert prim.csg_stub(maxh_foo=3.0) == '-maxh = 3.0'
    assert prim.csg_stub(maxh=2.0, maxh_foo=3.0) == '-maxh = 3.0'
    with pytest.raises(ValueError):
        prim.csg_stub(maxh_bar=4.0)


# --------------------------------------------------------------------------
# template volume invariants
# --------------------------------------------------------------------------

def test_template_sphere_volume(tmp_path):
    os.chdir(str(tmp_path))
    r = 20.0
    s = Sphere(r, center=(2, 3, -4))
    m = s.create_mesh(maxh=2.5, save_result=False)
    _check_volume(m, 4.0 / 3 * pi * r ** 3, TOL1)


def test_template_box_volume(tmp_path):
    os.chdir(str(tmp_path))
    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 10, 20, 30
    b = Box(x0, y0, z0, x1, y1, z1)
    m = b.create_mesh(maxh=8.0, save_result=False)
    _check_volume(m, (x1 - x0) * (y1 - y0) * (z1 - z0), TOL3)


def test_template_nanodisk_volume(tmp_path):
    os.chdir(str(tmp_path))
    d, h = 20.0, 5.0
    nd = Nanodisk(d, h, center=(2, 3, -4), valign='bottom')
    assert nd.valign == 'bottom'
    m = nd.create_mesh(maxh=2.5, save_result=False)
    _check_volume(m, pi * (0.5 * d) ** 2 * h, TOL1)


def test_template_elliptical_nanodisk_volume(tmp_path):
    os.chdir(str(tmp_path))
    d1, d2, h = 30.0, 20.0, 5.0
    for valign in ('bottom', 'center', 'top'):
        nd = EllipticalNanodisk(d1, d2, h, center=(2, 3, -4), valign=valign)
        assert nd.valign == valign
        m = nd.create_mesh(maxh=2.5, save_result=False)
        _check_volume(m, pi * (0.5 * d1) * (0.5 * d2) * h, TOL1)
    with pytest.raises(ValueError):
        EllipticalNanodisk(d1, d2, h, valign='foo')


def test_template_mesh_sum_volume(tmp_path):
    os.chdir(str(tmp_path))
    r1, r2, r3 = 10.0, 18.0, 12.0
    maxh = 2.0
    # Same automatic name must be rejected.
    with pytest.raises(ValueError):
        _ = Sphere(r1, center=(-30, 0, 0)) + Sphere(r2, center=(30, 0, 0))

    sphere1 = Sphere(r1, center=(-30, 0, 0), name='sphere_1')
    sphere2 = Sphere(r2, center=(+30, 0, 0), name='sphere_2')
    sphere3 = Sphere(r3, center=(0, 40, 0), name='sphere_3')
    three = sphere1 + sphere2 + sphere3
    m = three.create_mesh(maxh=maxh, save_result=False)
    vol_exact = sum(4.0 / 3 * pi * r ** 3 for r in (r1, r2, r3))
    _check_volume(m, vol_exact, TOL1)


def test_template_mesh_difference_volume(tmp_path):
    os.chdir(str(tmp_path))
    x1, y1, z1 = 50.0, 30.0, 20.0
    x2, y2, z2 = 30.0, 20.0, 15.0
    box1 = Box(0, 0, 0, x1, y1, z1, name='box1')
    box2 = Box(x2, y2, z2, x1 + 10, y1 + 10, z1 + 10, name='box2')
    diff = box1 - box2
    m = diff.create_mesh(maxh=10.0, save_result=False)
    vol_exact = x1 * y1 * z1 - (x1 - x2) * (y1 - y2) * (z1 - z2)
    _check_volume(m, vol_exact, TOL3)


def test_combined_mesh_per_primitive_maxh_changes_resolution(tmp_path):
    """A finer per-primitive ``maxh_sphere2`` yields more vertices than a
    coarser one -- the legacy per-solid discretisation contract."""
    os.chdir(str(tmp_path))
    sphere1 = Sphere(10.0, center=(-30, 0, 0), name='sphere1')
    sphere2 = Sphere(20.0, center=(+30, 0, 0), name='sphere2')
    two = sphere1 + sphere2
    m_fine = two.create_mesh(maxh=5.0, maxh_sphere2=8.0, save_result=False)
    m_coarse = two.create_mesh(maxh=5.0, maxh_sphere2=10.0, save_result=False)
    assert num_vertices(m_fine) > num_vertices(m_coarse)


# --------------------------------------------------------------------------
# caching contract (XDMF store)
# --------------------------------------------------------------------------

def test_generator_autoname_and_directory_override(tmp_path):
    os.chdir(str(tmp_path))
    sphere(r=20.0, maxh=8.0, save_result=True, directory='foo')
    assert os.path.exists('foo/sphere-20-8.xdmf')
    assert os.path.exists('foo/sphere-20-8.h5')


def test_template_autoname_and_explicit_filename(tmp_path):
    os.chdir(str(tmp_path))
    s = Sphere(20.0, center=(2, 3, -4))
    s.create_mesh(maxh=8.0, save_result=True, directory='foo')
    s.create_mesh(maxh=10.0, save_result=True, filename='bar/sphere.xml.gz')
    assert os.path.exists('foo/sphere__center_2_0_3_0_-4_0__r_20_0__maxh_8_0.xdmf')
    # A legacy '.xml.gz' filename is honoured by stripping the suffix.
    assert os.path.exists('bar/sphere.xdmf')


def test_combined_mesh_filename_uses_preserved_hash(tmp_path):
    os.chdir(str(tmp_path))
    box1 = Box(0, 0, 0, 50, 30, 20, name='box1')
    box2 = Box(30, 20, 15, 60, 40, 30, name='box2')
    diff = box1 - box2
    diff.create_mesh(maxh=10.0, save_result=True, directory=str(tmp_path))
    fname = "mesh_difference__{}.xdmf".format(box1.hash(maxh=10.0))
    assert os.path.exists(os.path.join(str(tmp_path), fname))


def test_cache_hit_skips_regeneration(tmp_path, monkeypatch):
    os.chdir(str(tmp_path))
    calls = {"n": 0}
    real = meshes._generate_mesh_via_gmsh

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(meshes, "_generate_mesh_via_gmsh", counting)

    m1 = sphere(r=15.0, maxh=3.0, save_result=True, directory='cache')
    assert calls["n"] == 1
    assert os.path.exists('cache/sphere-15-3.xdmf')
    # Second identical call must be a cache hit (read back, no regeneration).
    m2 = sphere(r=15.0, maxh=3.0, save_result=True, directory='cache')
    assert calls["n"] == 1
    assert np.isclose(mesh_volume(m1), mesh_volume(m2))


def test_distinct_inputs_distinct_cache_files(tmp_path):
    os.chdir(str(tmp_path))
    sphere(r=15.0, maxh=3.0, save_result=True, directory='cache')
    sphere(r=16.0, maxh=3.0, save_result=True, directory='cache')
    assert os.path.exists('cache/sphere-15-3.xdmf')
    assert os.path.exists('cache/sphere-16-3.xdmf')


# --------------------------------------------------------------------------
# deferred backends (documented, Task 29 review items)
# --------------------------------------------------------------------------

def test_netgen_backend_deferred_by_name():
    assert netgen_is_usable() is False


def test_deferred_generators_raise_by_name():
    for fn, kwargs in [
        (meshes.from_geofile, dict(geofile='x.geo')),
        (meshes.from_csg, dict(csg='algebraic3d\n')),
        (meshes.sphere_inside_box,
         dict(r_sphere=10, r_shell=15, l_box=50, maxh_sphere=5, maxh_box=10)),
    ]:
        with pytest.raises(NotImplementedError):
            fn(**kwargs)


# --------------------------------------------------------------------------
# physics smoke: FK demag on a generated sphere (demag factor ~ 1/3)
# --------------------------------------------------------------------------

def test_fk_demag_on_generated_sphere_smoke(tmp_path):
    os.chdir(str(tmp_path))
    from finmag.field import Field
    from finmag.energies.demag.fk_demag import FKDemag

    r = 20.0
    mesh = sphere(r=r, maxh=3.0, save_result=False)
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))
    Ms = 8.6e5
    m = Field(S3, (0.0, 0.0, 1.0))
    Ms_field = Field(DG, Ms)
    demag = FKDemag()
    demag.setup(m, Ms_field, unit_length=1e-9)
    avg = demag.average_field().reshape(-1)
    # Uniformly magnetised sphere: H_demag = -Ms/3 along the magnetisation.
    assert np.isclose(avg[2], -Ms / 3.0, rtol=0.05)
    assert abs(avg[0]) < 0.05 * Ms
    assert abs(avg[1]) < 0.05 * Ms
