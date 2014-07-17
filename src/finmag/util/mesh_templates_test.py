#!/usr/bin/env python

import pytest
import os
import numpy as np
import dolfin as df
from math import pi
from meshes import mesh_volume
from mesh_templates import *
import logging

TOL1 = 1e-2   # loose tolerance for bad approximations (e.g. for a spherical mesh)
TOL2 = 1e-7   # intermediate tolerance (used e.g. for the sum of two meshes;
              # the strict tolerance won't work here because Netgen seems to
              # mesh combined meshes slightly differently than their components)
TOL3 = 1e-14  # strict tolerance where we expect almost exact values (e.g. for a box mesh)
logger = logging.getLogger("finmag")


def check_mesh_volume(mesh, vol_expected, rtol, atol=0.0):
    vol_mesh = mesh_volume(mesh)
    logger.debug("Checking mesh volume. Expected: {}, got: {} (relative error: {})".format(
                 vol_expected, vol_mesh, abs((vol_expected - vol_mesh) / vol_expected)))
    if not (np.allclose(vol_mesh, vol_expected, atol=atol, rtol=rtol)):
        print "[DDD] Expected volume: {}".format(vol_expected)
        print "[DDD] Computed volume: {}".format(vol_mesh)
    assert(np.allclose(vol_mesh, vol_expected, atol=atol, rtol=rtol))


def test_mesh_templates(tmpdir):
    os.chdir(str(tmpdir))
    proto = MeshTemplate()
    with pytest.raises(NotImplementedError):
        proto.create_mesh('generic_mesh.xml.gz')


def test_disallowed_names(tmpdir):
    """
    Check that a ValueError is raised if the user tried to use a name
    for the mesh template that coincides with a Netgen primitive.

    """
    for name in netgen_primitives:
        with pytest.raises(ValueError):
            _ = Sphere(r=10, name=name)


def test_hash():
    sphere = Sphere(r=10, name='MySphere')
    h1 = sphere.hash(maxh=3.0)
    h2 = sphere.hash(maxh_MySphere=3.0)
    h3 = sphere.hash(maxh=4.0)
    assert h1 == '50f3b55770e40ba7a5f8e62d7ff7d327'
    assert h1 == h2
    assert h3 == '1ee55186811cfc21f22e17fbad35bfed'


def test_sphere(tmpdir):
    os.chdir(str(tmpdir))
    r = 20.0

    sphere = Sphere(r, center=(2, 3, -4))

    sphere.create_mesh(maxh=8.0, save_result=True, directory='foo')
    sphere.create_mesh(maxh=10.0, save_result=True, filename='bar/sphere.xml.gz')
    assert(os.path.exists('foo/sphere__center_2_0_3_0_-4_0__r_20_0__maxh_8_0.xml.gz'))
    assert(os.path.exists('bar/sphere.xml.gz'))

    mesh = sphere.create_mesh(maxh=2.5, save_result=False)
    check_mesh_volume(mesh, 4./3 * pi * r**3, TOL1)


def test_elliptical_nanodisk(tmpdir):
    os.chdir(str(tmpdir))
    d1 = 30.0
    d2 = 20.0
    h = 5.0

    nanodisk1 = EllipticalNanodisk(d1, d2, h, center=(2, 3, -4), valign='bottom')
    assert(nanodisk1.valign == 'bottom')
    nanodisk2 = EllipticalNanodisk(d1, d2, h, center=(2, 3, -4), valign='center')
    assert(nanodisk2.valign == 'center')
    nanodisk3 = EllipticalNanodisk(d1, d2, h, center=(2, 3, -4), valign='top')
    assert(nanodisk3.valign == 'top')
    with pytest.raises(ValueError):
        # 'valign' must be one of 'top', 'bottom', 'center'
        EllipticalNanodisk(d1, d2, h, center=(2, 3, -4), valign='foo')

    mesh = nanodisk1.create_mesh(maxh=2.5)
    assert(os.path.exists('elliptical_nanodisk__d1_30_0__d2_20_0__h_5_0__center_2_0_3_0_-4_0__valign_bottom__maxh_2_5.xml.gz'))
    check_mesh_volume(mesh, pi * (0.5 * d1) * (0.5 * d2) * h, TOL1)


def test_nanodisk(tmpdir):
    os.chdir(str(tmpdir))
    d = 20.0
    h = 5.0

    nanodisk1 = Nanodisk(d, h, center=(2, 3, -4), valign='bottom')
    assert(nanodisk1.valign == 'bottom')
    nanodisk2 = Nanodisk(d, h, center=(2, 3, -4), valign='center')
    assert(nanodisk2.valign == 'center')
    nanodisk3 = Nanodisk(d, h, center=(2, 3, -4), valign='top')
    assert(nanodisk3.valign == 'top')
    with pytest.raises(ValueError):
        Nanodisk(d, h, center=(2, 3, -4), valign='foo')

    mesh = nanodisk1.create_mesh(maxh=2.5)
    assert(os.path.exists('nanodisk__d_20_0__h_5_0__center_2_0_3_0_-4_0__valign_bottom__maxh_2_5.xml.gz'))
    check_mesh_volume(mesh, pi * (0.5 * d)**2 * h, TOL1)


def test_mesh_sum(tmpdir):
    os.chdir(str(tmpdir))
    r1 = 10.0
    r2 = 18.0
    r3 = 12.0
    maxh = 2.0

    # This should raise an error because the two spheres have the same name (which is given automatically)
    sphere1 = Sphere(r1, center=(-30, 0, 0))
    sphere2 = Sphere(r2, center=(+30, 0, 0))
    with pytest.raises(ValueError):
        _ = sphere1 + sphere2

    # Same again, but with different names
    sphere1 = Sphere(r1, center=(-30, 0, 0), name='sphere_1')
    sphere2 = Sphere(r2, center=(+30, 0, 0), name='sphere_2')
    sphere3 = Sphere(r3, center=(0, 10, 0), name='sphere_3')
    three_spheres = sphere1 + sphere2 + sphere3

    mesh = three_spheres.create_mesh(maxh=maxh, save_result=True, directory=str(tmpdir))
    meshfilename = "mesh_sum__dc3c0e05d5a4303b45570750d015c3f7.xml.gz"
    assert(os.path.exists(os.path.join(str(tmpdir), meshfilename)))

    vol1 = mesh_volume(sphere1.create_mesh(maxh=maxh))
    vol2 = mesh_volume(sphere2.create_mesh(maxh=maxh))
    vol3 = mesh_volume(sphere3.create_mesh(maxh=maxh))
    vol_exact = sum([4./3 * pi * r**3  for r in [r1, r2, r3]])

    check_mesh_volume(mesh, vol_exact, TOL1)
    check_mesh_volume(mesh, vol1 + vol2 + vol3, TOL2)


def test_mesh_difference(tmpdir):
    """
    Create two boxes with some overlap and subtract the second from the first.
    Then check that the volume of the remaining part is as expected.
    """
    os.chdir(str(tmpdir))

    # Coordinates of the top-right-rear corner of box1 and
    # the bottom-left-front corner of box2.
    x1, y1, z1 = 50.0, 30.0, 20.0
    x2, y2, z2 = 30.0, 20.0, 15.0

    # Create the overlapping boxes
    box1 = Box(0, 0, 0, x1, y1, z1, name='box1')
    box2 = Box(x2, y2, z2, x1 + 10, y1 + 10, z1 + 10, name='box2')
    box1_minus_box2 = box1 - box2

    mesh = box1_minus_box2.create_mesh(maxh=10.0, save_result=True, directory=str(tmpdir))
    meshfilename = "mesh_difference__dd77171c4364ace36c40e5f5fe94951f.xml.gz"
    assert(os.path.exists(os.path.join(str(tmpdir), meshfilename)))

    vol_box1_exact = x1 * y1 * z1
    vol_overlap_exact = (x1 - x2) * (y1 - y2) * (z1 - z2)
    vol_exact = vol_box1_exact - vol_overlap_exact

    check_mesh_volume(mesh, vol_exact, TOL3)


def test_maxh_with_mesh_primitive(tmpdir):
    os.chdir(str(tmpdir))

    prim = MeshPrimitive(name='foo')
    assert(prim._get_maxh(maxh=2.0, maxh_foo=5.0) == 5.0)
    assert(prim._get_maxh(maxh=2.0, maxh_bar=5.0) == 2.0)
    with pytest.raises(ValueError):
        prim._get_maxh(random_arg=42)

    # We don't use full CSG strings here because we only want to test the maxh functionality
    prim = MeshPrimitive(name='foo', csg_string='-maxh = {maxh_foo}')
    assert(prim.csg_stub(maxh=2.0) == '-maxh = 2.0')
    assert(prim.csg_stub(maxh_foo=3.0) == '-maxh = 3.0')
    assert(prim.csg_stub(maxh=2.0, maxh_foo=3.0) == '-maxh = 3.0')  # 'personal' value of maxh should take precedence over generic one
    with pytest.raises(ValueError):
        prim.csg_stub(maxh_bar=4.0)

    s = Sphere(r=10.0)
    s.csg_stub(maxh=2.0)

    s = Sphere(r=5.0, name='my_sphere')
    s.csg_stub(maxh_my_sphere=3.0)


def test_mesh_specific_maxh(tmpdir):
    """
    Check that we can pass in mesh-specific values of maxh by
    providing a keyword argument of the form 'maxh_NAME', where
    NAME is the name of the MeshTemplate.
    """
    os.chdir(str(tmpdir))
    sphere = Sphere(r=10.0, name='foobar')
    mesh1 = sphere.create_mesh(maxh=5.0)
    mesh2 = sphere.create_mesh(maxh_foobar=5.0)
    with pytest.raises(ValueError):
        sphere.create_mesh(maxh_quux=5.0)


def test_global_maxh_can_be_omitted_if_specific_maxh_is_provided(tmpdir):
    os.chdir(str(tmpdir))

    # Providing a global value for maxh or only the value specific to the
    # sphere should both work.
    sphere = Sphere(r=10.0, name='foobar')
    mesh1 = sphere.create_mesh(maxh=3.0)
    mesh2 = sphere.create_mesh(maxh_foobar=3.0)

    # Same with a combined mesh: if all specific values for maxh are
    # given then the global maxh can be omitted.
    sphere1 = Sphere(r=10, name='sphere1')
    sphere2 = Sphere(r=10, center=(20, 0, 0), name='sphere2')
    two_spheres = sphere1 + sphere2
    mesh = two_spheres.create_mesh(maxh_sphere1=4.0, maxh_sphere2=5.0)


def test_different_mesh_discretisations_for_combined_meshes(tmpdir):
    """
    Check that we can create a mesh consisting of two spheres for which
    we provide a generic value of maxh as well as a specific value for
    the second spheres.
    """
    os.chdir(str(tmpdir))
    r1 = 10.0
    r2 = 20.0

    sphere1 = Sphere(r1, center=(-30, 0, 0), name='sphere1')
    sphere2 = Sphere(r2, center=(+30, 0, 0), name='sphere2')

    two_spheres = sphere1 + sphere2

    # This should render the two spheres with different mesh discretisations.
    # XXX TODO: How to best check that this worked correctly?!? Currently my best idea is
    #           to create the mesh twice, once with a fine and once with a coarse discretisation
    #           for the second sphere, and to check that the second mesh has fewer vertices.
    mesh1 = two_spheres.create_mesh(maxh=5.0, maxh_sphere2=8.0, save_result=True, directory=str(tmpdir))
    mesh2 = two_spheres.create_mesh(maxh=5.0, maxh_sphere2=10.0, save_result=True, directory=str(tmpdir))
    assert(mesh1.num_vertices() > mesh2.num_vertices())


def test_box(tmpdir):
    os.chdir(str(tmpdir))
    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 10, 20, 30

    box = Box(x0, y0, z0, x1, y1, z1)

    box.create_mesh(maxh=8.0, save_result=True, directory='foo')
    box.create_mesh(maxh=10.0, save_result=True, filename='bar/box.xml.gz')
    assert(os.path.exists('foo/box__0_0__0_0__0_0__10_0__20_0__30_0__maxh_8_0.xml.gz'))
    assert(os.path.exists('bar/box.xml.gz'))

    mesh = df.Mesh('bar/box.xml.gz')
    check_mesh_volume(mesh, (x1 - x0) * (y1 - y0) * (z1 - z0), TOL3)
