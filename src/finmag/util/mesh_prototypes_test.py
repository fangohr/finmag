#!/usr/bin/env python

import pytest
import os
import numpy as np
from math import pi
from meshes import mesh_volume
from mesh_prototypes import *

TOL = 1e-2  # relative tolerance for mesh volumes


def check_mesh_volume(mesh, vol, atol=0.0, rtol=TOL):
    assert(np.allclose(mesh_volume(mesh), vol, atol=atol, rtol=rtol))


def test_mesh_prototypes(tmpdir):
    os.chdir(str(tmpdir))
    proto = MeshPrototype()
    with pytest.raises(NotImplementedError):
        proto.create_mesh('generic_mesh.xml.gz')


def test_sphere(tmpdir):
    os.chdir(str(tmpdir))
    r = 20.0

    sphere = Sphere(r, center=(2, 3, -4))

    sphere.create_mesh(maxh=8.0, save_result=True, directory='foo')
    sphere.create_mesh(maxh=10.0, save_result=True, filename='bar/sphere.xml.gz')
    assert(os.path.exists('foo/sphere__center_2_0_3_0_-4_0__r_20_0__maxh_8_0.xml.gz'))
    assert(os.path.exists('bar/sphere.xml.gz'))

    mesh = sphere.create_mesh(maxh=2.5, save_result=False)
    check_mesh_volume(mesh, 4./3 * pi * r**3)


def test_combining_meshes(tmpdir):
    os.chdir(str(tmpdir))
    r1 = 10.0
    r2 = 20.0
    r3 = 15.0

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

    mesh = three_spheres.create_mesh(maxh=2.0, save_result=True, directory=str(tmpdir))
    meshfilename = "mesh_sum__mesh_sum__sphere__center_-30_0_0_0_0_0__r_10_0__maxh_2_0__sphere__center_30_0_0_0_0_0__r_20_0__maxh_2_0__sphere__center_0_0_10_0_0_0__r_15_0__maxh_2_0.xml.gz"
    assert(os.path.exists(os.path.join(str(tmpdir), meshfilename)))

    vol1 = 4./3 * pi * r1**3
    vol2 = 4./3 * pi * r2**3
    vol3 = 4./3 * pi * r3**3
    check_mesh_volume(mesh, vol1 + vol2 + vol3)


def test_maxh_with_mesh_primitive(tmpdir):
    os.chdir(str(tmpdir))

    prim = MeshPrimitive(name='foo')
    assert(prim._get_maxh(maxh=2.0, maxh_foo=5.0) == 5.0)
    assert(prim._get_maxh(maxh=2.0, maxh_bar=5.0) == 2.0)
    with pytest.raises(ValueError):
        prim._get_maxh(random_arg=42)

    # We use non-valid CSG strings here because we only want to test the maxh functionality
    prim = MeshPrimitive(name='foo', csg_string='-maxh = {maxh_foo}')
    assert(prim.csg_string(maxh=2.0) == '-maxh = 2.0')
    assert(prim.csg_string(maxh_foo=3.0) == '-maxh = 3.0')
    assert(prim.csg_string(maxh=2.0, maxh_foo=3.0) == '-maxh = 3.0')  # 'personal' value of maxh should take precedence over generic one
    with pytest.raises(ValueError):
        prim.csg_string(maxh_bar=4.0)

    s = Sphere(r=10.0)
    s.csg_string(maxh=2.0)

    s = Sphere(r=5.0, name='my_sphere')
    s.csg_string(maxh_my_sphere=3.0)


def test_mesh_specific_maxh(tmpdir):
    """
    Check that we can pass in mesh-specific values of maxh by
    providing a keyword argument of the form 'maxh_NAME', where
    NAME is the name of the MeshPrototype.
    """
    os.chdir(str(tmpdir))
    sphere = Sphere(r=10.0, name='foobar')
    mesh1 = sphere.create_mesh(maxh=5.0)
    mesh2 = sphere.create_mesh(maxh_foobar=5.0)
    with pytest.raises(ValueError):
        sphere.create_mesh(maxh_quux=5.0)


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
