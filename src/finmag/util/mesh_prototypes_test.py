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

    #mesh = three_spheres.create_mesh(maxh=2.5, save_result=False)
    mesh = three_spheres.create_mesh(maxh=2.0, save_result=True, directory=str(tmpdir))
    assert(os.path.exists(os.path.join(str(tmpdir), 'generic_mesh_sphere_1_sphere_2_sphere_3.xml.gz')))

    vol1 = 4./3 * pi * r1**3
    vol2 = 4./3 * pi * r2**3
    vol3 = 4./3 * pi * r3**3
    check_mesh_volume(mesh, vol1 + vol2 + vol3)
