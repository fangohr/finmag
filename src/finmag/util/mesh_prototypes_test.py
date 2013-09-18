#!/usr/bin/env python

import pytest
import os
import numpy as np
from math import pi
from mesh_prototypes import *

TOL = 1e-2  # relative tolerance for mesh volumes


def check_mesh_volume(mesh, vol, atol=0.0 rtol=TOL):
    assert(np.allclose(mesh_volume(mesh), vol, atol=atol, rtol=rtol


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

    mesh = sphere.create_mesh(maxh=2.0, save_result=False)
    check_mesh_volume(mesh, 4./3 * pi * r**3)
