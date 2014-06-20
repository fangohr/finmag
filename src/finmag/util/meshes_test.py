import dolfin as df
import numpy as np
import pytest
import os
from meshes import *
from mesh_templates import *
from math import sin, cos, pi


def test_mesh_size():
    """
    Construct a couple of meshes (box, sphere) and check that
    the mesh size is reported as expected.

    """
    RTOL = 1e-3
    box_mesh = df.BoxMesh(-20, -30, 10, 30, 42, 20, 4, 4, 4)
    assert(np.isclose(mesh_size(box_mesh, unit_length=1.0), 72.0, rtol=RTOL))
    assert(np.isclose(mesh_size(box_mesh, unit_length=3e-5), 216e-5, rtol=RTOL))

    s = Sphere(12.0, center=(34, 12, 17))
    sphere_mesh = s.create_mesh(maxh=3.0, save_result=False)
    assert(np.isclose(mesh_size(sphere_mesh, unit_length=1.0), 24.0, rtol=RTOL))
    assert(np.isclose(mesh_size(sphere_mesh, unit_length=2e4), 48e4, rtol=RTOL))


def test_line_mesh():
    """
    Create vertices lying on a spiral in 3D space, build a line-mesh from it
    and check that it has the correct vertices.
    """
    vertices = [(sin(t), cos(t), t) for t in np.linspace(-2*pi, 4*pi, 100)]
    mesh = line_mesh(vertices)
    assert np.allclose(vertices, mesh.coordinates())


def test_embed3d():
    # Create a 2D mesh which we want to embed in 3D space
    mesh_2d = df.RectangleMesh(0, 0, 20, 10, 10, 5)
    coords_2d = mesh_2d.coordinates()
    z_embed = 4.2

    # Create array containing the expected 3D coordinates
    coords_3d_expected = z_embed * np.ones((len(coords_2d), 3))
    coords_3d_expected[:, :2] = coords_2d

    # Create the embedded 3D mesh
    mesh_3d = embed3d(mesh_2d, z_embed)
    coords_3d = mesh_3d.coordinates()

    # Check that the coordinates coincide
    assert(np.allclose(coords_3d, coords_3d_expected))

    # Check that we can't embed a 1D or 3D mesh (TODO: we could make these
    # work, but currently they are not implemented)
    with pytest.raises(NotImplementedError):
        embed3d(df.UnitIntervalMesh(4))
    with pytest.raises(NotImplementedError):
        embed3d(df.UnitCubeMesh(4, 4, 4))


def test_sphere_inside_box(tmpdir, debug=False):
    """
    TODO: Currently this test doesn't do much; it only checks whether we can execute the command `sphere_inside_box`.
    """
    os.chdir(str(tmpdir))
    mesh = sphere_inside_box(r_sphere=10, r_shell=15, l_box=50, maxh_sphere=5.0, maxh_box=10.0, center_sphere=(10, -5, 8))
    if debug:
        plot_mesh_with_paraview(mesh, representation='Wireframe', outfile='mesh__sphere_inside_box.png')
        f = df.File('mesh__sphere_inside_box.pvd')
        f << mesh
        del f


def test_build_mesh():
    """
    Create a few meshes, extract the vertices and cells from them and pass them
    to build_mesh() to rebuild the mesh. Then check that the result is the same
    as the original.
    """
    def assert_mesh_builds_correctly(mesh):
        coords = mesh.coordinates()
        cells = mesh.cells()
        mesh_new = build_mesh(coords, cells)
        assert np.allclose(coords, mesh_new.coordinates())
        assert np.allclose(cells, mesh_new.cells())

    mesh1 = df.RectangleMesh(0, 0, 20, 10, 12, 8)
    assert_mesh_builds_correctly(mesh1)

    mesh2 = df.CircleMesh(df.Point(2.0, -3.0), 10.0, 3.0)
    assert_mesh_builds_correctly(mesh2)

    mesh3 = df.BoxMesh(0, 0, 0, 20, 10, 5, 12, 8, 3)
    assert_mesh_builds_correctly(mesh3)

    mesh4 = df.SphereMesh(df.Point(2.0, 3.0, -4.0), 10.0, 3.0)
    assert_mesh_builds_correctly(mesh4)


def create_periodic_mesh(periodicity='none', dim=3):
    """
    Helper function to create a mesh which is either non-periodic
    (if periodicity='none'), periodic in one direction if
    (periodicity='x' or periodicity='y') or periodic in both
    directions (if periodicity='xy').

    The argument `dim` specified the dimension of the mesh (allowed
    values: dim=2 or dim=3).
    """
    if dim == 2 or dim == 3:
        if periodicity == 'none':
            # Unit square with added 'asymmetric' points on the four sides (to break periodicity)
            #vertices = [(0, 0), (1, 0), (1, 1), (0.5, 1), (0, 1), (0, 0.5)]
            #cells = [(0, 1, 5), (1, 2, 3), (3, 4, 5), (1, 3, 5)]
            vertices = [(0, 0), (0.7, 0), (1, 0), (1, 0.8), (1, 1), (0.3, 1), (0, 1), (0, 0.2)]
            cells = [(0, 1, 7), (1, 2, 3), (1, 3, 7), (3, 4, 5), (3, 5, 7), (5, 6, 7)]
        elif periodicity == 'x':
            # Unit square with added 'asymmetric' points on top/bottom side
            #vertices = [(0, 0), (1, 0), (1, 1), (0.5, 1), (0, 1)]
            #cells = [(0, 1, 3), (1, 2, 3), (0, 3, 4)]
            vertices = [(0, 0), (0.2, 0), (1, 0), (1, 1), (0.7, 1), (0, 1)]
            cells = [(0, 1, 5), (1, 2, 4), (2, 3, 4), (1, 4, 5)]
        elif periodicity == 'y':
            # Unit square with added point on left edge
            vertices = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0.5)]
            cells = [(0, 1, 4), (1, 2, 4), (2, 3, 4)]
        elif periodicity == 'xy':
            # Unit square
            vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
            cells = [(0, 1, 2), (0, 2, 3)]
        else:
            raise ValueError("Argument 'periodicity' must have one of the values 'none', 'x', 'y', 'z'")
    else:
        raise NotImplementedError('Can only create 2d and 3d meshes with predefined periodicity.')

    mesh = build_mesh(vertices, cells)

    if dim == 3:
        # XXX TODO: It would be better to build a truly 3D mesh,
        #           but this should suffice for now.
        mesh = embed3d(mesh)

    return mesh


def test_mesh_is_periodic(tmpdir):
    """

    """
    os.chdir(str(tmpdir))

    # Create a bunch of 2D meshes with different periodicity
    # and check that we detect this periodicity correctly.
    mesh1 = create_periodic_mesh(periodicity='none', dim=2)
    assert not mesh_is_periodic(mesh1, 'x')
    #assert not mesh_is_periodic(mesh1, 'y')
    assert not mesh_is_periodic(mesh1, 'xy')

    mesh2 = create_periodic_mesh(periodicity='x', dim=2)
    assert mesh_is_periodic(mesh2, 'x')
    #assert not mesh_is_periodic(mesh2, 'y')
    assert not mesh_is_periodic(mesh2, 'xy')

    mesh3 = create_periodic_mesh(periodicity='y', dim=2)
    assert not mesh_is_periodic(mesh3, 'x')
    #assert mesh_is_periodic(mesh3, 'y')
    assert not mesh_is_periodic(mesh3, 'xy')

    mesh4 = create_periodic_mesh(periodicity='xy', dim=2)
    assert mesh_is_periodic(mesh4, 'x')
    #assert mesh_is_periodic(mesh4, 'y')
    assert mesh_is_periodic(mesh4, 'xy')

    mesh_rectangle = df.RectangleMesh(0, 0, 20, 10, 12, 8)
    assert mesh_is_periodic(mesh_rectangle, 'x')
    #assert mesh_is_periodic(mesh_rectangle, 'y')
    assert mesh_is_periodic(mesh_rectangle, 'xy')


    # Repeat this process for a bunch of 3D meshes with
    # different periodicity.
    mesh5 = create_periodic_mesh(periodicity='none', dim=3)
    assert not mesh_is_periodic(mesh5, 'x')
    #assert not mesh_is_periodic(mesh5, 'y')
    assert not mesh_is_periodic(mesh5, 'xy')

    mesh6 = create_periodic_mesh(periodicity='x', dim=3)
    assert mesh_is_periodic(mesh6, 'x')
    #assert not mesh_is_periodic(mesh6, 'y')
    assert not mesh_is_periodic(mesh6, 'xy')

    mesh7 = create_periodic_mesh(periodicity='y', dim=3)
    assert not mesh_is_periodic(mesh7, 'x')
    #assert mesh_is_periodic(mesh7, 'y')
    assert not mesh_is_periodic(mesh7, 'xy')

    mesh8 = create_periodic_mesh(periodicity='xy', dim=3)
    assert mesh_is_periodic(mesh8, 'x')
    #assert mesh_is_periodic(mesh8, 'y')
    assert mesh_is_periodic(mesh8, 'xy')

    mesh_box = df.BoxMesh(0, 0, 0, 20, 10, 5, 12, 8, 3)
    assert mesh_is_periodic(mesh_box, 'x')
    #assert mesh_is_periodic(mesh_box, 'y')
    assert mesh_is_periodic(mesh_box, 'xy')
