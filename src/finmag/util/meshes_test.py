import dolfin as df
import os
from meshes import *
from mesh_templates import *


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


def test_embed3d():
    # Create a 2D mesh which is to be embedded in 3D space
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
