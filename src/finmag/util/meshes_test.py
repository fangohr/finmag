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
