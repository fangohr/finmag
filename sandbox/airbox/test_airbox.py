import os
import dolfin as df
from finmag.util.meshes import from_geofile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
mesh = from_geofile(os.path.join(MODULE_DIR, "cube.geo"))

# This holds the region number for each cell.
mesh_function = df.MeshFunction("uint", mesh, "cube_mat.xml")

def test_create_meshfunction_from_filename():
    assert mesh_function.size() == mesh.num_cells()
