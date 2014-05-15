import dolfin as df
import numpy as np
from field import *


class TestField(object):
    def setup(self):
        # Create meshes of several dimensions
        self.mesh1d = df.UnitIntervalMesh(10)
        self.mesh2d = df.UnitSquareMesh(10, 10)
        self.mesh3d = df.UnitCubeMesh(10, 10, 10)

    def test_init(self):
        """
        """
        # Try various dimensions for the mesh and field values
        for mesh in [self.mesh1d, self.mesh2d, self.mesh3d]:
            for dim in [1, 2, 3]:
                f = Field(mesh, 'CG', 1, dim=dim)
                assert(f.family == 'CG')
                assert(f.degree == 1)
                assert(f.dim == dim)

        # Try different kinds of finite element families
        f4 = Field(self.mesh2d, 'CG', 5, dim=2)
        f5 = Field(self.mesh2d, 'DG', 0, dim=3)
        f6 = Field(self.mesh3d, 'N1curl', 2, dim=3)
        assert(f4.family == 'CG' and f4.degree == 5 and f4.dim == 2)
        assert(f5.family == 'DG' and f5.degree == 0 and f5.dim == 3)
        assert(f6.family == 'N1curl' and f6.degree == 2 and f6.dim == 3)

    def test_get_coords_and_values(self):
        mesh = self.mesh3d
        f = Field(mesh, 'CG', 1, dim=3)
        f.set(df.Expression(['x[0]', '2.3*x[1]', '-4.2*x[2]']))
        coords, values = f.get_coords_and_values()
        assert(np.allclose(coords, mesh.coordinates()))
        assert(np.allclose(values, mesh.coordinates() * np.array([1, 2.3, -4.2])))
