import dolfin as df
import numpy as np
from field import *


class TestField(object):
    def setup(self):
        # Create meshes of several dimensions
        self.mesh1d = df.UnitIntervalMesh(10)
        self.mesh2d = df.UnitSquareMesh(10, 10)
        self.mesh3d = df.UnitCubeMesh(10, 10, 10)

        # Create function spaces on these meshes, all CG unless named explicitely
        self.fs_1d_scalar = df.FunctionSpace(self.mesh1d, family="CG", degree=1)
        self.fs_2d_scalar = df.FunctionSpace(self.mesh2d, family="CG", degree=1)
        self.fs_3d_scalar = df.FunctionSpace(self.mesh2d, family="CG", degree=1)

        # all vector function spaces for 3d-valued fields for now - should extend later to 2 
        # to possible catch problems
        self.fs_1d_vector = df.VectorFunctionSpace(self.mesh1d, family="CG", degree=1, dim=3)
        self.fs_2d_vector = df.VectorFunctionSpace(self.mesh2d, family="CG", degree=1, dim=3)
        self.fs_3d_vector = df.VectorFunctionSpace(self.mesh2d, family="CG", degree=1, dim=3)

    def test_init_scalar_constant(self):
        # sequence of function spaces, all scalar fields but on 1d, 2d and 3d mesh
        functionspaces = (self.fs_1d_scalar, self.fs_2d_scalar, self.fs_3d_scalar)

        for functionspace in functionspaces:

            # for each function space, test varies ways to express the constant 42
            for value in [df.Constant("42"), df.Constant("42.0"), df.Constant(42), "42", 42, 42.0]:
                f = Field(functionspace, value)

                # check values in vector, should be exact
                assert np.all(f.f.vector().array()[0] == 42.)

                # check values that are interpolated, dolfin is fairly
                # inaccurate here, see field_test.ipynb
                probe_point = f.f.geometric_dimension() * (0.5,)
                print("Probing at {}".format(probe_point))
                assert abs(f.f(probe_point) - 42) < 1e-13

                coords, values = f.get_coords_and_values()

                # this should also be exact
                assert np.all(values == 42)


    # The next lines are original from Max and may need updating after we changed the field class.
    def test_get_coords_and_values(self):
        mesh = self.mesh3d
        f = Field(self.fs_3d_vector)
        f.set(df.Expression(['x[0]', '2.3*x[1]', '-4.2*x[2]']))
        coords, values = f.get_coords_and_values()
        #print coords
        #print mesh.coordinates()
        #assert(np.allclose(coords, mesh.coordinates()))
        assert(np.allclose(values, mesh.coordinates() * np.array([1, 2.3, -4.2])))
