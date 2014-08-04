import dolfin as df
from finmag.field import Field
from finmag.energies.demag.fk_demag_2d import Demag2D


def test_create_mesh():
    mesh = df.UnitSquareMesh(20,2)

    demag = Demag2D(thickness=0.1)

    mesh3 = demag.create_3d_mesh(mesh)

    coord1 = mesh.coordinates()
    coord2 = mesh3.coordinates()

    nv = len(coord1)
    eps = 1e-16
    for i in range(nv):
        assert abs(coord1[i][0] - coord2[i][0]) < eps
        assert abs(coord1[i][0] - coord2[i+nv][0]) < eps
        assert abs(coord1[i][1] - coord2[i][1]) < eps
        assert abs(coord1[i][1] - coord2[i+nv][1]) < eps


def demag_2d():

    mesh = df.UnitSquareMesh(4,4)

    Ms = 1.0
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    m0 = df.Expression(("0", "0", "1"))

    m = Field(S3, m0)

    h = 0.001

    demag = Demag2D(thickness=h)

    demag.setup(m, Ms)
    print demag.compute_field()

    f0 = demag.compute_field()
    m.set_with_numpy_array_debug(f0)
    df.plot(m.f)

    print demag.m.probe(0., 0., 0)
    print demag.m.probe(1., 0., 0)
    print demag.m.probe(0., 1., 0)
    print demag.m.probe(1., 1., 0)
    print '='*50

    print demag.m.probe(0., 0., h)
    print demag.m.probe(1., 0., h)
    print demag.m.probe(0., 1., h)
    print demag.m.probe(1., 1., h)

    df.interactive()

if __name__=="__main__":

    test_create_mesh()
    demag_2d()




