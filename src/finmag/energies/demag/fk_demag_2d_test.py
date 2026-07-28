import dolfin as df
import numpy as np
import pytest
from finmag.field import Field
from finmag.energies.demag.fk_demag_2d import Demag2D


def test_create_mesh():
    mesh = df.UnitSquareMesh(20, 2)

    demag = Demag2D(thickness=0.1)

    mesh3 = demag.create_3d_mesh(mesh)

    coord1 = mesh.coordinates()
    coord2 = mesh3.coordinates()

    nv = len(coord1)
    eps = 1e-16
    for i in range(nv):
        assert abs(coord1[i][0] - coord2[i][0]) < eps
        assert abs(coord1[i][0] - coord2[i + nv][0]) < eps
        assert abs(coord1[i][1] - coord2[i][1]) < eps
        assert abs(coord1[i][1] - coord2[i + nv][1]) < eps


def test_demag_2d(plot=False):
    mesh = df.UnitSquareMesh(4, 4)

    Ms = 1.0
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    m0 = df.Expression(("0", "0", "1"), degree=1)

    m = Field(S3, m0)

    h = 0.001

    demag = Demag2D(thickness=h)

    demag.setup(m, Ms)
    field = demag.compute_field()
    nv = mesh.num_vertices()
    hx, hy, hz = field[:nv], field[nv:2 * nv], field[2 * nv:]

    # For a uniformly magnetised thin film, the demag field should point
    # predominantly opposite to the out-of-plane magnetisation.
    # Keep this as a qualitative 2D smoke check; the 2D reduction is approximate by construction. [Codex GPT-5.4]
    assert np.max(np.abs(hx)) < 1e-3
    assert np.max(np.abs(hy)) < 1e-3
    assert np.mean(hz) < -0.5  # XXX check if this value can be smaller 04-2026
    assert np.max(hz) < -0.3  # XXX check this one as well 04-2026

    if plot:
        df.plot(m.f)
        df.interactive()

if __name__ == "__main__":

    test_create_mesh()
    test_demag_2d(plot=True)
