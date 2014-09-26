import equation
import numpy as np
import dolfin as df


def test_dolfins_reordering_is_active():
    # if this test fails, someone disabled dolfin's reordering of dofs
    # but this module counts on it being enabled (which is also dolfin's
    # default state since 1.2.0 or so)
    assert df.parameters.reorder_dofs_serial
    # as of September 2014 this will fail if finmag was imported beforehand
    # in the same session (since finmag disables reordering)


def test_components_of_a_vector_are_grouped_together():
    mesh = df.UnitCubeMesh(2, 2, 2)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    f = df.Function(V)
    f.assign(df.Constant((1, 2, 3)))

    # we don't know anything about the order of the degrees of freedom in
    # relation to the vertices, but we expect to have the x, y and z
    # components next to each other like this: (x1, y1, z1, ..., xn, yn, zn)
    expected = np.zeros(mesh.num_vertices() * 3)
    expected[0::3] = 1
    expected[1::3] = 2
    expected[2::3] = 3

    assert np.all(f.vector().array() == expected)
