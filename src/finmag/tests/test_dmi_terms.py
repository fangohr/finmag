import dolfin as df
import numpy as np

from finmag.util.helpers import times_curl


def test_dmi_term3d_matches_analytical_solution():
    # Reinstate this DMI-term check as a real Python 3 regression rather than dead skipped code. [Codex GPT-5.4]
    """
    For m(x, y, z) = (-y/2, x/2, c), curl(m) = (0, 0, 1), so
    m . curl(m) = c everywhere on the unit cube.
    """
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 10, 10, 10)
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    m = df.interpolate(df.Expression(("-0.5*x[1]", "0.5*x[0]", "1"), degree=1), V)
    energy_density = times_curl(m, 3)
    assert np.allclose(df.assemble(energy_density * df.dx), 1.0, atol=1e-13)

    m = df.interpolate(df.Expression(("-x[1]", "x[0]", "1"), degree=1), V)
    energy_density = times_curl(m, 3)
    assert np.allclose(df.assemble(3.0 * energy_density * df.dx), 6.0, atol=1e-12)


def test_dmi_term2d_matches_3d_expression_for_in_plane_derivatives():
    """
    On a 2D mesh, times_curl should agree with the 3D expression when the
    field only depends on x and y.
    """
    mesh_2d = df.RectangleMesh(df.Point(0, 0), df.Point(1, 1), 10, 10)
    mesh_3d = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 10, 10, 10)
    V_2d = df.VectorFunctionSpace(mesh_2d, "CG", 1, dim=3)
    V_3d = df.VectorFunctionSpace(mesh_3d, "CG", 1)

    expr = df.Expression(("-0.5*x[1]", "0.5*x[0]", "1"), degree=1)
    m_2d = df.interpolate(expr, V_2d)
    m_3d = df.interpolate(expr, V_3d)

    E_2d = df.assemble(times_curl(m_2d, 2) * df.dx)
    E_3d = df.assemble(times_curl(m_3d, 3) * df.dx)

    assert np.allclose(E_2d, E_3d, atol=1e-13)


def test_dmi_term_can_be_differentiated_and_assembled():
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 4, 4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    m = df.interpolate(df.Expression(("-0.5*x[1]", "0.5*x[0]", "1"), degree=1), V)
    v = df.TestFunction(V)

    E = times_curl(m, 3) * df.dx
    dE_dm = df.derivative(E, m, v)
    assert df.assemble(dE_dm).size() > 0

    jacobian = df.PETScMatrix()
    df.assemble(df.derivative(dE_dm, m), tensor=jacobian)
    assert jacobian.size(0) > 0
