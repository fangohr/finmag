import pytest  # D33: added for the not_ported/xfail markers below (master had no pytest import)
try:  # master: import dolfin as df
    import dolfin as df
    from finmag.util.pbc2d import PeriodicBoundary2D, PeriodicBoundary1D
except ImportError:
    df = None  # not ported (D33): tests below xfail
    PeriodicBoundary2D = PeriodicBoundary1D = None


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: Simulation(pbc=...) periodicity (register D19)", strict=True)
def test_pbc1d_2dmesh():

    mesh = df.UnitSquareMesh(2, 2)

    pbc = PeriodicBoundary1D(mesh)

    S = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

    expr = df.Expression('cos(x[0])', degree=1)
    M = df.interpolate(expr, S)

    assert abs(M(0, 0.1) - M(1, 0.1)) < 1e-15
    assert abs(M(0, 0) - M(1, 0)) < 1e-15
    assert abs(M(0, 1) - M(1, 1)) < 2e-15


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: Simulation(pbc=...) periodicity (register D19)", strict=True)
def test_pbc2d_2dmesh():

    mesh = df.UnitSquareMesh(2, 2)

    pbc = PeriodicBoundary2D(mesh)

    S = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

    expr = df.Expression('cos(x[0])+x[1]', degree=1)
    M = df.interpolate(expr, S)

    assert abs(M(0, 0.1) - M(1, 0.1)) < 1e-15
    assert abs(M(0, 0) - M(1, 0)) < 1e-15
    assert abs(M(0, 0) - M(1, 1)) < 5e-15

    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 1, 1, 2)


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: Simulation(pbc=...) periodicity (register D19)", strict=True)
def test_pbc2d_3dmesh():

    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 2, 2, 1)

    pbc = PeriodicBoundary2D(mesh)

    S = df.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

    expr = df.Expression('cos(x[0])+x[1]', degree=1)
    M = df.interpolate(expr, S)

    assert abs(M(0, 0, 0) - M(1, 0, 0)) < 1e-15
    assert abs(M(0, 0, 0) - M(1, 1, 0)) < 1e-15
    assert abs(M(0, 0.1, 0) - M(1, 0.1, 0)) < 1e-15

    assert abs(M(0, 0, 0) - M(0.5, 0.5, 0)) > 0.1
    assert abs(M(0, 0, 1) - M(0.5, 0.5, 1)) > 0.1


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: Simulation(pbc=...) periodicity (register D19)", strict=True)
def test_pbc2d_3dmesh2():

    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(50, 50, 3), 15, 15, 1)
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), 2, 2, 1)

    pbc = PeriodicBoundary2D(mesh)

    S = df.VectorFunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

    expr = df.Expression(('cos(x[0]+x[2])', 'sin(x[0]+x[2])', '0'), degree=1)
    M = df.interpolate(expr, S)
    #file = df.File('poisson.pvd')
    #file << M
    # df.plot(M)
    # df.interactive()

    print(abs(M(0, 0, 1) - M(0.5, 0.5, 1)))


if __name__ == "__main__":
    test_pbc1d_2dmesh()
    test_pbc2d_2dmesh()
    test_pbc2d_3dmesh()
    test_pbc2d_3dmesh2()
