import dolfin as df
from scipy.integrate import odeint
from finmag.sim.llg import LLG
from finmag.sim.helpers import vectors

TOLERANCE = 5e-10

def test_uniform_external_field():
    llg = LLG(df.UnitCube(2, 2, 2))
    llg.set_m0((1, 0, 0))
    llg.H_app = (0, 0, llg.Ms/2)
    llg.setup(exchange_flag=False)

    ts = [0, 1e-9]
    ys = odeint(llg.solve_for, llg.m, ts)

    mx, my, mz = llg.m_average

    assert abs(mx) < TOLERANCE
    assert abs(my) < TOLERANCE
    assert abs(1 - mz) < TOLERANCE

def test_non_uniform_external_field():
    vertices = 9
    length = 20e-9
    llg = LLG(df.Interval(vertices, 0, length))
    llg.set_m0((1, 0, 0))
    # applied field
    # (0, -H, 0) for 0 <= x <= a
    # (0, +H, 0) for a <  x <= length 
    H_expr = df.Expression(("0","H*(x[0]-a)/fabs(x[0]-a)","0"),
            a=length/2, H=llg.Ms/2)
    llg._H_app = df.interpolate(H_expr, llg.V)
    llg.setup(exchange_flag=False)

    ts = [0, 1e-9]
    ys = odeint(llg.solve_for, llg.m, ts)

    for i in xrange(len(llg.mesh.coordinates())):
        x = llg.mesh.coordinates()[i]
        _, Hy, _ = vectors(llg.H_app)[i]
        mx, my, mz = vectors(llg.m)[i]
        print "x={0} --> Hy={1} --> my={2}.".format(x, Hy, my)
        assert abs(mx) < TOLERANCE
        assert abs(mz) < TOLERANCE

        if Hy > 0:
            assert abs(my - 1) < TOLERANCE
        else:
            assert abs(my + 1) < TOLERANCE

def test_non_uniform_external_field_with_exchange():
    pass

def test_time_dependent_uniform_field():
    pass


if __name__ == "__main__":
    test_non_uniform_external_field()
