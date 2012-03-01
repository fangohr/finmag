import dolfin as df
from scipy.integrate import odeint, ode
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
        print "x={0} --> Hy={1} --> my={2}".format(x, Hy, my)
        assert abs(mx) < TOLERANCE
        assert abs(mz) < TOLERANCE

        if Hy > 0:
            assert abs(my - 1) < TOLERANCE
        else:
            assert abs(my + 1) < TOLERANCE

def _test_non_uniform_external_field_with_exchange():
    vertices = 9
    length = 20e-9
    llg = LLG(df.Box(0, 0, 0, length, 1e-9, 1e-9, vertices, 1, 1))
    #llg = LLG(df.Interval(vertices, 0, length))
    llg.set_m0((1, 0, 0))
    # applied field
    # (0, -H, 0) for 0 <= x <= a
    # (0, +H, 0) for a <  x <= length 
    H_expr = df.Expression(("fmax(H*(x[0]-a)/fabs(x[0]-a)),0)","fmin(H*(x[0]-a)/fabs(x[0]-a)), 0)","0"),
            a=length/2, H=llg.Ms/2)
    llg._H_app = df.interpolate(H_expr, llg.V)
    llg.setup(exchange_flag=True)

    llg_wrap = lambda t, y: llg.solve_for(y, t)
    t0 = 0; dt = 1e-10; t1 = 1e-9;
    r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5)
    r.set_initial_value(llg.m, t0)

    while r.successful() and r.t <= t1:
        r.integrate(r.t + dt)
    # what do we want to see?

def _test_time_dependent_uniform_field():
    llg = LLG(df.Box(0, 0, 0, 20e-9, 1e-9, 1e-9, 10, 1, 1))
    llg.alpha = 0.8 # high damping
    llg.set_m0((1, 0, 0))
    # field will change from (0, 0, H) to (0, H, 0) in one nanosecond,
    # magnitude will be constant and equal to H during that time.
    H_expr = df.Expression(("0", "H * t * pow(10,9)", "H * sqrt(1 - pow(t * pow(10,9), 2))"), t=llg.t, H=llg.Ms/2)
    llg._H_app = df.interpolate(H_expr, llg.V)
    llg.setup(exchange_flag=False)

    llg_wrap = lambda t, y: llg.solve_for(y, t)
    t0 = 0; dt = 1e-10; t1 = 1e-9;
    r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5)
    r.set_initial_value(llg.m, t0)

    while r.successful() and r.t <= t1:
        H_expr.t = r.t
        # need to interpolate at every time step?
        llg._H_app = df.interpolate(H_expr, llg.V)
        r.integrate(r.t + dt)
    # what do we want to see?
