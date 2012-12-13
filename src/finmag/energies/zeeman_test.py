import numpy as np
import dolfin as df
from finmag.energies import TimeZeeman, DiscreteTimeZeeman

mesh = df.UnitCube(2, 2, 2)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
m = df.Function(S3)
m.assign(df.Constant((1, 0, 0)))
Ms = 1
TOL = 1e-14

def diff(H_ext, expected_field):
    H = H_ext.compute_field().reshape((3, -1)).mean(1)
    print "Got H={}, expecting H_ref={}.".format(H, expected_field)
    return np.max(np.abs(H - expected_field))

def test_time_dependent_field_update():
    field_expr = df.Expression(("0", "t", "0"), t=0)
    H_ext = TimeZeeman(field_expr)
    H_ext.setup(S3, m, Ms)

    assert diff(H_ext, np.array([0, 0, 0])) < TOL
    H_ext.update(1)
    assert diff(H_ext, np.array([0, 1, 0])) < TOL

def test_time_dependent_field_switched_off():
    field_expr = df.Expression(("0", "t", "0"), t=0)
    H_ext = TimeZeeman(field_expr, t_off=1)
    H_ext.setup(S3, m, Ms)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL
    H_ext.update(0.9)
    assert diff(H_ext, np.array([0, 0.9, 0])) < TOL
    H_ext.update(2)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL # It's off!

def test_discrete_time_zeeman_updates_in_intervals():
    field_expr = df.Expression(("0", "t", "0"), t=0)
    H_ext = DiscreteTimeZeeman(field_expr, dt_update=2)
    H_ext.setup(S3, m, Ms)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL
    H_ext.update(1)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL # not yet updating
    H_ext.update(3)
    assert diff(H_ext, np.array([0, 3, 0])) < TOL
