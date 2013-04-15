from __future__ import division
import numpy as np
import dolfin as df
import pytest
from finmag.energies import Zeeman, TimeZeeman, DiscreteTimeZeeman
from finmag.util.consts import mu0
from math import sqrt, pi, cos

mesh = df.UnitCubeMesh(2, 2, 2)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
m = df.Function(S3)
m.assign(df.Constant((1, 0, 0)))
Ms = 1
TOL = 1e-14


def diff(H_ext, expected_field):
    """
    Helper function which computes the maximum deviation between H_ext
    and the expected field.
    """
    H = H_ext.compute_field().reshape((3, -1)).mean(1)
    print "Got H={}, expecting H_ref={}.".format(H, expected_field)
    return np.max(np.abs(H - expected_field))


def test_compute_energy():
    """
    Compute Zeeman energies of a cuboid and sphere for various
    arrangements.
    """
    lx = 2.0
    ly = 3.0
    lz = 5.0
    nx = ny = nz = 10  # XXX TODO: why does the approximation get
                       # worse if we use a finer mesh?!?
    unit_length = 1e-9
    Ms = 8e5
    h = 1e6

    mesh = df.BoxMesh(0, 0, 0, lx, ly, lz, nx, ny, nz)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    def check_energy_for_m(m, E_expected):
        """
        Helper function to compare the computed energy for a given
        magnetisation with an expected analytical value.
        """
        m_fun = df.Function(S3)
        m_fun.assign(df.Constant(m))
        H_ext = Zeeman(h * np.array([1, 0, 0]))
        H_ext.setup(S3, m_fun, Ms, unit_length=unit_length)

        E_computed = H_ext.compute_energy()
        assert np.allclose(E_computed, E_expected, atol=0, rtol=1e-12)

    volume = lx * ly * lz * unit_length**3
    E_aligned = -mu0 * Ms * h * volume

    check_energy_for_m((1, 0, 0), E_aligned)
    check_energy_for_m((-1, 0, 0), -E_aligned)
    check_energy_for_m((0, 1, 0), 0.0)
    check_energy_for_m((0, 0, 1), 0.0)
    check_energy_for_m((1/sqrt(2), 1/sqrt(2), 0), E_aligned * cos(pi * 45 / 180))
    check_energy_for_m((-0.5, 2/sqrt(3), 0), -E_aligned * cos(pi * 60 / 180))


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
    assert diff(H_ext, np.array([0, 0, 0])) < TOL  # It's off!


def test_discrete_time_zeeman_updates_in_intervals():
    field_expr = df.Expression(("0", "t", "0"), t=0)
    H_ext = DiscreteTimeZeeman(field_expr, dt_update=2)
    H_ext.setup(S3, m, Ms)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL
    H_ext.update(1)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL  # not yet updating
    H_ext.update(3)
    assert diff(H_ext, np.array([0, 3, 0])) < TOL


def test_discrete_time_zeeman_check_arguments_are_sane():
    """
    At least one of the arguments 'dt_update' and 't_off' must be given.
    """
    field_expr = df.Expression(("1", "2", "3"))
    with pytest.raises(ValueError):
        H_ext = DiscreteTimeZeeman(field_expr, dt_update=None, t_off=None)


def test_discrete_time_zeeman_switchoff_only():
    """
    Check that switching off a field works even if no dt_update is
    given (i.e. the field is just a pulse that is switched off after a
    while).
    """
    field_expr = df.Expression(("1", "2", "3"))
    H_ext = DiscreteTimeZeeman(field_expr, dt_update=None, t_off=2)
    H_ext.setup(S3, m, Ms)
    assert diff(H_ext, np.array([1, 2, 3])) < TOL
    H_ext.update(1)
    assert diff(H_ext, np.array([1, 2, 3])) < TOL  # not yet updating
    H_ext.update(2.1)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL
