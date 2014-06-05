from __future__ import division
import numpy as np
import dolfin as df
import pytest
import os
import finmag
import logging
from finmag import sim_with
from finmag.energies import Zeeman, TimeZeeman, DiscreteTimeZeeman, OscillatingZeeman
from finmag.util.consts import mu0
from finmag.util.meshes import pair_of_disks
from finmag.example import sphere_inside_airbox
from math import sqrt, pi, cos, sin
from zeeman import DipolarField

mesh = df.UnitCubeMesh(2, 2, 2)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
m = df.Function(S3)
m.assign(df.Constant((1, 0, 0)))
Ms = 1
TOL = 1e-14

logger = logging.getLogger('finmag')


def diff(H_ext, expected_field):
    """
    Helper function which computes the maximum deviation between H_ext
    and the expected field.
    """
    H = H_ext.compute_field().reshape((3, -1)).mean(1)
    print "Got H={}, expecting H_ref={}.".format(H, expected_field)
    return np.max(np.abs(H - expected_field))


def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    field_expr = df.Expression(("0", "t", "0"), t=0)

    zeeman = Zeeman([0, 0, 1], name='MyZeeman')
    assert hasattr(zeeman, 'name')
    zeeman = TimeZeeman(field_expr, name='MyTimeZeeman')
    assert hasattr(zeeman, 'name')
    zeeman = DiscreteTimeZeeman(field_expr, dt_update=2, name='MyDiscreteTimeZeeman')
    assert hasattr(zeeman, 'name')


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
    H = 1e6

    mesh = df.BoxMesh(0, 0, 0, lx, ly, lz, nx, ny, nz)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    def check_energy_for_m(m, E_expected):
        """
        Helper function to compare the computed energy for a given
        magnetisation with an expected analytical value.
        """
        m_fun = df.Function(S3)
        m_fun.assign(df.Constant(m))
        H_ext = Zeeman(H * np.array([1, 0, 0]))
        H_ext.setup(S3, m_fun, Ms, unit_length=unit_length)

        E_computed = H_ext.compute_energy()
        assert np.allclose(E_computed, E_expected, atol=0, rtol=1e-12)

    volume = lx * ly * lz * unit_length**3
    E_aligned = -mu0 * Ms * H * volume

    check_energy_for_m((1, 0, 0), E_aligned)
    check_energy_for_m((-1, 0, 0), -E_aligned)
    check_energy_for_m((0, 1, 0), 0.0)
    check_energy_for_m((0, 0, 1), 0.0)
    check_energy_for_m((1/sqrt(2), 1/sqrt(2), 0), E_aligned * cos(pi * 45 / 180))
    check_energy_for_m((-0.5, 2/sqrt(3), 0), -E_aligned * cos(pi * 60 / 180))

def test_energy_density_function():
    """
    Compute the Zeeman energy density over the entire mesh, integrate it, and
    compare it to the expected result.
    """

    mesh = df.RectangleMesh(-50, -50, 50, 50, 10, 10)
    unit_length = 1e-9
    H = 1e6

    # Create simulation object.
    sim = finmag.Simulation(mesh, 1e5, unit_length=unit_length)

    # Set uniform magnetisation.
    def m_ferromagnetic(pos):
        return np.array([0., 0., 1.])

    sim.set_m(m_ferromagnetic)

    # Assign zeeman object to simulation
    sim.add(Zeeman(H * np.array([0., 0., 1.])))

    # Get energy density function
    edf = sim.get_interaction('Zeeman').energy_density_function()

    # Integrate it over the mesh and compare to expected result.
    total_energy = df.assemble(edf * df.dx) * unit_length
    expected_energy = -mu0 * H
    assert (total_energy + expected_energy) < 1e-6

def test_compute_energy_in_regions(tmpdir):
    os.chdir(str(tmpdir))
    d = 30.0
    h1 = 5.0
    h2 = 10.0
    sep = 10.0
    maxh = 2.5
    RTOL = 5e-3  # depends on maxh
    unit_length = 1e-9

    Ms = 8e5
    H = 1e6

    mesh = pair_of_disks(d, d, h1, h2, sep, theta=0, maxh=maxh)
    S3 = df.VectorFunctionSpace(mesh, "CG", 1)

    # Create a mesh function for the two domains (each representing one disk),
    # where the regions are marked with '0' (first disk) and '1' (second disk).
    class Disk1(df.SubDomain):
        def inside(self, pt, on_boundary):
            x, y, z = pt
            return np.linalg.norm([x, y]) < 0.5 * (d + sep)

    class Disk2(df.SubDomain):
        def inside(self, pt, on_boundary):
            x, y, z = pt
            return np.linalg.norm([x, y, z]) > 0.5 * (d + sep)

    disk1 = Disk1()
    disk2 = Disk2()
    domains = df.CellFunction("size_t", mesh)
    domains.set_all(0)
    disk1.mark(domains, 1)
    disk2.mark(domains, 2)
    dx = df.Measure("dx")[domains]
    dx_disk_1 = dx(1)
    dx_disk_2 = dx(2)

    volume_1 = pi * (0.5 * d)**2 * h1 * unit_length**3  # volume of disk #1
    volume_2 = pi * (0.5 * d)**2 * h2 * unit_length**3  # volume of disk #2
    #E_aligned_1 = -mu0 * Ms * H * volume_1  # energy of disk #1 if m || H_ext
    #E_aligned_2 = -mu0 * Ms * H * volume_2  # energy of disk #2 if m || H_ext

    def check_energies(m=None, theta=None):
        """
        Helper function to compare the computed energy for a given
        magnetisation with an expected analytical value. The argument
        theta is the angle between the magnetisation vector and the
        x-axis.

        """
        # Exactly one of m, theta must be given
        assert((m is None or theta is None) and not (m is None and theta is None))
        if m is None:
            if theta is None:
                raise ValueError("Exactly one of m, theta must be given.")
            theta_rad = theta * pi / 180.
            m = (cos(theta_rad), sin(theta_rad), 0)
        else:
            if theta != None:
                raise ValueError("Exactly one of m, theta must be given.")
        m = m / np.linalg.norm(m)
        m_fun = df.Function(S3)
        m_fun.assign(df.Constant(m))
        H_ext = Zeeman(H * np.array([1, 0, 0]))
        H_ext.setup(S3, m_fun, Ms, unit_length=unit_length)

        #E_analytical_1 = -mu0 * Ms * H * volume_1 * cos(theta_rad)
        E_analytical_1 = -mu0 * Ms * H * volume_1 * np.dot(m, [1, 0, 0])
        E_analytical_2 = -mu0 * Ms * H * volume_2 * np.dot(m, [1, 0, 0])
        E_analytical_total = E_analytical_1 + E_analytical_2

        E_computed_1 = H_ext.compute_energy(dx=dx_disk_1)
        E_computed_2 = H_ext.compute_energy(dx=dx_disk_2)
        E_computed_total = H_ext.compute_energy(dx=df.dx)

        # Check that the sum of the computed energies for disk #1 and #2 equals the total computed energy
        assert np.allclose(E_computed_1 + E_computed_2, E_computed_total, atol=0, rtol=1e-12)

        # Check that the computed energies are within the tolerance of the analytical expressions
        assert np.allclose(E_computed_1, E_analytical_1, atol=0, rtol=RTOL)
        assert np.allclose(E_computed_2, E_analytical_2, atol=0, rtol=RTOL)
        assert np.allclose(E_computed_total, E_analytical_total, atol=0, rtol=RTOL)
        #finmag.logger.debug("E_computed: {}".format(E_computed))
        #finmag.logger.debug("E_expected: {}".format(E_expected))
        #finmag.logger.debug("E_computed - E_expected: {}".format(E_computed - E_expected))

    # Check a bunch of energies
    for theta in [0, 20, 45, 60, 90, 144, 180]:
        check_energies(theta=theta)
        check_energies(theta=-theta)
    check_energies(m=(0, 0, 1))
    check_energies(m=(2, -3, -1))

def test_value_set_update():
    """
    Test to check that the value member variable updates when set_value is
    called.
    """
    init_value = [1., 2., 3.]
    second_value = [100., 200., 400.]

    zeeman = Zeeman(init_value)
    mesh = df.RectangleMesh(0, 0, 1, 1, 10, 10)
    sim = finmag.Simulation(mesh, 1e5)
    sim.add(zeeman)
    zeeman.set_value(second_value)

    assert zeeman.value == second_value

def test_time_zeeman_init():
    field_expr = df.Expression(("0", "t", "0"), t=0)
    field_lst = [1, 0, 0]
    field_tpl = (1, 0, 0)
    field_arr = np.array([1, 0, 0])
    # These should work
    TimeZeeman(field_expr)
    TimeZeeman(field_expr, t_off=1e-9)

    # These should *not* work, since there is no time update
    with pytest.raises(ValueError): TimeZeeman(field_lst, t_off=None)
    with pytest.raises(ValueError): TimeZeeman(field_tpl, t_off=None)
    with pytest.raises(ValueError): TimeZeeman(field_arr, t_off=None)

    # These *should* work, since there is a time update (the field is
    # switched off at some point)
    TimeZeeman(field_lst, t_off=1e-9)
    TimeZeeman(field_tpl, t_off=1e-9)
    TimeZeeman(field_arr, t_off=1e-9)


def test_time_dependent_field_update():
    field_expr = df.Expression(("0", "t", "0"), t=0)
    H_ext = TimeZeeman(field_expr)
    H_ext.setup(S3, m, Ms)

    assert diff(H_ext, np.array([0, 0, 0])) < TOL
    H_ext.update(1)
    assert diff(H_ext, np.array([0, 1, 0])) < TOL


def test_time_dependent_field_switched_off():
    # Check the time update (including switching off) with a varying field
    field_expr = df.Expression(("0", "t", "0"), t=0)
    H_ext = TimeZeeman(field_expr, t_off=1)
    H_ext.setup(S3, m, Ms)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL
    assert(H_ext.switched_off == False)
    H_ext.update(0.9)
    assert diff(H_ext, np.array([0, 0.9, 0])) < TOL
    assert(H_ext.switched_off == False)
    H_ext.update(2)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL  # It's off!
    assert(H_ext.switched_off == True)

    # The same again with a constant field
    a = [42, 0, 5]
    H_ext = TimeZeeman(a, t_off=1)
    H_ext.setup(S3, m, Ms)
    assert diff(H_ext, a) < TOL
    assert(H_ext.switched_off == False)
    H_ext.update(0.9)
    assert diff(H_ext, a) < TOL
    assert(H_ext.switched_off == False)
    H_ext.update(2)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL  # It's off!
    assert(H_ext.switched_off == True)



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
    assert(H_ext.switched_off == False)
    H_ext.update(1)
    assert diff(H_ext, np.array([1, 2, 3])) < TOL  # not yet updating
    assert(H_ext.switched_off == False)
    H_ext.update(2.1)
    assert diff(H_ext, np.array([0, 0, 0])) < TOL
    assert(H_ext.switched_off == True)


def test_oscillating_zeeman():
    """
    """
    def check_field_at_time(t, val):
        H_osc.update(t)
        a = H_osc.compute_field().reshape(3, -1).T
        assert(np.allclose(a, val, atol=0, rtol=1e-8))

    H = np.array([1e6, 0, 0])
    freq = 2e9
    t_off = 10e-9

    H_osc = OscillatingZeeman(H0=H, freq=freq, phase=0, t_off=t_off)
    H_osc.setup(S3, m, Ms)

    # Check that the field has the original value at the end of the
    # first few cycles.
    for i in xrange(19):
        check_field_at_time(i * 1.0/freq, H)

    # Check that the field is switched off at the specified time (and
    # stays switched off thereafter)
    assert(H_osc.switched_off == False)
    check_field_at_time(t_off, [0, 0, 0])
    assert(H_osc.switched_off == True)
    check_field_at_time(t_off + 1e-11, [0, 0, 0])
    assert(H_osc.switched_off == True)
    check_field_at_time(t_off + 1, [0, 0, 0])
    assert(H_osc.switched_off == True)

    # Check that the field values vary sinusoidally as expected
    phase = 0.1e-9
    H_osc = OscillatingZeeman(H0=H, freq=freq, phase=phase, t_off=None)
    H_osc.setup(S3, m, Ms)
    for t in np.linspace(0, 20e-9, 100):
        check_field_at_time(t, H * cos(2 * pi * freq * t + phase))


def test_dipolar_field_class(tmpdir):
    os.chdir(str(tmpdir))
    H_dipole = DipolarField(pos=[0, 0, 0], m=[1, 0, 0], magnitude=3e9)
    mesh = df.BoxMesh(-50, -50, -50, 50, 50, 50, 20, 20, 20)
    V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    H_dipole.setup(V, df.Constant([1, 0, 0]), Ms=8.6e5, unit_length=1e-9)


def compute_field_diffs(sim):
    vals_demag = sim.get_field_as_dolfin_function('Demag', region='air').vector().array().reshape(3, -1)
    vals_dipole = sim.get_field_as_dolfin_function('DipolarField', region='air').vector().array().reshape(3, -1)
    absdiffs = np.linalg.norm(vals_demag - vals_dipole, axis=0)
    reldiffs = absdiffs / np.linalg.norm(vals_dipole, axis=0)
    return absdiffs, reldiffs


@pytest.mark.slow
def test_compare_stray_field_of_sphere_with_dipolar_field(tmpdir, debug=True):
    """
    Check that the stray field of a sphere in an 'airbox'
    is close to the field of a point dipole with the same
    magnetic moment.

    """
    os.chdir(str(tmpdir))

    # Create a mesh of a sphere enclosed in an "airbox"
    m_init = [7, -4, 3]  # some random magnetisation direction
    center_sphere = [0, 0, 0]
    r_sphere = 3
    r_shell = 30
    l_box = 100
    maxh_sphere = 2.5
    maxh_shell = None
    maxh_box = 10.0
    Ms_sphere = 8.6e5

    sim = sphere_inside_airbox(r_sphere, r_shell, l_box, maxh_sphere, maxh_shell, maxh_box, center_sphere, m_init)
    if debug:
        sim.render_scene(field_name='Demag', region='air', representation='Outline', outfile='ddd_demag_field_air.png')
        sim.render_scene(field_name='Demag', region='sphere', representation='Outline', outfile='ddd_demag_field_sphere.png')

    # Add an external field representing a point dipole
    # (with the same magnetic moment as the sphere).
    dipole_magnitude = Ms_sphere * 4/3 * pi * r_sphere**3
    logger.debug("dipole_magnitude = {}".format(dipole_magnitude))
    H_dipole = DipolarField(pos=[0, 0, 0], m=m_init, magnitude=dipole_magnitude)
    sim.add(H_dipole)

    # Check that the absolute and relative difference between the
    # stray field of the sphere and the field of the point dipole
    # are below a given tolerance.
    absdiffs, reldiffs = compute_field_diffs(sim)
    assert np.max(reldiffs) < 0.4
    assert np.mean(reldiffs) < 0.15
    assert np.max(absdiffs) < 140.0

    print np.max(reldiffs), np.mean(reldiffs), np.max(absdiffs)
