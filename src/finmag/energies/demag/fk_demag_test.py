import dolfin as df
import numpy as np
from math import pi
from finmag.util.meshes import sphere
from finmag.energies.demag.fk_demag import FKDemag
from finmag.util.consts import mu0

radius = 1.0
maxh = 0.2
unit_length = 1e-9
volume = 4 * pi * (radius * unit_length) ** 3 / 3


def setup_demag_sphere(Ms):
    mesh = sphere(r=radius, maxh=maxh)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.Function(S3)
    m.assign(df.Constant((1, 0, 0)))
    demag = FKDemag()
    demag.setup(S3, m, Ms, unit_length)
    return demag


def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    demag = FKDemag(name='MyDemag')
    assert hasattr(demag, 'name')


def test_demag_field_for_uniformly_magnetised_sphere():
    demag = setup_demag_sphere(Ms=1)
    H = demag.compute_field().reshape((3, -1))
    H_expected = np.array([-1.0 / 3.0, 0.0, 0.0])
    print "Got demagnetising field H =\n{}.\nExpected mean H = {}.".format(
        H, H_expected)

    TOL = 7e-3
    diff = np.max(np.abs(H - H_expected[:, np.newaxis]), axis=1)
    print "Maximum difference to expected result per axis is {}. Comparing to limit {}.".format(diff, TOL)
    assert np.max(diff) < TOL

    TOL = 8e-3
    spread = np.abs(H.max(axis=1) - H.min(axis=1))
    print "The values spread {} per axis. Comparing to limit {}.".format(spread, TOL)
    assert np.max(spread) < TOL


def test_demag_energy_for_uniformly_magnetised_sphere():
    Ms = 800e3
    demag = setup_demag_sphere(Ms)
    E = demag.compute_energy()
    E_expected = (1.0 / 6.0) * mu0 * Ms ** 2 * volume  # -mu0/2 Integral H * M with H = - M / 3
    print "Got E = {}. Expected E = {}.".format(E, E_expected)

    REL_TOL = 3e-2
    rel_diff = abs(E - E_expected) / abs(E_expected)
    print "Relative difference is {:.3g}%. Comparing to limit {:.3g}%.".format(
        100 * rel_diff, 100 * REL_TOL)
    assert rel_diff < REL_TOL


def test_energy_density_for_uniformly_magnetised_sphere():
    Ms = 800e3
    demag = setup_demag_sphere(Ms)
    rho = demag.energy_density()

    E_expected = (1.0 / 6.0) * mu0 * Ms**2 * volume  # -mu0/2 Integral H * M with H = - M / 3
    rho_expected = E_expected / volume
    print "Got mean rho = {:.3e}. Expected rho = {:.3e}.".format(np.mean(rho), rho_expected)

    REL_TOL = 1.7e-2
    rel_diff = np.max(np.abs(rho - rho_expected)) / abs(rho_expected)
    print "Maximum relative difference = {:.3g}%. Comparing to limit {:.3g}%.".format(
        100 * rel_diff, 100 * REL_TOL)
    assert rel_diff < REL_TOL


def test_energy_density_for_uniformly_magnetised_sphere_as_function():
    Ms = 800e3
    demag = setup_demag_sphere(Ms)
    rho = demag.energy_density_function()
    print "Probing the energy density at the center of the sphere."
    rho_center = rho([0.0, 0.0, 0.0])

    E_expected = (1.0 / 6.0) * mu0 * Ms**2 * volume  # -mu0/2 Integral H * M with H = - M / 3
    rho_expected = E_expected / volume
    print "Got rho = {:.3e}. Expected rho = {:.3e}.".format(rho_center, rho_expected)

    REL_TOL = 1.3e-2
    rel_diff = np.max(np.abs(rho_center - rho_expected)) / abs(rho_expected)
    print "Maximum relative difference = {:.3g}%. Comparing to limit {:.3g}%.".format(
        100 * rel_diff, 100 * REL_TOL)
    assert rel_diff < REL_TOL

def test_regression_Ms_numpy_type():
    mesh = sphere(r=radius, maxh=maxh)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    m = df.Function(S3)
    m.assign(df.Constant((1, 0, 0)))

    Ms = np.sqrt(6.0 / mu0)  # math.sqrt(6.0 / mu0) would work
    demag = FKDemag()
    demag.setup(S3, m, Ms, unit_length)  # this used to fail
