import numpy as np
from finmag.util.consts import mu0
from finmag.energies.magnetostatic_field import MagnetostaticField


def test_magnetostatic_field_for_uniformly_magnetised_sphere():
    """
    Our favourite test when it comes to demag!

    """
    Ms = 1
    m = np.array((1, 0, 0))
    Nxx = Nyy = Nzz = 1.0 / 3.0  # demagnetising factors for sphere

    demag = MagnetostaticField(Ms, Nxx, Nyy, Nzz)
    H = demag.compute_field(m)
    H_expected = np.array((-1.0 / 3.0, 0.0, 0.0))
    print "Got demagnetising field H =\n{}.\nExpected mean H = {}.".format(H, H_expected)

    TOL = 1e-15
    diff = np.max(np.abs(H - H_expected))
    print "Maximum difference to expected result per axis is {}. Comparing to limit {}.".format(diff, TOL)
    assert np.max(diff) < TOL


def test_magnetostatic_energy_density_for_uniformly_magnetised_sphere():
    Ms = 1
    m = np.array((1, 0, 0))
    Nxx = Nyy = Nzz = 1.0 / 3.0  # demagnetising factors for sphere

    demag = MagnetostaticField(Ms, Nxx, Nyy, Nzz)
    E = demag.compute_energy(m)
    E_expected = (1.0 / 6.0) * mu0 * Ms ** 2
    print "Got E = {}. Expected E = {}.".format(E, E_expected)

    REL_TOL = 1e-15
    rel_diff = abs(E - E_expected) / abs(E_expected)
    print "Relative difference is {:.3g}%. Comparing to limit {:.3g}%.".format(
        100 * rel_diff, 100 * REL_TOL)
    assert rel_diff < REL_TOL
