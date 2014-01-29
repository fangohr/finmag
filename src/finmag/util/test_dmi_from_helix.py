import DMI_from_helix as dfh

import numpy as np
import pytest

a = 3.53e-13
ms = 1.56e5
l = 22e-9
d0 = 4 * np.pi * a / l
hDirection = np.array([1., 0., 0.])
h = hDirection * 0.


def test_dmi_from_helix_solution():

    # Test the helix length solution with an example that is known to produce a
    # certain value.
    l = dfh.Find_Helix_Length(d0, a, ms, H=h)

    expectedSolution = 2.50e-8

    assert abs(l - expectedSolution) < 1e-9


def test_helix_strong_field():

    # Test whether a helix is found (it shouldn't be) in a very strong magnetic
    # field.
    h = hDirection * ms
    with pytest.raises(ValueError):
        dfh.Find_Helix_Length(d0, a, ms, H=h)

    # Perform the test with a strong negative field also.
    h -= 2 * h
    with pytest.raises(ValueError):
        dfh.Find_Helix_Length(d0, a, ms, H=h)

    # As well as a field in a funky direction.
    h = np.array([1., 1., 1.]) * ms
    with pytest.raises(ValueError):
        dfh.Find_Helix_Length(d0, a, ms, H=h)


def test_zero_helix():

    # Test whether a DMI value can be found for a helix of length zero (in the
    # absence of a magnetic field)
    l = 0.
    with pytest.raises(ValueError):
        dfh.Find_DMI(a, ms, l, H=h)
