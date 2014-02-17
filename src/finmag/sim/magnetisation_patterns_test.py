import numpy as np
import finmag
from finmag.sim.magnetisation_patterns import *
from finmag.util.meshes import cylinder


def test_vortex_functions():
    """
    Testing for correct polarity and 'handiness' of the two vortex functions,
    vortex_simple() and vortex_feldtkeller()
    """

    mesh = cylinder(10, 1, 3)
    coords = mesh.coordinates()

    def functions(hand, p):
        f_simple = vortex_simple(r=10.1, center=(0, 0, 1),
                                 right_handed=hand, polarity=p)
        f_feldtkeller = vortex_feldtkeller(beta=15, center=(0, 0, 1),
                                           right_handed=hand, polarity=p)
        return [f_simple, f_feldtkeller]

    # The polarity test evaluates the function at the mesh coordinates and
    # checks that the polarity of z-component from this matches the user input
    # polarity
    def polarity_test(func, coords, p):
        assert(np.alltrue([(p * func(coord)[2] > 0) for coord in coords]))

    # This function finds cross product of radius vector and the evaluated
    # function vector, rxm. The z- component of this will be:
    #	- negative for a clockwise vortex
    #	- positive for a counter-clockwise vortex
    # When (rxm)[2] is multiplied by the polarity, p, (rxm)[2] * p is:
    #	- negative for a left-handed state
    #	- positive for a right-handed state
    def handiness_test(func, coords, hand, p):
        r = coords
        m = [func(coord) for coord in coords]
        cross_product = np.cross(r, m)
        if hand is True:
            assert(np.alltrue((cross_product[:, 2] * p) > 0))
        elif hand is False:
            assert(np.alltrue((cross_product[:, 2] * p) < 0))

    # run the tests
    for hand in [True, False]:
        for p in [-1, 1]:
            funcs = functions(hand, p)
            for func in funcs:
                polarity_test(func, coords, p)
                handiness_test(func, coords, hand, p)

    # Final sanity check: f_simple should yield zero z-coordinate
    # outside the vortex core radius, and the magnetisation should
    # curl around the center.
    f_simple = vortex_simple(r=20, center=(0, 0, 1),
                             right_handed=True, polarity=1)

    assert(np.allclose(f_simple((21, 0, 0)), [0, 1, 0]))
    assert(np.allclose(f_simple((-16, 16, 20)),
                       [-1. / np.sqrt(2), -1. / np.sqrt(2), 0]))
