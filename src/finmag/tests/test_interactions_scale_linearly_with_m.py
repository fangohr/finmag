#!/usr/bin/env python

import dolfin as df
import numpy as np
from finmag.field import Field
import pytest
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman, Demag, DMI

np.random.seed(0)

mesh = df.BoxMesh(0, 0, 0, 40, 40, 5, 15, 15, 1)
N = mesh.num_vertices()
V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)

randvec1 = np.random.random_sample(3*N)
randvec2 = np.random.random_sample(3*N)
randvec3 = np.random.random_sample(3*N)


def compute_field_for_linear_combination(EnergyClass, init_args, a, b, c):
    v = Field(V)
    v.set_with_numpy_array_debug(a * randvec1 + b * randvec2 + c * randvec3)
    e = EnergyClass(**init_args)
    e.setup(v, Ms=8e5, unit_length=1e-9)
    return e.compute_field()


def create_demag_params(atol, rtol, maxiter):
    """
    Helper function to create a dictionary with the given
    demag tolerances and maximum iterations. This can be
    directly passed to the Demag class in order to set
    these parameters.
    """
    demag_params = {
        'absolute_tolerance': atol,
        'relative_tolerance': rtol,
        'maximum_iterations': int(maxiter),
        }
    return {'phi_1': demag_params, 'phi_2': demag_params}


# All the interactions should be linear in the magnetisation. However,
# for the demag field, this is only true if we use a LU solver or a
# Krylov solver with sufficiently strict tolerances. All the values of
# 'TOL' used in the tests below are the strictest that still make the
# tests pass. It is interesting to see that the various interactions
# have different accuracies (e.g. UniaxialAnisotropy is essentially
# linear in m up to machine precision whereas the Exchange and the
# demag are much less accurate).
@pytest.mark.parametrize(("EnergyClass", "init_args", "TOL"), [
        (Exchange, {'A': 13e-12}, 1e-11),
        (DMI, {'D': 1.58e-3}, 1e-12),
        (UniaxialAnisotropy, {'K1': 1e5, 'axis': (0, 0, 1)}, 1e-15),
        # Demag with LU solver should be linear in m
        (Demag, {'solver_type': 'LU'}, 1e-10),
        # Demag with Krylov solver and strict tolerances should be linear in m
        (Demag, {'solver_type': 'Krylov', 'parameters': create_demag_params(1e-15, 1e-15, 1e4)}, 1e-10),
        # Demag with Krylov solver and weak tolerances is *not* linear in m
        pytest.mark.xfail((Demag, {'solver_type': 'Krylov', 'parameters': create_demag_params(1e-6, 1e-6, 1e4)}, 1e-8)),
        ])
def test_interactions_scale_linearly_with_m(EnergyClass, init_args, TOL):
    """
    For each energy class, compute the associated effective field
    for three random configurations of m. Then compute it for various
    linear combinations of and check that the result is the same
    linear combination of the individual effective fields.

    """

    fld_1 = compute_field_for_linear_combination(EnergyClass, init_args, 1.0, 0.0, 0.0)
    fld_2 = compute_field_for_linear_combination(EnergyClass, init_args, 0.0, 1.0, 0.0)
    fld_3 = compute_field_for_linear_combination(EnergyClass, init_args, 0.0, 0.0, 1.0)

    # Check a few linear combinations with different coefficients (a, b, c)
    for (a, b, c) in [(0.37, 7.47, 0.68), (2.0, 0.65, 0.1), (4.76, 3.3, 0.028), (1.01, 2.04, 1.26), (1.111, 4.20, 2.3)]:
        fld = compute_field_for_linear_combination(EnergyClass, init_args, a, b, c)
        print("   Testing linear combination with coefficients ({}, {}, {}): {}".format(a, b, c, np.allclose(fld, a * fld_1 + b * fld_2 + c * fld_3, atol=TOL, rtol=TOL)))
        assert np.allclose(fld, a * fld_1 + b * fld_2 + c * fld_3, atol=TOL, rtol=TOL)


if __name__ == '__main__':
    test_interactions_scale_linearly_with_m(Exchange, [13e-12])
