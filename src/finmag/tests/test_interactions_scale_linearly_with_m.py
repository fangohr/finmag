#!/usr/bin/env python

import dolfin as df
import numpy as np
import pytest
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman, Demag

np.random.seed(0)

mesh = df.BoxMesh(0, 0, 0, 40, 40, 5, 15, 15, 1)
N = mesh.num_vertices()
V = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)

randvec1 = np.random.random_sample(3*N)
randvec2 = np.random.random_sample(3*N)
randvec3 = np.random.random_sample(3*N)


def compute_field_for_linear_combination(EnergyClass, init_args, a, b, c):
    v = df.Function(V)
    v.vector()[:] = a * randvec1 + b * randvec2 + c * randvec3
    e = EnergyClass(**init_args)
    e.setup(V, v, Ms=8e5, unit_length=1e-9)
    return e.compute_field()


@pytest.mark.parametrize(("EnergyClass", "init_args"), [
        (Exchange, {'A': 13e-12}),
        (UniaxialAnisotropy, {'K1': 1e5, 'axis': (0, 0, 1)}),
        (Demag, {'solver_type': 'LU'}),
        pytest.mark.xfail((Demag, {'solver_type': 'Krylov'})),
        ])
def test_interactions_scale_linearly_with_m(EnergyClass, init_args):
    """
    For each energy class, compute the effective field for this interaction

    """

    b = 7.47
    c = 0.68

    fld_1 = compute_field_for_linear_combination(EnergyClass, init_args, 1.0, 0.0, 0.0)
    fld_2 = compute_field_for_linear_combination(EnergyClass, init_args, 0.0, 1.0, 0.0)
    fld_3 = compute_field_for_linear_combination(EnergyClass, init_args, 0.0, 0.0, 1.0)

    # Check a few linear combinations with different coefficients (a, b, c)
    for (a, b, c) in [(0.37, 7.47, 0.68), (2.0, 0.65, 0.1), (4.76, 3.3, 0.028), (10.1, 20.4, 12.6), (11.11, 42.0, 2.3)]:
        fld = compute_field_for_linear_combination(EnergyClass, init_args, a, b, c)
        print("   Testing linear combination with coefficients ({}, {}, {}): {}".format(a, b, c, np.allclose(fld, a * fld_1 + b * fld_2 + c * fld_3)))
        assert np.allclose(fld, a * fld_1 + b * fld_2 + c * fld_3)


if __name__ == '__main__':
    test_interactions_scale_linearly_with_m(Exchange, [13e-12])
