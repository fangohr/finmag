import pytest
import numpy as np
import dolfin as df
from finmag.util.meshes import sphere
from finmag.energies import Demag

TOL = 1e-2
solvers = ['FK', 'GCR', 'Treecode']


@pytest.fixture(scope="module")
def uniformly_magnetised_sphere():
    Ms = 1
    mesh = sphere(r=1, maxh=0.25)
    S3 = df.VectorFunctionSpace(mesh, "CG", 1)
    m = df.Function(S3)
    m.assign(df.Constant((1, 0, 0)))

    solutions = []
    for solver in solvers:
        demag = Demag(solver)
        demag.setup(S3, m, Ms, unit_length=1e-9)
        demag.H = demag.compute_field()
        solutions.append(demag)
    return solutions


def test_H_demag(uniformly_magnetised_sphere):
    for solution in uniformly_magnetised_sphere:
        H = solution.H.reshape((3, -1)).mean(1)
        H_expected = np.array([-1.0 / 3.0, 0, 0])
        print "{}: Hx = {}, should be {}.".format(
            solution.__class__.__name__, H, H_expected)
        diff = np.max(np.abs(H - H_expected))
        assert diff < TOL


def test_H_demag_deviation(uniformly_magnetised_sphere):
    for solution in uniformly_magnetised_sphere:
        H = solution.H.reshape((3, -1))
        delta = np.max(np.abs(H.max(axis=1) - H.min(axis=1)))
        assert delta < TOL


if __name__ == "__main__":
    uniformly_magnetised_sphere()
