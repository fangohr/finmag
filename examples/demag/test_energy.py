import os
import py
import pytest
import logging
import dolfin as df
from numpy import pi, sqrt
from finmag.energies import Demag
from finmag.field import Field
from finmag.util.meshes import from_geofile
from finmag.util.consts import mu0

log = logging.getLogger("finmag")
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
energy_file = os.path.join(MODULE_DIR, "demagenergies.txt")
Ms = 1e5
volume = 4 * pi / 3
E_analytical = mu0 * Ms**2 * volume / 6
TOL = 1.9e-2


@pytest.mark.slow
def test_demag_energy_fk():
    E, error = demag_energy("FK")
    assert error < TOL


@pytest.mark.xfail
@pytest.mark.slow
def test_demag_energy_gcr():
    E, error = demag_energy("GCR")
    assert error < TOL


def demag_energy(solver):
    mesh = from_geofile(os.path.join(MODULE_DIR, "sphere_fine.geo"))
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m_function = df.interpolate(df.Constant((1, 0, 0)), S3)
    m = Field(S3, m_function)

    demag = Demag(solver)
    demag.setup(m, Field(df.FunctionSpace(mesh, 'DG', 0), Ms), unit_length=1)

    E = demag.compute_energy()
    rel_error = abs(E - E_analytical) / abs(E_analytical)
    print "Energy with {} method: {}.".format(solver, E)
    return E, rel_error


if __name__ == '__main__':
    with open(energy_file, "w") as f:
        for solver in ["FK", "GCR"]:
            try:
                E, error = demag_energy(solver)
            except Exception as e:
                log.warning("Could not add {} demag energy to documentation example.".format(solver))
                print e
            else:
                f.write("{}: E = {}, relative error = {}.\n".format(solver, E, error))
