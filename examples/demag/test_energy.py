# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print(); dropped the unused `py` import (pytest is
#     kept for the restored @pytest.mark.slow marker).
#   - `import dolfin` removed; VectorFunctionSpace/FunctionSpace -> dolfinx.fem;
#     df.interpolate(df.Constant(...)) -> Field(S3, constant).
#   - from_geofile("sphere_fine.geo") kept UNCHANGED: the Netgen-CSG '.geo'
#     loader was ported for the examples subset (Task 30 amendment).
#   - the legacy @pytest.mark.slow marker and the demagenergies.txt run artifact
#     (gitignored) are restored.
# [Claude Opus 4.8]
import os
import logging
import pytest
from numpy import pi
from dolfinx import fem
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
    E, error = demag_energy()
    assert error < TOL


def demag_energy():
    mesh = from_geofile(os.path.join(MODULE_DIR, "sphere_fine.geo"))
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    m = Field(S3, (1.0, 0.0, 0.0))

    demag = Demag('FK')
    demag.setup(m, Field(fem.functionspace(mesh, ("DG", 0)), Ms), unit_length=1)

    E = demag.compute_energy()
    rel_error = abs(E - E_analytical) / abs(E_analytical)
    print("Energy with FK method: {}.".format(E))
    return E, rel_error


if __name__ == '__main__':
    E, error = demag_energy()
    with open(energy_file, "w") as f:
        f.write("FK Method: E = {}, relative error = {}.\n".format(E, error))
    print("FK Method: E = {}, relative error = {}.".format(E, error))
    assert error < TOL, "relative error {} exceeds tolerance {}".format(error, TOL)
    print("OK: FK demag energy of a uniform sphere matches mu0*Ms^2*V/6.")
