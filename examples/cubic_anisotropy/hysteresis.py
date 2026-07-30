# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - dolfin.BoxMesh(dolfin.Point(...)) -> dolfinx.mesh.create_box.
#   - CubicAnisotropy(K1, u1, K2, u2, K3, u3) -> CubicAnisotropy(u1, u2, K1,
#     K2, K3). NOTE: this is NOT port drift -- the example was already STALE
#     against legacy master, whose CubicAnisotropy signature is (u1, u2, K1,
#     K2, K3) and computes u3 = cross(u1, u2) internally (the explicit u3 the
#     example passed, (-1,0,0), equals cross(u1,u2), so the physics is
#     unchanged). See the Task 30 report "stale examples" note.
#   - Gate mode: the field sweep is 6 points by default (fast); the full
#     250-point one-way sweep runs when FINMAG_EXAMPLE_FULL=1.
# The sim.hysteresis(fields, fun) API is unchanged (Task 15).
# [Claude Opus 4.8]
import os
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from finmag import Simulation
from finmag.energies import Exchange, CubicAnisotropy
from finmag.util.consts import mu0

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

mesh = dmesh.create_box(
    MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (1.0, 1.0, 40.0)],
    [1, 1, 40], dmesh.CellType.tetrahedron)

Ms = 876626  # A/m
A = 1.46e-11  # J/m

K1 = -8608726
K2 = -13744132
K3 =  1100269
u1 = (0, -0.7071, 0.7071)
u2 = (0,  0.7071, 0.7071)

npoints = 250 if os.environ.get("FINMAG_EXAMPLE_FULL") == "1" else 6

# fields close to oommf reference cubicEight_100pc.mif
fields = np.zeros((npoints, 3))
fields[:, 0] = 5
fields[:, 1] = 5
fields[:, 2] = np.linspace(20000, -20000, npoints)
fields = fields * 0.001 / mu0  # mT to A/m

sim = Simulation(mesh, Ms, unit_length=1e-9)
sim.set_m((0, 0, 1))
sim.add(Exchange(A))
sim.add(CubicAnisotropy(u1, u2, K1, K2, K3))

# this is not a hysteresis loop, but just a one-way swipe.
# INTERFACE-DRIFT: the ported (legacy-verbatim) sim.hysteresis() guards with
# `if H_ext_list == []`, which raises under modern NumPy when given an ndarray
# (it worked in the Python-2 / old-NumPy legacy). Pass a plain list of field
# 3-vectors instead of the (N,3) array. src/finmag is left untouched.
mzs = sim.hysteresis(list(fields), lambda sim: sim.m_average[2])
result = np.zeros((npoints, 2))
result[:, 0] = fields[:, 2]
result[:, 1] = mzs
np.savetxt(os.path.join(MODULE_DIR, "hysteresis.txt"), result,
           header="field in A/m and corresponding unit magnetisation in z-direction")

if __name__ == "__main__":
    mzs = np.asarray(mzs)
    assert np.all(np.abs(mzs) <= 1.0 + 1e-6), "|m_z| exceeded 1"
    # One-way sweep from +z-saturating field down to -z-saturating field:
    # the magnetisation switches from ~ +1 to ~ -1.
    assert mzs[0] > 0.9, "did not start z-saturated (m_z={})".format(mzs[0])
    assert mzs[-1] < -0.9, "did not switch to -z (m_z={})".format(mzs[-1])
    print("cubic_anisotropy/hysteresis: one-way sweep switches m_z from +1 to -1.")
