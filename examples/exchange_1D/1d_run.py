# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.Interval -> dolfinx.mesh.create_interval; FunctionSpace /
#     VectorFunctionSpace -> dolfinx.fem.functionspace.
#   - llg.set_m() with legacy string-Expression components became a vectorized
#     Python callable (the port drops string Expressions -- see
#     INTERFACE-DRIFT: Expression-strings).
#   - mesh.coordinates() -> mesh.geometry.x (dolfinx API).
#   - llg.m (the state vector) is exposed as llg.m_numpy in the port
#     (INTERFACE-DRIFT: llg.m -> llg.m_numpy).
# The low-level LLG interface (LLG(S1, S3), .pins, effective_field.add,
# .solve_for) is otherwise unchanged.
# [Claude Opus 4.8]
import os
import numpy as np
from mpi4py import MPI
from scipy.integrate import odeint
from dolfinx import mesh as dmesh, fem
from finmag.physics.llg import LLG
from finmag.energies import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Compute the behaviour of a one-dimensional strip of magnetic material,
with exchange interaction.
"""

A = 1.3e-11
Ms = 8.6e5

length = 20e-9  # in meters
simplexes = 10
mesh = dmesh.create_interval(MPI.COMM_WORLD, simplexes, [0.0, length])
S1 = fem.functionspace(mesh, ("Lagrange", 1))
S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))

llg = LLG(S1, S3)


def m_init(x):
    u = 2 * x[0] / length - 1.0
    return np.array([u, np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0)), np.zeros_like(u)])


llg.set_m(m_init)
llg.pins = [0, 10]
exchange = Exchange(A)
llg.effective_field.add(exchange)

print("Solving problem...")

ts = np.linspace(0, 1e-9, 10)
ys, infodict = odeint(llg.solve_for, llg.m_numpy, ts, full_output=True)

print("Used {} function evaluations.".format(infodict["nfe"][-1]))
print("Saving data...")

np.savetxt(os.path.join(MODULE_DIR, "1d_times.txt"), ts)
np.savetxt(os.path.join(MODULE_DIR, "1d_M.txt"), ys)
np.savetxt(os.path.join(MODULE_DIR, "1d_coord.txt"), mesh.geometry.x[:, 0])

print("Done.")

if __name__ == "__main__":
    # Self-validation: |m| = 1 at every node and time, and the pinned end nodes
    # stay fixed at their initial orientation.
    m0 = ys[0].reshape(3, -1)
    for row in ys:
        comp = row.reshape(3, -1)
        norms = np.linalg.norm(comp, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-6), "magnetisation not unit-length"
    mfinal = ys[-1].reshape(3, -1)
    assert np.allclose(mfinal[:, 0], m0[:, 0], atol=1e-6), "pin 0 moved"
    assert np.allclose(mfinal[:, 10], m0[:, 10], atol=1e-6), "pin 10 moved"
    assert not np.allclose(mfinal, m0, atol=1e-3), "strip did not evolve"
    print("exchange_1D: 1D strip relaxed, |m|=1, pinned ends held fixed.")
