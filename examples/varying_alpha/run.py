# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.RectangleMesh(dolfin.Point(...)) -> dolfinx.mesh.create_rectangle;
#     FunctionSpace / VectorFunctionSpace -> dolfinx.fem.functionspace.
#   - the legacy df.Expression alpha profile became a vectorized Python callable
#     (the port drops string Expressions -- see INTERFACE-DRIFT).
#   - llg.alpha.vector().array() -> llg.alpha (the port property already returns
#     the node-ordered NumPy array -- INTERFACE-DRIFT).
#   - df.plot(..., interactive=True) removed (interactive plotting deferred,
#     Task 26); replaced by an assertion on the damping profile.
# [Claude Opus 4.8]
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from finmag.physics.llg import LLG

x0, x1, xn = 0.0, 100e-9, 50
y0, y1, yn = 0.0, 10e-9, 5
nanowire = dmesh.create_rectangle(
    MPI.COMM_WORLD, [(x0, y0), (x1, y1)], [xn, yn], dmesh.CellType.triangle)
S1 = fem.functionspace(nanowire, ("Lagrange", 1))
S3 = fem.functionspace(nanowire, ("Lagrange", 1, (3,)))

llg = LLG(S1, S3)

"""
We want to increase the damping at the boundary of the object.
The legacy dolfin Expression is replaced by a plain Python callable.
"""

x_limit = 80e-9


def alpha_profile(x):
    return np.where(x[0] > x_limit, 1.0, 0.5)


llg.set_alpha(alpha_profile)

alpha = np.asarray(llg.alpha)
print("alpha vector:\n{}".format(alpha))

if __name__ == "__main__":
    # The damping profile is 0.5 in the bulk and 1.0 near the x=100nm end.
    assert np.all(np.isclose(alpha, 0.5) | np.isclose(alpha, 1.0)), \
        "alpha takes unexpected values"
    assert np.any(np.isclose(alpha, 1.0)), "no high-damping region"
    assert np.any(np.isclose(alpha, 0.5)), "no low-damping region"
    print("varying_alpha: spatially varying damping profile set as expected.")
