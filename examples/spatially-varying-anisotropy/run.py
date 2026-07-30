"""
Demonstrating spatially varying anisotropy. Anisotropy easy axis is (0, 0, 1)
in the lower half of the film and (1, 0, 0) in the upper half -- a toy model of
an exchange-spring system.

DOLFINx port (Task 30): converted from the legacy dolfin example.
Changes vs legacy (all mechanically necessary for the ported package):
  - `import dolfin` / pylab removed; VectorFunctionSpace -> dolfinx.fem.
  - the legacy df.Expression easy-axis field became a vectorized Python
    callable (the port drops string Expressions in favour of callables --
    see INTERFACE-DRIFT: Expression-strings). UniaxialAnisotropy(K1, a) with a
    spatially varying Field axis is unchanged (Task 16).
  - the legacy pointwise probing `sim.m_field((x,y,z))` / `a((x,y,z))` is not
    ported (Field.__call__ raises NotImplementedError -- see INTERFACE-DRIFT:
    point-probing); the profile is read from Field.coords_and_values() instead.
  - matplotlib profile plot / save_pvd removed (deferred, Task 26); replaced by
    physical sanity assertions so the example self-validates.
[Claude Opus 4.8]
"""
import os
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem

from finmag import Simulation
from finmag.field import Field
from finmag.energies import UniaxialAnisotropy, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_simulation():
    mu0 = 4.0 * np.pi * 10**-7
    Ms = 1.0e6
    A = 13.0e-12
    Km = 0.5 * mu0 * Ms**2
    lexch = (A / Km)**0.5
    unit_length = 1e-9
    K1 = Km

    L = lexch / unit_length
    nx, ny, nz = 10, 1, 30
    Lx, Ly, Lz = nx * L, ny * L, nz * L
    mesh = dmesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (Lx, Ly, Lz)],
        [nx, ny, nz], dmesh.CellType.tetrahedron)

    boundary = Lz / 2.0

    # Easy axis: (0,0,1) for z<=boundary, (1,0,0) above. Vectorized callable
    # replacing the legacy df.Expression("x[2] <= b ? 0 : 1", "0", ...).
    def easy_axis(x):
        lower = x[2] <= boundary
        ax = np.where(lower, 0.0, 1.0)
        az = np.where(lower, 1.0, 0.0)
        return np.array([ax, np.zeros_like(ax), az])

    V = fem.functionspace(mesh, ("DG", 0, (3,)))
    a = Field(V, easy_axis)

    sim = Simulation(mesh, Ms, unit_length)
    sim.set_m((1, 0, 1))
    sim.add(UniaxialAnisotropy(K1, a))
    sim.add(Exchange(A))
    sim.relax()

    # Read the relaxed magnetisation at the CG1 mesh nodes (the easy-axis field
    # is DG0 / per-cell so it is not vertex-probed here; its effect is validated
    # through the magnetisation response below).
    coords, m_vals = sim.m_field.coords_and_values()
    zs = coords[:, 2]
    return zs, m_vals, boundary


if __name__ == "__main__":
    zs, m_vals, boundary = run_simulation()

    # |m| = 1 at every node.
    norms = np.linalg.norm(m_vals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "magnetisation is not unit-length"

    below = zs < boundary - 2.0
    above = zs > boundary + 2.0
    # Exchange spring: m follows the anisotropy -- more m_z in the lower half,
    # more m_x in the upper half.
    assert np.mean(m_vals[below, 2]) > np.mean(m_vals[above, 2]), \
        "lower half should be more z-aligned"
    assert np.mean(m_vals[above, 0]) > np.mean(m_vals[below, 0]), \
        "upper half should be more x-aligned"
    print("spatially-varying-anisotropy: exchange-spring profile relaxed, |m|=1.")
