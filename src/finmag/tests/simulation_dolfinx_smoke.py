"""Task 10 core DOLFINx smoke: end-to-end Simulation workflow witness.

Runnable as ``python -m finmag.tests.simulation_dolfinx_smoke`` (see the
``dolfinx-src-core-smoke`` pixi task), matching the executable-module
convention of the existing ``*_mpi_probe.py`` witnesses.

This builds the Task 9 core ``Simulation`` on a small 3D box with the three
ported interactions (Exchange, Zeeman, UniaxialAnisotropy) on the default
backend, advances to ``1e-12`` s with ``run_until`` (both ported drivers step
adaptively rather than with a fixed dt), and prints a short JSON
witness: the final time reached, the average magnetisation, and the maximum
nodal deviation from unit norm. A failing physics/tolerance check is a plain
``assert``, which raises and makes the module exit with a nonzero status.

SR1 P1.3: the default backend is now the native Sundials/CVODE driver, so this
smoke also pins the witnessed backend provenance -- it is the end-to-end
witness that the flipped public default really steps. [Claude Opus 4.8]
"""

import json

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.sim.sim import Simulation

T_TARGET = 1e-12
UNIT_NORM_TOL = 1e-5


def main():
    box = mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)],
        [2, 2, 2],
        mesh.CellType.tetrahedron,
    )
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name="core_smoke")
    sim.alpha = 0.5
    sim.set_m((1.0, 0.0, 0.0))
    sim.add(Exchange(13.0e-12))
    sim.add(Zeeman((0.0, 0.0, 1e6)))
    sim.add(UniaxialAnisotropy(1e5, (0.0, 0.0, 1.0)))

    sim.run_until(T_TARGET)

    m_nodal = sim.m.reshape((3, -1))
    max_unit_norm_deviation = float(
        np.max(np.abs(1.0 - np.linalg.norm(m_nodal, axis=0)))
    )
    m_average = [float(component) for component in sim.m_average]

    witness = {
        "t": float(sim.t),
        "t_target": T_TARGET,
        "m_average": m_average,
        "max_unit_norm_deviation": max_unit_norm_deviation,
        "integrator_backend": sim.integrator_backend,
    }

    assert sim.integrator_backend == "sundials", (
        "default integrator backend is not the restored native Sundials "
        "default: {}".format(witness)
    )
    assert type(sim.integrator).__name__ == "SundialsIntegrator", (
        "default backend did not instantiate the native Sundials driver: "
        "{}".format(type(sim.integrator).__name__)
    )
    assert sim.t >= T_TARGET, (
        "run_until did not reach the requested time: {}".format(witness)
    )
    assert max_unit_norm_deviation < UNIT_NORM_TOL, (
        "unit-norm invariant violated: {}".format(witness)
    )

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(witness, sort_keys=True))


if __name__ == "__main__":
    main()
