"""Generate coordinate-ordered legacy FK demag references (Task 11b review
Findings 1 and 2).

Runs at the immutable FEniCS-2019 oracle. Builds the two Task 11b meshes
(a unit cube and the "barmini" bar) carrying a uniform magnetisation, runs
the frozen legacy ``finmag.energies.demag.fk_demag.FKDemag`` two-potential
solver, and dumps coordinate-sorted per-vertex ``H``, together with the
scalar energy and average field, to JSON on stdout.

Three cases are generated:

- ``cube``: ``UnitCubeMesh(4, 4, 4)``, standard legacy solver tolerances
  (``relative_tolerance = absolute_tolerance = 1e-6``, the legacy default).
- ``cube_tight_tolerance``: the same mesh/physics, with both Krylov solves
  (``phi_1``, ``phi_2``) tightened to ``1e-12``. This isolates whether the
  ~6.4e-6 pointwise gap between the DOLFINx port (at the same 1e-6 default)
  and the frozen oracle fixture is caused by the *oracle's own* frozen 1e-6
  Krylov solves rather than a systematic method difference between the port
  and the legacy formulation: compare this datum against the port run with
  its own KSP rtol/atol tightened to 1e-12 (see
  ``test_oracle_cube_tight_tolerance_isolates_krylov_residual`` in
  ``test_fk_demag_dolfinx.py``). Empirically (2026-07-21), tightening
  *both* sides collapses the pointwise gap from 6.4e-6 (relative to
  |H|_max) to ~1.4e-12, and the energy gap from 4.7e-8 to ~5.5e-14 -- proof
  the standard-tolerance residual is attributable to the oracle's own frozen
  1e-6 solves, not the discretisation or the DOLFINx port.
- ``barmini``: the 3x3x10 nm bar, standard legacy solver tolerances.

Each case's ``H_vertex`` tolerance is generated *from the oracle's own
run-to-run conditioning*, not from an observed difference against the port:
the pointwise ``absolute`` tolerance is a fixed multiple of the Krylov
solver's linear-algebra tolerance times the field's own scale
(``KRYLOV_TOL_FACTOR`` below), which is a numerical-conditioning argument
(the KSP is converged to ``relative_tolerance`` fractional residual on the
right-hand side, which propagates to an ``O(relative_tolerance * scale)``
error in the recovered field), independent of what the port happens to
produce.
"""
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag.field import Field
from finmag.energies.demag.fk_demag import FKDemag

# Multiple of the Krylov solver's relative_tolerance used as the pointwise
# H_vertex absolute-tolerance floor (numerical-conditioning argument, not
# fit to an observed port/oracle difference: the KSP residual on the RHS
# propagates at this order of magnitude into the recovered nodal field).
KRYLOV_TOL_FACTOR = 100.0

STANDARD_PARAMETERS = {
    "phi_1": {"relative_tolerance": 1e-6, "absolute_tolerance": 1e-6,
              "maximum_iterations": int(1e4)},
    "phi_2": {"relative_tolerance": 1e-6, "absolute_tolerance": 1e-6,
              "maximum_iterations": int(1e4)},
}
TIGHT_PARAMETERS = {
    "phi_1": {"relative_tolerance": 1e-12, "absolute_tolerance": 1e-12,
              "maximum_iterations": int(1e5)},
    "phi_2": {"relative_tolerance": 1e-12, "absolute_tolerance": 1e-12,
              "maximum_iterations": int(1e5)},
}


def _run_case(mesh, m_vec, Ms, unit_length, solver_parameters):
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    DG = df.FunctionSpace(mesh, "DG", 0)

    mv = np.asarray(m_vec, dtype=float)
    mv = mv / np.linalg.norm(mv)
    m = Field(S3, tuple(mv))
    Ms_field = Field(DG, Ms)

    demag = FKDemag(solver_type="Krylov", parameters=solver_parameters)
    demag.setup(m, Ms_field, unit_length)

    E = demag.compute_energy()
    avg = demag.average_field()

    H_flat = demag.compute_field()
    Hf = Field(S3)
    Hf.f.vector().set_local(H_flat)
    Hf.f.vector().apply("insert")
    H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(-1, 3)

    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)
    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))

    return {
        "energy": E,
        "average_field": avg,
        "coordinates": coords3[order],
        "H_vertex": H_xyz[order],
        "m": mv,
    }


def _case_doc(name, description, mesh_recipe, mesh_parameters, Ms,
              unit_length, solver_parameters, result, energy_tol,
              avg_tol, generator_argv):
    H = result["H_vertex"]
    scale = float(np.abs(H).max())
    rtol = solver_parameters["phi_1"]["relative_tolerance"]
    h_atol = KRYLOV_TOL_FACTOR * rtol * scale

    return {
        "oracle": {
            "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
            "command": generator_argv,
            "generator": "src/finmag/tests/fixtures/gen_fk_demag_oracle.py",
            "generator_note": (
                "Committed driver run at the oracle by absolute path via "
                "dev/bin/run-legacy-oracle (the script does not exist inside "
                "the detached oracle checkout, so it is supplied by absolute "
                "path). H_vertex/energy/average_field are computed by the "
                "frozen legacy FKDemag; the driver only sorts and serialises "
                "coordinate-paired nodal values."
            ),
        },
        "description": description,
        "mesh": {"recipe": mesh_recipe, "parameters": mesh_parameters},
        "physical_parameters": {
            "Ms": {"value": Ms, "unit": "A/m"},
            "unit_length": {"value": unit_length, "unit": "m"},
            "m": {"value": result["m"].tolist(), "unit": "1"},
        },
        "solver_parameters": {
            "relative_tolerance": {"value": rtol, "unit": "1"},
            "absolute_tolerance": {
                "value": solver_parameters["phi_1"]["absolute_tolerance"],
                "unit": "1"},
            "maximum_iterations": {
                "value": solver_parameters["phi_1"]["maximum_iterations"],
                "unit": "1"},
        },
        "coordinates": {
            "unit": "mesh_coordinate",
            "ordering": "lexicographic_xyz",
            "values": result["coordinates"].tolist(),
        },
        "quantities": [
            {
                "name": "H_vertex",
                "unit": "A/m",
                "value_shape": [3],
                "values": H.tolist(),
                "tolerances": {"absolute": h_atol, "relative": 0.0},
            },
        ],
        "scalar_quantities": [
            {
                "name": "energy",
                "unit": "J",
                "value_shape": [],
                "value": result["energy"],
                "tolerances": {"absolute": 0.0, "relative": energy_tol},
            },
            {
                "name": "average_field",
                "unit": "A/m",
                "value_shape": [3],
                "value": result["average_field"].tolist(),
                "tolerances": {"absolute": 1.0, "relative": avg_tol},
            },
        ],
    }


def main():
    generator_argv = [
        "dev/bin/run-legacy-oracle",
        "--",
        "pixi",
        "run",
        "--locked",
        "env",
        "PYTHONPATH=src",
        "python",
        os.path.abspath(__file__),
    ]

    cube_mesh = df.UnitCubeMesh(4, 4, 4)
    cube_std = _run_case(cube_mesh, (1.0, 0.0, 0.0), 8.0e5, 1e-9,
                         STANDARD_PARAMETERS)
    cube_tight_mesh = df.UnitCubeMesh(4, 4, 4)
    cube_tight = _run_case(cube_tight_mesh, (1.0, 0.0, 0.0), 8.0e5, 1e-9,
                           TIGHT_PARAMETERS)

    barmini_mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(3, 3, 10), 2, 2, 4)
    barmini_std = _run_case(barmini_mesh, (1.0, 0.0, 1.0), 0.86e6, 1e-9,
                            STANDARD_PARAMETERS)

    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 FK "
            "demag references, generated at the frozen oracle commit via "
            "dev/bin/run-legacy-oracle. H_vertex in A/m, energy in J. Each "
            "case below is a schema-v1 fixture in its own right (oracle/"
            "mesh/physical_parameters/coordinates/quantities); "
            "'scalar_quantities' is a documented extension for whole-domain "
            "results (energy, average field) that are not per-coordinate "
            "nodal quantities and so fall outside 'quantities'."
        ),
        "cases": {
            "cube": _case_doc(
                "cube",
                "UnitCubeMesh(4,4,4), m=(1,0,0), Ms=8e5, standard "
                "(1e-6) legacy Krylov tolerances -- the default comparison "
                "datum for the DOLFINx port at its own default tolerances.",
                "dolfin.UnitCubeMesh(4, 4, 4)",
                {"nx": 4, "ny": 4, "nz": 4},
                8.0e5, 1e-9, STANDARD_PARAMETERS, cube_std,
                energy_tol=1e-5, avg_tol=1e-3,
                generator_argv=generator_argv,
            ),
            "cube_tight_tolerance": _case_doc(
                "cube_tight_tolerance",
                "Same mesh/physics as 'cube', legacy Krylov tolerances "
                "tightened to 1e-12 on both phi_1 and phi_2. Exists solely "
                "to isolate the standard-tolerance pointwise gap: compare "
                "against the DOLFINx port run with its own KSP rtol/atol "
                "tightened to 1e-12 (see Finding 1, Task 11b review round "
                "1). Empirically collapses the ~6.4e-6 pointwise gap to "
                "~1.4e-12 and the ~4.7e-8 energy gap to ~5.5e-14.",
                "dolfin.UnitCubeMesh(4, 4, 4)",
                {"nx": 4, "ny": 4, "nz": 4},
                8.0e5, 1e-9, TIGHT_PARAMETERS, cube_tight,
                energy_tol=1e-9, avg_tol=1e-8,
                generator_argv=generator_argv,
            ),
            "barmini": _case_doc(
                "barmini",
                "BoxMesh(Point(0,0,0),Point(3,3,10),2,2,4), m=(1,0,1)/sqrt2, "
                "Ms=0.86e6, standard (1e-6) legacy Krylov tolerances -- "
                "barmini-class witness geometry.",
                "dolfin.BoxMesh(Point(0,0,0), Point(3,3,10), 2, 2, 4)",
                {"x0": [0, 0, 0], "x1": [3, 3, 10], "nx": 2, "ny": 2, "nz": 4},
                0.86e6, 1e-9, STANDARD_PARAMETERS, barmini_std,
                energy_tol=1e-5, avg_tol=1e-3,
                generator_argv=generator_argv,
            ),
        },
    }

    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
