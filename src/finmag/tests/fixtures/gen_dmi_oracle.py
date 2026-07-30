"""Generate coordinate-ordered legacy DMI references (Task 13).

Runs at the immutable FEniCS-2019 oracle. Builds a single box mesh carrying a
nonuniform, analytically unit-norm magnetisation, runs the frozen legacy
``finmag.energies.dmi.DMI`` box-assemble solver for two ``dmi_type`` variants
(bulk ``'auto'`` and ``'interfacial'``), and dumps coordinate-sorted
per-vertex ``H`` together with the scalar energy to JSON on stdout.

The magnetisation ``m = (0.6, 0.8*cos(2*pi*x0), 0.8*sin(2*pi*x0))`` has exact
unit norm everywhere (0.6**2 + 0.8**2*(cos**2+sin**2) = 1) and depends only
on the mesh's ``x`` coordinate. The box's ``x``-extent is deliberately *not*
a whole multiple of the period (0.7, not 1.0): an interfacial-DMI density of
``D*(mx*dmz/dx)`` integrates ``cos(2*pi*x0)`` in ``x0``, which vanishes
exactly over a whole period and would make that case a degenerate all-zero
regression pin.

No iterative linear solve is involved (``box-assemble`` computes the field by
direct form assembly and a lumped nodal-volume division), so the only
disagreement expected between the legacy FEniCS-2019 assembly and the
DOLFINx port is floating-point summation-order noise, not solver residual.
"""
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag.field import Field
from finmag.energies.dmi import DMI

D = 5.0e-3
Ms = 8.0e5
UNIT_LENGTH = 1e-9
BOX_EXTENT = 0.7


def _run_case(mesh, dmi_type):
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)

    m_expr = df.Expression(
        ("0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])"), degree=1)
    m = Field(S3, m_expr, name="m")
    Ms_field = Field(DG, Ms)

    dmi = DMI(D, method="box-assemble", dmi_type=dmi_type)
    dmi.setup(m, Ms_field, UNIT_LENGTH)

    E = dmi.compute_energy()
    H_flat = dmi.compute_field()

    Hf = Field(S3)
    Hf.f.vector().set_local(H_flat)
    Hf.f.vector().apply("insert")
    H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(-1, 3)

    m_xyz = m.get_ordered_numpy_array_xyz().reshape(-1, 3)

    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)
    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))

    return {
        "energy": E,
        "coordinates": coords3[order],
        "H_vertex": H_xyz[order],
        "m_vertex": m_xyz[order],
    }


def _case_doc(name, description, dmi_type, mesh_recipe, mesh_parameters,
              result, energy_tol, h_atol, h_rtol, generator_argv):
    return {
        "oracle": {
            "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
            "command": generator_argv,
            "generator": "src/finmag/tests/fixtures/gen_dmi_oracle.py",
            "generator_note": (
                "Committed driver run at the oracle by absolute path via "
                "dev/bin/run-legacy-oracle (the script does not exist inside "
                "the detached oracle checkout, so it is supplied by absolute "
                "path). H_vertex/energy are computed by the frozen legacy "
                "DMI(method='box-assemble'); the driver only sorts and "
                "serialises coordinate-paired nodal values."
            ),
        },
        "description": description,
        "dmi_type": dmi_type,
        "mesh": {"recipe": mesh_recipe, "parameters": mesh_parameters},
        "physical_parameters": {
            "D": {"value": D, "unit": "J/m**2"},
            "Ms": {"value": Ms, "unit": "A/m"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "m_expression": {
                "value": ["0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])"],
                "unit": "1",
            },
        },
        "coordinates": {
            "unit": "mesh_coordinate",
            "ordering": "lexicographic_xyz",
            "values": result["coordinates"].tolist(),
        },
        "quantities": [
            {
                "name": "m_vertex",
                "unit": "1",
                "value_shape": [3],
                "values": result["m_vertex"].tolist(),
                "tolerances": {"absolute": 1e-12, "relative": 1e-12},
            },
            {
                "name": "H_vertex",
                "unit": "A/m",
                "value_shape": [3],
                "values": result["H_vertex"].tolist(),
                "tolerances": {"absolute": h_atol, "relative": h_rtol},
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

    mesh_bulk = df.BoxMesh(
        df.Point(0, 0, 0), df.Point(BOX_EXTENT, BOX_EXTENT, BOX_EXTENT), 4, 4, 4)
    bulk = _run_case(mesh_bulk, "auto")

    mesh_interfacial = df.BoxMesh(
        df.Point(0, 0, 0), df.Point(BOX_EXTENT, BOX_EXTENT, BOX_EXTENT), 4, 4, 4)
    interfacial = _run_case(mesh_interfacial, "interfacial")

    scale_bulk = float(np.abs(bulk["H_vertex"]).max())
    scale_interfacial = float(np.abs(interfacial["H_vertex"]).max())

    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 DMI "
            "references, generated at the frozen oracle commit via "
            "dev/bin/run-legacy-oracle. H_vertex in A/m, energy in J. Each "
            "case below is a schema-v1 fixture in its own right (oracle/"
            "mesh/physical_parameters/coordinates/quantities); "
            "'scalar_quantities' is a documented extension for the "
            "whole-domain energy result (not a per-coordinate nodal "
            "quantity, so it falls outside 'quantities'). No iterative "
            "linear solve is involved in box-assemble DMI, so tolerances "
            "reflect floating-point summation-order noise between the two "
            "FEM stacks, not solver residual."
        ),
        "cases": {
            "bulk_3d": _case_doc(
                "bulk_3d",
                "BoxMesh(Point(0,0,0),Point(0.7,0.7,0.7),4,4,4), "
                "dmi_type='auto' (bulk 3D, D*m.curl(m)), "
                "m=(0.6, 0.8*cos(2*pi*x0), 0.8*sin(2*pi*x0)) "
                "(exact unit norm), D=5e-3, Ms=8e5, unit_length=1e-9.",
                "auto",
                "dolfin.BoxMesh(Point(0, 0, 0), Point(0.7, 0.7, 0.7), 4, 4, 4)",
                {"x0": [0, 0, 0], "x1": [0.7, 0.7, 0.7],
                 "nx": 4, "ny": 4, "nz": 4},
                bulk,
                energy_tol=1e-9,
                h_atol=1e-9 * scale_bulk,
                h_rtol=1e-9,
                generator_argv=generator_argv,
            ),
            "interfacial": _case_doc(
                "interfacial",
                "BoxMesh(Point(0,0,0),Point(0.7,0.7,0.7),4,4,4), "
                "dmi_type='interfacial' (Rohart-Thiaville), "
                "m=(0.6, 0.8*cos(2*pi*x0), 0.8*sin(2*pi*x0)) "
                "(exact unit norm), D=5e-3, Ms=8e5, unit_length=1e-9.",
                "interfacial",
                "dolfin.BoxMesh(Point(0, 0, 0), Point(0.7, 0.7, 0.7), 4, 4, 4)",
                {"x0": [0, 0, 0], "x1": [0.7, 0.7, 0.7],
                 "nx": 4, "ny": 4, "nz": 4},
                interfacial,
                energy_tol=1e-9,
                h_atol=1e-9 * scale_interfacial,
                h_rtol=1e-9,
                generator_argv=generator_argv,
            ),
        },
    }

    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
