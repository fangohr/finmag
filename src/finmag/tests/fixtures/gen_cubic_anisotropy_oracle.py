"""Generate a coordinate-ordered legacy cubic-anisotropy reference (Task 14).

Runs at the immutable FEniCS-2019 oracle. Builds a single box mesh carrying a
nonuniform, analytically unit-norm magnetisation, runs the frozen legacy
``finmag.energies.cubic_anisotropy.CubicAnisotropy`` with the exact
constants/axes used by the legacy module's own
``cubic_anisotropy_test.py::test_cubic_anisotropy_energy`` (K1, K2, K3 all
nonzero; ``u1``/``u2`` neither unit nor exactly orthogonal, since
``0.7071 != 1/sqrt(2)`` to full precision), and dumps coordinate-sorted
per-vertex ``H`` together with the scalar energy to JSON on stdout.

The legacy default ``assemble=False`` uses a separate native/compiled direct
field computation that is not part of the DOLFINx port (see
``src/finmag/energies/cubic_anisotropy.py``'s docstring); this generator
therefore constructs the legacy class with ``assemble=True`` so the recorded
``H_vertex`` is the box-assemble weak-form-derivative field the DOLFINx port
actually reproduces, not the (unported) native one. The *energy* is identical
either way, since the legacy class always box-assembles the energy
regardless of ``assemble``.

The magnetisation ``m = (0.6, 0.8*cos(2*pi*x0), 0.8*sin(2*pi*x0))`` has exact
unit norm everywhere (0.6**2 + 0.8**2*(cos**2+sin**2) = 1) and depends only
on the mesh's ``x`` coordinate, matching the Task 13 DMI oracle's
magnetisation for consistency across fixtures.
"""
import contextlib
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag.field import Field
from finmag.energies.cubic_anisotropy import CubicAnisotropy


@contextlib.contextmanager
def _suppress_native_stdout():
    """Silence FFC's C++-level "number of integration points" warning.

    The K3 term is degree 8 in the (non-polynomial, cos/sin) magnetisation
    expression, so FFC's automatic quadrature-degree estimate is large; its
    warning is written directly to the OS-level file descriptor 1 (bypassing
    Python's ``sys.stdout``/logging), which would otherwise corrupt the JSON
    this script prints to stdout. Redirecting fd 1 to ``/dev/null`` for the
    assembly calls only (restored immediately afterwards, before the JSON
    dump) leaves the physics untouched. [Claude Sonnet 5]
    """
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, stdout_fd)
        os.close(devnull_fd)
        os.close(saved_fd)

Ms = 876626  # A/m, matching cubic_anisotropy_test.py
UNIT_LENGTH = 1e-9
BOX_EXTENT = 0.7

K1 = -8608726
K2 = -13744132
K3 = 1100269
U1 = (0, -0.7071, 0.7071)
U2 = (0, 0.7071, 0.7071)


def _run_case(mesh):
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)

    m_expr = df.Expression(
        ("0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])"), degree=1)
    m = Field(S3, m_expr, name="m")
    Ms_field = Field(DG, Ms)

    ca = CubicAnisotropy(U1, U2, K1, K2, K3, assemble=True)
    with _suppress_native_stdout():
        ca.setup(m, Ms_field, UNIT_LENGTH)
        E = ca.compute_energy()
        H_flat = ca.compute_field()

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

    mesh_3d = df.BoxMesh(
        df.Point(0, 0, 0), df.Point(BOX_EXTENT, BOX_EXTENT, BOX_EXTENT), 4, 4, 4)
    bulk = _run_case(mesh_3d)

    scale = float(np.abs(bulk["H_vertex"]).max())

    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 "
            "cubic-anisotropy reference, generated at the frozen oracle "
            "commit via dev/bin/run-legacy-oracle. H_vertex in A/m, energy "
            "in J. K1/K2/K3/u1/u2 are the exact constants/axes used by the "
            "legacy module's own cubic_anisotropy_test.py. The legacy class "
            "was constructed with assemble=True so H_vertex is the "
            "box-assemble field (the only field-computation path the "
            "DOLFINx port implements); the energy is identical to the "
            "legacy default (assemble=False) since energy is always "
            "box-assembled regardless of that flag. No iterative linear "
            "solve is involved (box-assemble computes the field by direct "
            "form assembly and a lumped nodal-volume division), so the only "
            "disagreement expected between the legacy FEniCS-2019 assembly "
            "and the DOLFINx port is floating-point summation-order noise, "
            "not solver residual."
        ),
        "cases": {
            "cubic_3d": {
                "oracle": {
                    "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
                    "command": generator_argv,
                    "generator": (
                        "src/finmag/tests/fixtures/"
                        "gen_cubic_anisotropy_oracle.py"
                    ),
                    "generator_note": (
                        "Committed driver run at the oracle by absolute "
                        "path via dev/bin/run-legacy-oracle (the script "
                        "does not exist inside the detached oracle "
                        "checkout, so it is supplied by absolute path). "
                        "H_vertex/energy are computed by the frozen legacy "
                        "CubicAnisotropy(assemble=True); the driver only "
                        "sorts and serialises coordinate-paired nodal "
                        "values."
                    ),
                },
                "description": (
                    "BoxMesh(Point(0,0,0),Point(0.7,0.7,0.7),4,4,4), "
                    "K1=-8608726, K2=-13744132, K3=1100269 (all nonzero), "
                    "u1=(0,-0.7071,0.7071), u2=(0,0.7071,0.7071) (neither "
                    "unit nor exactly orthogonal), "
                    "m=(0.6, 0.8*cos(2*pi*x0), 0.8*sin(2*pi*x0)) "
                    "(exact unit norm), Ms=876626, unit_length=1e-9, "
                    "assemble=True."
                ),
                "mesh": {
                    "recipe": (
                        "dolfin.BoxMesh(Point(0, 0, 0), "
                        "Point(0.7, 0.7, 0.7), 4, 4, 4)"
                    ),
                    "parameters": {
                        "x0": [0, 0, 0], "x1": [0.7, 0.7, 0.7],
                        "nx": 4, "ny": 4, "nz": 4,
                    },
                },
                "physical_parameters": {
                    "K1": {"value": K1, "unit": "J/m**3"},
                    "K2": {"value": K2, "unit": "J/m**3"},
                    "K3": {"value": K3, "unit": "J/m**3"},
                    "u1": {"value": list(U1), "unit": "1"},
                    "u2": {"value": list(U2), "unit": "1"},
                    "Ms": {"value": Ms, "unit": "A/m"},
                    "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
                    "m_expression": {
                        "value": [
                            "0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])"
                        ],
                        "unit": "1",
                    },
                },
                "coordinates": {
                    "unit": "mesh_coordinate",
                    "ordering": "lexicographic_xyz",
                    "values": bulk["coordinates"].tolist(),
                },
                "quantities": [
                    {
                        "name": "m_vertex",
                        "unit": "1",
                        "value_shape": [3],
                        "values": bulk["m_vertex"].tolist(),
                        "tolerances": {"absolute": 1e-12, "relative": 1e-12},
                    },
                    {
                        "name": "H_vertex",
                        "unit": "A/m",
                        "value_shape": [3],
                        "values": bulk["H_vertex"].tolist(),
                        "tolerances": {
                            "absolute": 1e-9 * scale, "relative": 1e-9,
                        },
                    },
                ],
                "scalar_quantities": [
                    {
                        "name": "energy",
                        "unit": "J",
                        "value_shape": [],
                        "value": bulk["energy"],
                        "tolerances": {"absolute": 0.0, "relative": 1e-9},
                    },
                ],
            },
        },
    }

    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
