"""Generate the legacy *native* (assemble=False) cubic-anisotropy field oracle.

Task 14, fix round 1. Companion to ``gen_cubic_anisotropy_oracle.py`` (which
records the ``assemble=True`` box-assemble field). This generator records the
legacy *default* ``assemble=False`` effective field, i.e. the per-node analytic
field produced by the native routine
``finmag.native.llg.compute_cubic_field`` (``native/src/llg/energy.cc``), which
the DOLFINx port reproduces in NumPy for the ``assemble=False`` path.

Four sub-cases isolate each anisotropy term (``K1``-only, ``K2``-only,
``K3``-only) plus the all-nonzero case, on the same mesh / magnetisation /
axes / ``Ms`` family as the existing ``cubic_anisotropy_oracle`` fixture. The
``K2``-only case is the evidence for the native ``K2[2]`` index typo at
``native/src/llg/energy.cc:116``: because ``K2`` is spatially constant the
nodal ``K2`` array is uniform, so ``K2[2] == K2[i]`` and the typo is
numerically dormant -- this fixture pins that the native field then equals the
correct analytic derivation. [Claude Opus 4.8]
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

    The K3 term is degree 8 in the (non-polynomial cos/sin) magnetisation, so
    FFC's automatic quadrature-degree estimate is large and its warning is
    written to OS-level fd 1 (bypassing ``sys.stdout``), which would corrupt
    the JSON on stdout. Redirect fd 1 to ``/dev/null`` around the energy
    assembly only. The native field path itself does no assembly. [Claude Opus 4.8]
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


Ms = 876626  # A/m, matching cubic_anisotropy_test.py / the box-field oracle
UNIT_LENGTH = 1e-9
BOX_EXTENT = 0.7

K1_FULL = -8608726
K2_FULL = -13744132
K3_FULL = 1100269
U1 = (0, -0.7071, 0.7071)
U2 = (0, 0.7071, 0.7071)

CASES = {
    "k1_only": (K1_FULL, 0, 0),
    "k2_only": (0, K2_FULL, 0),
    "k3_only": (0, 0, K3_FULL),
    "all_nonzero": (K1_FULL, K2_FULL, K3_FULL),
}


def _run_case(mesh, K1, K2, K3):
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    # The native assemble=False field routine expects a per-vertex (CG1) Ms
    # array (shape == number of nodes), not a per-cell DG0 one; Ms is constant
    # so the value is identical either way (and the box-assembled energy is
    # unaffected).
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)

    m_expr = df.Expression(
        ("0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])"), degree=1)
    m = Field(S3, m_expr, name="m")
    Ms_field = Field(S1, Ms)

    # Legacy default assemble=False -> native/direct field computation.
    ca = CubicAnisotropy(U1, U2, K1, K2, K3)
    with _suppress_native_stdout():
        ca.setup(m, Ms_field, UNIT_LENGTH)
        E = ca.compute_energy()
        H_flat = np.array(ca.compute_field(), copy=True)

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

    cases = {}
    for name, (K1, K2, K3) in CASES.items():
        res = _run_case(mesh_3d, K1, K2, K3)
        scale = float(np.abs(res["H_vertex"]).max())
        cases[name] = {
            "oracle": {
                "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
                "command": generator_argv,
                "generator": (
                    "src/finmag/tests/fixtures/"
                    "gen_cubic_anisotropy_native_oracle.py"
                ),
                "generator_note": (
                    "Legacy CubicAnisotropy constructed with the default "
                    "assemble=False, so compute_field() returns the native "
                    "finmag.native.llg.compute_cubic_field analytic field "
                    "(native/src/llg/energy.cc). The driver only sorts and "
                    "serialises coordinate-paired nodal values."
                ),
            },
            "description": (
                "Native (assemble=False) cubic-anisotropy field. "
                "K1={}, K2={}, K3={}. "
                "BoxMesh(Point(0,0,0),Point(0.7,0.7,0.7),4,4,4), "
                "u1=(0,-0.7071,0.7071), u2=(0,0.7071,0.7071), "
                "m=(0.6, 0.8*cos(2*pi*x0), 0.8*sin(2*pi*x0)), "
                "Ms=876626, unit_length=1e-9.".format(K1, K2, K3)
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
                "assemble": {"value": False, "unit": "1"},
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
                "values": res["coordinates"].tolist(),
            },
            "quantities": [
                {
                    "name": "m_vertex",
                    "unit": "1",
                    "value_shape": [3],
                    "values": res["m_vertex"].tolist(),
                    "tolerances": {"absolute": 1e-12, "relative": 1e-12},
                },
                {
                    "name": "H_vertex",
                    "unit": "A/m",
                    "value_shape": [3],
                    "values": res["H_vertex"].tolist(),
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
                    "value": res["energy"],
                    "tolerances": {"absolute": 0.0, "relative": 1e-9},
                },
            ],
        }

    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 "
            "cubic-anisotropy NATIVE (assemble=False) field reference, "
            "generated at the frozen oracle commit via "
            "dev/bin/run-legacy-oracle. H_vertex in A/m (the per-node "
            "analytic field from native/src/llg/energy.cc "
            "compute_cubic_field), energy in J. Four sub-cases isolate the "
            "K1/K2/K3 terms plus the all-nonzero case. The K2-only case "
            "documents the native K2[2] index typo at energy.cc:116, which is "
            "numerically dormant for the spatially constant K2 supported here "
            "(nodal K2 array uniform -> K2[2] == K2[i]), so the native field "
            "equals the correct analytic derivation to floating-point noise."
        ),
        "cases": cases,
    }

    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
