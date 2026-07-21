"""Generate the legacy native cubic-anisotropy field for a spatially VARYING K2 (Task 16).

Runs at the immutable FEniCS-2019 oracle. Companion to
``gen_cubic_anisotropy_native_oracle.py`` (constant K), this records the legacy
*default* ``assemble=False`` native field
(``finmag.native.llg.compute_cubic_field``, ``native/src/llg/energy.cc``) for a
spatially VARYING ``K2`` with ``K1 = K3 = 0``.

This is the evidence fixture for the native ``K2[2]`` index typo at
``energy.cc:116`` (``hz[i] += K2[2]*(...)`` -- a fixed node index ``2`` where
every other line uses the per-node ``K2[i]``). For spatially constant K2 the
nodal K2 array is uniform so the typo is dormant; for the varying K2 here it is
LIVE: every node's ``hz`` uses ``K2`` at native array index 2 (``v*``) instead
of its own ``K2[i]``, so the legacy native ``hz`` diverges from the correct
per-node field. ``hx``/``hy`` are unaffected (they use ``K2[i]``).

Records: the native (buggy) H_vertex; the mass-lumped per-node ``K2_nodal``
array legacy feeds the native routine (``assemble(K2_field * v * dx)/volumes``);
the correct per-node analytic field (independent NumPy derivation) so the fixture
documents predicted-vs-measured divergence; and the box-assembled energy (which
does NOT use the native field path -- verify: E is assembled from E_integrand
with the CG1 K2 field -- so it matches the correct varying-K2 energy).
[Claude Opus 4.8]
"""
import contextlib
import json
import os
import sys

import numpy as np
import dolfin as df


@contextlib.contextmanager
def _suppress_native_stdout():
    """Redirect OS-level fd 1 to /dev/null around energy assembly.

    FFC writes a C++-level quadrature warning to fd 1 (bypassing sys.stdout),
    which would corrupt the JSON emitted on stdout. See
    gen_cubic_anisotropy_native_oracle.py. [Claude Opus 4.8]
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

from finmag.field import Field
from finmag.energies.cubic_anisotropy import CubicAnisotropy
from finmag.util.consts import mu0

Ms = 876626
UNIT_LENGTH = 1e-9
BOX_EXTENT = 0.7
K2_EXPR = "-1.3e7*(1.0 + 0.4*x[0]/0.7)"
U1 = (0, -0.7071, 0.7071)
U2 = (0, 0.7071, 0.7071)
M_EXPR = ("0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])")


def _analytic_field(m_vecs, u1, u2, K1, K2_nodal, K3, Ms_value):
    """Correct per-node cubic field ``H = -1/(mu0 Ms) dE/dm`` with per-node K2."""
    u1 = np.asarray(u1, float)
    u2 = np.asarray(u2, float)
    u3 = np.cross(u1, u2)
    a = m_vecs @ u1
    b = m_vecs @ u2
    c = m_vecs @ u3
    K2 = K2_nodal
    g1 = 2 * K1 * a * (b**2 + c**2) + 2 * K2 * a * b**2 * c**2 \
        + 4 * K3 * a**3 * (b**4 + c**4)
    g2 = 2 * K1 * b * (a**2 + c**2) + 2 * K2 * b * a**2 * c**2 \
        + 4 * K3 * b**3 * (a**4 + c**4)
    g3 = 2 * K1 * c * (a**2 + b**2) + 2 * K2 * c * a**2 * b**2 \
        + 4 * K3 * c**3 * (a**4 + b**4)
    dEdm = g1[:, None] * u1 + g2[:, None] * u2 + g3[:, None] * u3
    return -(1.0 / (mu0 * Ms_value)) * dEdm


def main():
    generator_argv = [
        "dev/bin/run-legacy-oracle", "--", "pixi", "run", "--locked",
        "env", "PYTHONPATH=src", "python", os.path.abspath(__file__),
    ]

    mesh = df.BoxMesh(
        df.Point(0, 0, 0),
        df.Point(BOX_EXTENT, BOX_EXTENT, BOX_EXTENT), 4, 4, 4)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)

    m = Field(S3, df.Expression(M_EXPR, degree=1), name="m")
    Ms_field = Field(S1, Ms)

    ca = CubicAnisotropy(U1, U2, K1=0, K2=df.Expression(K2_EXPR, degree=1), K3=0)
    with _suppress_native_stdout():
        ca.setup(m, Ms_field, UNIT_LENGTH)
        E = ca.compute_energy()
        H_flat = np.array(ca.compute_field(), copy=True)  # native (buggy) field

    # Mass-lumped per-node K2 that legacy feeds the native routine.
    K2_field = Field(S1, df.Expression(K2_EXPR, degree=1))
    volumes = df.assemble(df.TestFunction(S1) * df.dx).get_local()
    K2_nodal = df.assemble(
        K2_field.f * df.TestFunction(S1) * df.dx).get_local() / volumes

    Hf = Field(S3)
    Hf.f.vector().set_local(H_flat)
    Hf.f.vector().apply("insert")
    H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(-1, 3)
    m_xyz = m.get_ordered_numpy_array_xyz().reshape(-1, 3)

    # order K2_nodal into the same xyz vertex order
    K2f = Field(S1)
    K2f.f.vector().set_local(K2_nodal)
    K2f.f.vector().apply("insert")
    K2_nodal_xyz = K2f.get_ordered_numpy_array()

    # correct field from the mass-lumped per-node K2 (independent derivation)
    H_correct = _analytic_field(m_xyz, U1, U2, 0.0, K2_nodal_xyz, 0.0, Ms)

    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)
    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))

    coords3 = coords3[order]
    H_xyz = H_xyz[order]
    m_xyz = m_xyz[order]
    K2_nodal_xyz = K2_nodal_xyz[order]
    H_correct = H_correct[order]

    scale = float(np.abs(H_xyz).max())
    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 cubic "
            "anisotropy NATIVE (assemble=False) field for a spatially VARYING "
            "K2 (K1=K3=0), generated at the frozen oracle commit via "
            "dev/bin/run-legacy-oracle. 'H_vertex' is the legacy native "
            "(BUGGY) field: every node's hz uses K2 at native array index 2 "
            "(the energy.cc:116 K2[2] typo, now LIVE for varying K2). "
            "'H_correct' is the correct per-node field from the same "
            "mass-lumped 'K2_nodal' array; hx/hy of the two agree, hz "
            "diverges. 'energy' is box-assembled (never uses the native field "
            "path) and equals the correct varying-K2 energy."
        ),
        "cases": {
            "k2_varying": {
                "oracle": {
                    "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
                    "command": generator_argv,
                    "generator": (
                        "src/finmag/tests/fixtures/"
                        "gen_cubic_k2_varying_oracle.py"),
                    "generator_note": (
                        "Legacy CubicAnisotropy(assemble=False) native field "
                        "for varying K2; K2_nodal is the mass-lumped per-node "
                        "K2 legacy feeds the native routine; H_correct is an "
                        "independent NumPy derivation with that per-node K2."),
                },
                "description": (
                    "Native (assemble=False) cubic field, K1=K3=0, "
                    "K2=-1.3e7*(1+0.4 x0/0.7) (spatially varying), "
                    "u1=(0,-0.7071,0.7071), u2=(0,0.7071,0.7071), "
                    "m=(0.6, 0.8 cos(2 pi x0), 0.8 sin(2 pi x0)), Ms=876626, "
                    "unit_length=1e-9, BoxMesh(0..0.7, 4,4,4)."),
                "mesh": {
                    "recipe": ("dolfin.BoxMesh(Point(0,0,0), "
                               "Point(0.7,0.7,0.7), 4, 4, 4)"),
                    "parameters": {"x0": [0, 0, 0], "x1": [0.7, 0.7, 0.7],
                                   "nx": 4, "ny": 4, "nz": 4},
                },
                "physical_parameters": {
                    "K1": {"value": 0, "unit": "J/m**3"},
                    "K2_expression": {"value": K2_EXPR, "unit": "J/m**3"},
                    "K3": {"value": 0, "unit": "J/m**3"},
                    "u1": {"value": list(U1), "unit": "1"},
                    "u2": {"value": list(U2), "unit": "1"},
                    "Ms": {"value": Ms, "unit": "A/m"},
                    "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
                    "assemble": {"value": False, "unit": "1"},
                    "m_expression": {"value": list(M_EXPR), "unit": "1"},
                },
                "coordinates": {"unit": "mesh_coordinate",
                                "ordering": "lexicographic_xyz",
                                "values": coords3.tolist()},
                "quantities": [
                    {"name": "m_vertex", "unit": "1", "value_shape": [3],
                     "values": m_xyz.tolist(),
                     "tolerances": {"absolute": 1e-12, "relative": 1e-12}},
                    {"name": "H_vertex", "unit": "A/m", "value_shape": [3],
                     "values": H_xyz.tolist(),
                     "tolerances": {"absolute": 1e-9 * scale,
                                    "relative": 1e-9}},
                    {"name": "K2_nodal", "unit": "J/m**3", "value_shape": [],
                     "values": K2_nodal_xyz.tolist(),
                     "tolerances": {"absolute": 1e-6, "relative": 1e-9}},
                    {"name": "H_correct", "unit": "A/m", "value_shape": [3],
                     "values": H_correct.tolist(),
                     "tolerances": {"absolute": 1e-9 * scale,
                                    "relative": 1e-9}},
                ],
                "scalar_quantities": [
                    {"name": "energy", "unit": "J", "value_shape": [],
                     "value": float(E),
                     "tolerances": {"absolute": 0.0, "relative": 1e-9}}],
            }
        },
    }
    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
