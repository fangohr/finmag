"""Generate the legacy native cubic-anisotropy field for a spatially VARYING
Ms (fix round 1, Finding 1).

Runs at the immutable FEniCS-2019 oracle. Companion to
``gen_cubic_anisotropy_native_oracle.py`` (constant Ms) and
``gen_cubic_k2_varying_oracle.py`` (varying K2), this records the legacy
*default* ``assemble=False`` native field
(``finmag.native.llg.compute_cubic_field``, ``native/src/llg/energy.cc``) for
a spatially VARYING ``Ms`` with ``K1``/``K3`` nonzero and ``K2 = 0`` (chosen
to sidestep the documented ``energy.cc:116`` ``K2[2]`` index typo entirely, so
this fixture isolates ``Ms`` handling exactly as the Finding-1 review comment
required).

Legacy's ``assemble=False`` path feeds the compiled routine ``self.Ms =
self.Ms.get_numpy_array_debug()`` -- the *raw* (not mass-lumped) nodal array
of whatever ``Ms`` ``Field`` was passed to ``setup``. The native routine
hard-requires that array to have exactly as many entries as ``m`` has nodes
(``Ms_arr.check_shape(nodes, ...)``), so -- exactly like the other
``gen_cubic_*_native_oracle.py`` generators -- ``Ms`` is placed on the CG1
scalar space matching ``m``, not DG0 (a DG0 Ms, e.g. from
``finmag.sim.sim.Simulation``, does not satisfy this size requirement and
legacy raises a ``ValueError`` there; see
``cubic_anisotropy.py::CubicAnisotropy._ms_per_node`` for the port's
documented reproduction of that limitation).
[Claude Sonnet 5]
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

UNIT_LENGTH = 1e-9
BOX_EXTENT = 0.7
MS_EXPR = "876626.0*(1.0 + 0.35*x[0]/0.7)"
K1 = -8608726
K2 = 0  # deliberately zero: isolates Ms handling from the K2[2] typo
K3 = 1100269
U1 = (0, -0.7071, 0.7071)
U2 = (0, 0.7071, 0.7071)
M_EXPR = ("0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])")


def _analytic_field(m_vecs, u1, u2, K1, K2, K3, Ms_nodal):
    """Correct per-node cubic field ``H = -1/(mu0 Ms) dE/dm`` with per-node Ms."""
    u1 = np.asarray(u1, float)
    u2 = np.asarray(u2, float)
    u3 = np.cross(u1, u2)
    a = m_vecs @ u1
    b = m_vecs @ u2
    c = m_vecs @ u3
    g1 = 2 * K1 * a * (b**2 + c**2) + 2 * K2 * a * b**2 * c**2 \
        + 4 * K3 * a**3 * (b**4 + c**4)
    g2 = 2 * K1 * b * (a**2 + c**2) + 2 * K2 * b * a**2 * c**2 \
        + 4 * K3 * b**3 * (a**4 + c**4)
    g3 = 2 * K1 * c * (a**2 + b**2) + 2 * K2 * c * a**2 * b**2 \
        + 4 * K3 * c**3 * (a**4 + b**4)
    dEdm = g1[:, None] * u1 + g2[:, None] * u2 + g3[:, None] * u3
    return -(1.0 / (mu0 * Ms_nodal[:, None])) * dEdm


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
    Ms_field = Field(S1, df.Expression(MS_EXPR, degree=1), name="Ms")

    ca = CubicAnisotropy(U1, U2, K1=K1, K2=K2, K3=K3)  # assemble=False default
    with _suppress_native_stdout():
        ca.setup(m, Ms_field, UNIT_LENGTH)
        E = ca.compute_energy()
        H_flat = np.array(ca.compute_field(), copy=True)  # native field

    Hf = Field(S3)
    Hf.f.vector().set_local(H_flat)
    Hf.f.vector().apply("insert")
    H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(-1, 3)
    m_xyz = m.get_ordered_numpy_array_xyz().reshape(-1, 3)
    Ms_xyz = Ms_field.get_ordered_numpy_array()

    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)
    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))

    coords3 = coords3[order]
    H_xyz = H_xyz[order]
    m_xyz = m_xyz[order]
    Ms_xyz = Ms_xyz[order]

    # Independent NumPy derivation with the same per-node Ms, as a
    # self-consistency check that the fixture's H_vertex is genuinely the
    # varying-Ms analytic field (no typo term is involved for K2=0).
    H_correct = _analytic_field(m_xyz, U1, U2, K1, K2, K3, Ms_xyz)
    np.testing.assert_allclose(H_xyz, H_correct, rtol=1e-9, atol=1e-9)

    scale = float(np.abs(H_xyz).max())
    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 "
            "cubic-anisotropy NATIVE (assemble=False) field for a spatially "
            "VARYING Ms (K1, K3 nonzero, K2=0 to sidestep the documented "
            "energy.cc:116 K2[2] typo), generated at the frozen oracle "
            "commit via dev/bin/run-legacy-oracle. Ms is placed on the CG1 "
            "scalar space matching m -- the space legacy's native routine "
            "requires (Ms_arr.check_shape(nodes, ...)) -- consistent with "
            "cubic_anisotropy_test.py and the other native-oracle "
            "generators. H_vertex is the legacy native field; it matches an "
            "independent NumPy analytic derivation with the same per-node "
            "Ms to floating-point noise (verified in this generator before "
            "serialisation)."
        ),
        "cases": {
            "ms_varying": {
                "oracle": {
                    "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
                    "command": generator_argv,
                    "generator": (
                        "src/finmag/tests/fixtures/"
                        "gen_cubic_varying_ms_oracle.py"),
                    "generator_note": (
                        "Legacy CubicAnisotropy(assemble=False) native field "
                        "for varying Ms with K2=0; Ms is a CG1 Field on m's "
                        "own function space, matching legacy's own working "
                        "usage of the native path."),
                },
                "description": (
                    "Native (assemble=False) cubic field, K1={}, K2=0, "
                    "K3={}, Ms=876626*(1+0.35 x0/0.7) (spatially varying), "
                    "u1=(0,-0.7071,0.7071), u2=(0,0.7071,0.7071), "
                    "m=(0.6, 0.8 cos(2 pi x0), 0.8 sin(2 pi x0)), "
                    "unit_length=1e-9, BoxMesh(0..0.7, 4,4,4).".format(K1, K3)
                ),
                "mesh": {
                    "recipe": ("dolfin.BoxMesh(Point(0,0,0), "
                               "Point(0.7,0.7,0.7), 4, 4, 4)"),
                    "parameters": {"x0": [0, 0, 0], "x1": [0.7, 0.7, 0.7],
                                   "nx": 4, "ny": 4, "nz": 4},
                },
                "physical_parameters": {
                    "K1": {"value": K1, "unit": "J/m**3"},
                    "K2": {"value": K2, "unit": "J/m**3"},
                    "K3": {"value": K3, "unit": "J/m**3"},
                    "u1": {"value": list(U1), "unit": "1"},
                    "u2": {"value": list(U2), "unit": "1"},
                    "Ms_expression": {"value": MS_EXPR, "unit": "A/m"},
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
                    {"name": "Ms_vertex", "unit": "A/m", "value_shape": [],
                     "values": Ms_xyz.tolist(),
                     "tolerances": {"absolute": 1e-6, "relative": 1e-9}},
                    {"name": "H_vertex", "unit": "A/m", "value_shape": [3],
                     "values": H_xyz.tolist(),
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
