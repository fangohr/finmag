"""Generate coordinate-ordered legacy ThinFilmDemag references (Task 19).

Runs at the immutable FEniCS-2019 oracle. Builds a single box mesh carrying a
nonuniform, analytically unit-norm magnetisation, runs the frozen legacy
``finmag.energies.thin_film_demag.ThinFilmDemag`` (``Hi = -strength_i * m_i``)
for two branches, and dumps coordinate-sorted per-vertex ``H`` to JSON on
stdout.

Two cases exercise the two ``field_strength`` branches (the analytic legacy
test module ``thin_film_demag_test.py`` only ever pins the constant-Ms,
``field_strength=None`` case with a *uniform* Ms, which is already fully
covered by a closed-form analytic test in the DOLFINx port and would not
justify a fixture on its own -- see "When Not to Create a Fixture" in
``docs/superpowers/specs/legacy-oracle-fixtures.md``):

- ``varying_ms_default_strength``: ``field_strength=None``, direction ``"z"``,
  a spatially VARYING DG0 ``Ms``. This exercises the box-assemble nodal
  average (``df.assemble(Ms.f * TestFunction(S1) * dx) / volumes``) that
  legacy computes internally to turn a non-uniform ``Ms`` into a per-node
  ``strength`` -- material legacy behavior with no practical closed-form
  result (the box average is mesh- and quadrature-dependent), so this is
  exactly the "assembled, non-uniform field" case the fixture guidance calls
  for.
- ``explicit_strength_x_direction``: a user-supplied scalar
  ``field_strength`` (bypassing the Ms-average branch entirely) with
  direction ``"x"``, confirming the direction-index dispatch (``ord(direction)
  - 120``) and the "strength multiplies m verbatim, no Ms lookup" branch.

The magnetisation ``m = (0.6, 0.8*cos(2*pi*x0), 0.8*sin(2*pi*x0))`` has exact
unit norm everywhere (0.6**2 + 0.8**2*(cos**2+sin**2) = 1) and depends only on
the mesh's ``x`` coordinate -- the same expression used by
``gen_dmi_oracle.py``/``gen_cubic_varying_ms_oracle.py``. No iterative linear
solve is involved (box assembly + explicit division), so the only expected
port/oracle disagreement is floating-point summation-order noise. [Claude
Sonnet 5]
"""
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag.field import Field
from finmag.energies.thin_film_demag import ThinFilmDemag

UNIT_LENGTH = 1e-9
BOX_EXTENT = 0.7
MS_EXPR = "800000.0*(1.0 + 0.3*x[0]/0.7)"
M_EXPR = ("0.6", "0.8*cos(2*pi*x[0])", "0.8*sin(2*pi*x[0])")
EXPLICIT_STRENGTH = 5.0e5


def _mesh():
    return df.BoxMesh(
        df.Point(0, 0, 0), df.Point(BOX_EXTENT, BOX_EXTENT, BOX_EXTENT), 3, 3, 3)


def _coords_sorted(mesh):
    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)
    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))
    return coords3, order


def _run_case(direction, field_strength):
    mesh = _mesh()
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)

    m = Field(S3, df.Expression(M_EXPR, degree=1), name="m")
    ms_field = Field(DG, df.Expression(MS_EXPR, degree=1), name="Ms")

    demag = ThinFilmDemag(direction=direction, field_strength=field_strength)
    demag.setup(m, ms_field, UNIT_LENGTH)
    H_flat = np.array(demag.compute_field(), copy=True)
    E = demag.compute_energy()

    Hf = Field(S3)
    Hf.f.vector().set_local(H_flat)
    Hf.f.vector().apply("insert")
    H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(-1, 3)
    m_xyz = m.get_ordered_numpy_array_xyz().reshape(-1, 3)

    coords3, order = _coords_sorted(mesh)
    return {
        "coordinates": coords3[order],
        "H_vertex": H_xyz[order],
        "m_vertex": m_xyz[order],
        "energy": E,
    }


def main():
    generator_argv = [
        "dev/bin/run-legacy-oracle", "--", "pixi", "run", "--locked",
        "env", "PYTHONPATH=src", "python", os.path.abspath(__file__),
    ]

    varying = _run_case("z", None)
    explicit = _run_case("x", EXPLICIT_STRENGTH)

    def _case_doc(name, description, direction, field_strength, result):
        H = result["H_vertex"]
        scale = float(np.abs(H).max())
        return {
            "oracle": {
                "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
                "command": generator_argv,
                "generator": (
                    "src/finmag/tests/fixtures/"
                    "gen_thin_film_demag_oracle.py"),
                "generator_note": (
                    "Committed driver run at the oracle by absolute path via "
                    "dev/bin/run-legacy-oracle (the script does not exist "
                    "inside the detached oracle checkout, so it is supplied "
                    "by absolute path). H_vertex/energy are computed by the "
                    "frozen legacy ThinFilmDemag; the driver only sorts and "
                    "serialises coordinate-paired nodal values."),
            },
            "description": description,
            "mesh": {
                "recipe": ("dolfin.BoxMesh(Point(0,0,0), "
                           "Point({0},{0},{0}), 3, 3, 3)".format(BOX_EXTENT)),
                "parameters": {"x0": [0, 0, 0],
                               "x1": [BOX_EXTENT, BOX_EXTENT, BOX_EXTENT],
                               "nx": 3, "ny": 3, "nz": 3},
            },
            "physical_parameters": {
                "direction": {"value": direction, "unit": "1"},
                "field_strength": {
                    "value": field_strength, "unit": "A/m"},
                "Ms_expression": {"value": MS_EXPR, "unit": "A/m"},
                "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
                "m_expression": {"value": list(M_EXPR), "unit": "1"},
            },
            "coordinates": {
                "unit": "mesh_coordinate",
                "ordering": "lexicographic_xyz",
                "values": result["coordinates"].tolist(),
            },
            "quantities": [
                {"name": "m_vertex", "unit": "1", "value_shape": [3],
                 "values": result["m_vertex"].tolist(),
                 "tolerances": {"absolute": 1e-12, "relative": 1e-12}},
                {"name": "H_vertex", "unit": "A/m", "value_shape": [3],
                 "values": H.tolist(),
                 "tolerances": {"absolute": 1e-9 * scale, "relative": 1e-9}},
            ],
            "scalar_quantities": [
                {"name": "energy", "unit": "J", "value_shape": [],
                 "value": float(result["energy"]),
                 "tolerances": {"absolute": 0.0, "relative": 0.0}},
            ],
        }

    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 "
            "ThinFilmDemag references, generated at the frozen oracle "
            "commit via dev/bin/run-legacy-oracle. H_vertex in A/m. Each "
            "case below is a schema-v1 fixture in its own right (oracle/"
            "mesh/physical_parameters/coordinates/quantities); "
            "'scalar_quantities' is a documented extension for the "
            "whole-domain 'energy' result (legacy ThinFilmDemag.compute_energy "
            "is a hard-coded literal 0, pinned here too)."
        ),
        "cases": {
            "varying_ms_default_strength": _case_doc(
                "varying_ms_default_strength",
                "direction='z', field_strength=None (Ms box-average), "
                "Ms=800000*(1+0.3 x0/0.7) (spatially varying DG0), "
                "m=(0.6, 0.8 cos(2 pi x0), 0.8 sin(2 pi x0)), "
                "unit_length=1e-9, BoxMesh(0..0.7, 3,3,3).",
                "z", None, varying,
            ),
            "explicit_strength_x_direction": _case_doc(
                "explicit_strength_x_direction",
                "direction='x', field_strength={} (explicit scalar, "
                "bypasses the Ms box-average branch entirely), "
                "same Ms/m/mesh as 'varying_ms_default_strength'.".format(
                    EXPLICIT_STRENGTH),
                "x", EXPLICIT_STRENGTH, explicit,
            ),
        },
    }

    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
