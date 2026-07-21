"""Generate coordinate-ordered legacy time-Zeeman references (Task 15).

Runs at the immutable FEniCS-2019 oracle. Builds a single box mesh carrying a
constant unit magnetisation and exercises the frozen legacy
``finmag.energies.zeeman.TimeZeeman``/``DiscreteTimeZeeman`` classes through a
*sequence* of ``update(t)`` calls (order matters -- both classes are
stateful), dumping coordinate-sorted ``H`` and the scalar energy at each step
to JSON on stdout.

Two cases:

- ``time_zeeman``: a genuinely space-*and*-time varying field,
  ``H = (0, A*x0*cos(w*t), 0)``, exercising ``TimeZeeman``'s "update as
  continuously as possible" contract with a legacy ``dolfin.Expression``.
  This is the oracle-side counterpart of the ported DOLFINx contract's
  ``field_function(t) -> callable(x)`` nested-callable case (see
  ``_as_time_field_function`` in ``finmag/energies/zeeman.py``).
- ``discrete_time_zeeman``: a spatially uniform field ``H = (0, t, 0)`` with
  ``dt_update=2e-10``, sampled at a *sequence* of ``t`` values that cross the
  update threshold more than once. This is deliberately designed to pin the
  preserved legacy quirk documented on the ported ``DiscreteTimeZeeman``:
  ``update()`` never advances ``t_last_update``, so once ``t`` first reaches
  ``dt_update`` every subsequent ``update(t)`` call refreshes the field again
  (rather than only every ``dt_update`` thereafter).

No iterative linear solve is involved (Zeeman field/energy is a direct
evaluation), so the only disagreement expected between the legacy FEniCS-2019
assembly and the DOLFINx port is floating-point summation-order noise.
"""
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag.field import Field
from finmag.energies.zeeman import TimeZeeman, DiscreteTimeZeeman

Ms = 8.0e5
UNIT_LENGTH = 1e-9
BOX_EXTENT = 1.0
NX = 3

A_AMPLITUDE = 2.0e5
W_ANGULAR = 2.0 * np.pi * 1.0e9  # rad/s


def _mesh():
    return df.BoxMesh(
        df.Point(0, 0, 0), df.Point(BOX_EXTENT, BOX_EXTENT, BOX_EXTENT),
        NX, NX, NX)


def _coords_sorted(mesh):
    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)
    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))
    return coords3, order


def _snapshot(H_ext, m, order):
    E = H_ext.compute_energy()
    H_flat = H_ext.compute_field()
    Hf = Field(H_ext.m.functionspace)
    Hf.f.vector().set_local(H_flat)
    Hf.f.vector().apply("insert")
    H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(-1, 3)[order]
    return {"energy": E, "H_vertex": H_xyz.tolist()}


def _run_time_zeeman():
    mesh = _mesh()
    coords3, order = _coords_sorted(mesh)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)
    m = Field(S3, df.Constant((0.6, 0.8, 0.0)), name="m")
    Ms_field = Field(DG, Ms)

    field_expr = df.Expression(
        ("0.0", "A*x[0]*cos(w*t)", "0.0"), A=A_AMPLITUDE, w=W_ANGULAR,
        t=0.0, degree=1)
    H_ext = TimeZeeman(field_expr)
    H_ext.setup(m, Ms_field, unit_length=UNIT_LENGTH)

    # Deliberately not symmetric around half a period (cos(2*pi*f-x) ==
    # cos(x) would make some sample pairs coincide and weaken the pin).
    t_values = [0.0, 0.2e-9, 0.55e-9, 0.9e-9]
    snapshots = []
    for t in t_values:
        if t != 0.0:
            H_ext.update(t)
        snapshots.append(_snapshot(H_ext, m, order))

    return coords3[order], t_values, snapshots


def _run_discrete_time_zeeman():
    mesh = _mesh()
    coords3, order = _coords_sorted(mesh)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)
    m = Field(S3, df.Constant((0.6, 0.8, 0.0)), name="m")
    Ms_field = Field(DG, Ms)

    field_expr = df.Expression(("0.0", "t", "0.0"), t=0.0, degree=1)
    H_ext = DiscreteTimeZeeman(field_expr, dt_update=2e-10)
    H_ext.setup(m, Ms_field, unit_length=UNIT_LENGTH)

    t_values = [0.0, 1e-10, 2e-10, 2.5e-10, 4e-10, 4.1e-10, 4.5e-10]
    snapshots = []
    for t in t_values:
        if t != 0.0:
            H_ext.update(t)
        snapshots.append(_snapshot(H_ext, m, order))

    return coords3[order], t_values, snapshots


def _case_doc(name, description, coords, t_values, snapshots,
              generator_argv, m_value):
    scale = max(
        1e-30,
        float(np.max(np.abs(np.asarray(
            [s["H_vertex"] for s in snapshots])))))
    return {
        "oracle": {
            "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
            "command": generator_argv,
            "generator": "src/finmag/tests/fixtures/gen_timezeeman_oracle.py",
            "generator_note": (
                "Committed driver run at the oracle by absolute path via "
                "dev/bin/run-legacy-oracle. H_vertex/energy at each step are "
                "computed by the frozen legacy TimeZeeman/DiscreteTimeZeeman "
                "after a *sequence* of stateful update(t) calls in the order "
                "given by 't_values' (order matters); the driver only sorts "
                "and serialises coordinate-paired nodal values."
            ),
        },
        "description": description,
        "mesh": {
            "recipe": "dolfin.BoxMesh(Point(0,0,0),Point(1,1,1),3,3,3)",
            "parameters": {"x0": [0, 0, 0], "x1": [1, 1, 1],
                            "nx": NX, "ny": NX, "nz": NX},
        },
        "physical_parameters": {
            "Ms": {"value": Ms, "unit": "A/m"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "m_constant": {"value": m_value, "unit": "1"},
        },
        "coordinates": {
            "unit": "mesh_coordinate",
            "ordering": "lexicographic_xyz",
            "values": coords.tolist(),
        },
        "t_values": t_values,
        "snapshots": [
            {
                "t": t,
                "quantities": [
                    {
                        "name": "H_vertex",
                        "unit": "A/m",
                        "value_shape": [3],
                        "values": snap["H_vertex"],
                        "tolerances": {
                            "absolute": 1e-9 * scale, "relative": 1e-6},
                    },
                ],
                "scalar_quantities": [
                    {
                        "name": "energy",
                        "unit": "J",
                        "value_shape": [],
                        "value": snap["energy"],
                        "tolerances": {"absolute": 1e-18, "relative": 1e-6},
                    },
                ],
            }
            for t, snap in zip(t_values, snapshots)
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

    tz_coords, tz_t, tz_snapshots = _run_time_zeeman()
    dtz_coords, dtz_t, dtz_snapshots = _run_discrete_time_zeeman()

    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 "
            "time-Zeeman references, generated at the frozen oracle commit "
            "via dev/bin/run-legacy-oracle. H_vertex in A/m, energy in J. "
            "Each 'cases' entry holds a sequence of 'snapshots' (one per "
            "'t_values' entry) produced by *sequential*, stateful update(t) "
            "calls on a single TimeZeeman/DiscreteTimeZeeman instance -- "
            "order matters, this is not a set of independent evaluations."
        ),
        "cases": {
            "time_zeeman": _case_doc(
                "time_zeeman",
                "TimeZeeman with field_expression=Expression(('0','A*x0*cos(w*t)','0'))"
                ", A=2e5, w=2*pi*1e9 rad/s, m=(0.6,0.8,0) constant, "
                "Ms=8e5, unit_length=1e-9, sampled via sequential update(t) "
                "at t in [0, 0.2e-9, 0.55e-9, 0.9e-9] (deliberately not "
                "symmetric around half a period, so no two samples share a "
                "cosine value by coincidence).",
                tz_coords, tz_t, tz_snapshots, generator_argv,
                m_value=[0.6, 0.8, 0.0],
            ),
            "discrete_time_zeeman": _case_doc(
                "discrete_time_zeeman",
                "DiscreteTimeZeeman with field_expression=Expression(('0','t','0'))"
                ", dt_update=2e-10, m=(0.6,0.8,0) constant, Ms=8e5, "
                "unit_length=1e-9, sampled via sequential update(t) at "
                "t in [0, 1e-10, 2e-10, 2.5e-10, 4e-10, 4.1e-10, 4.5e-10] "
                "(deliberately crossing dt_update more than once to pin the "
                "preserved 't_last_update never advances' quirk).",
                dtz_coords, dtz_t, dtz_snapshots, generator_argv,
                m_value=[0.6, 0.8, 0.0],
            ),
        },
    }

    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
