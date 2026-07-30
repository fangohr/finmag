"""Generate a coordinate-ordered legacy LLG RHS reference with varying alpha (Task 16).

Runs at the immutable FEniCS-2019 oracle. Identical in spirit to
``gen_llg_rhs_nonuniform.py`` but with a *spatially varying* Gilbert damping
``alpha(x)`` (a CG1 nodal field), so the per-node ``alpha`` in both the damping
term and ``gamma_LL = gamma/(1+alpha^2)`` of the native ``calc_llg_dmdt`` is
genuinely exercised. 1D interval mesh: connectivity is identical in dolfin and
dolfinx, isolating the RHS physics from tetrahedralisation.
"""
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag.physics.llg import LLG
from finmag.energies import Exchange, Zeeman
from finmag.field import Field as LegacyField

L = 4.0
CELLS = 4
UNIT_LENGTH = 1e-9
MS = 8.6e5
A = 1.3e-11
ALPHA_EXPR = "0.05 + 0.1*x[0]/4.0"
H_ZEEMAN = (1.0e4, 2.0e4, -5.0e3)
M_EXPR = ("cos(0.4*x[0])", "sin(0.4*x[0])", "0.5")


def main():
    mesh = df.IntervalMesh(CELLS, 0.0, L)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

    llg = LLG(S1, S3, unit_length=UNIT_LENGTH)
    llg.Ms = MS
    llg.set_alpha(df.Expression(ALPHA_EXPR, degree=1))
    llg.do_precession = True
    llg.set_m(df.Expression(M_EXPR, degree=1), normalise=True)

    llg.effective_field.add(Exchange(A))
    llg.effective_field.add(Zeeman(H_ZEEMAN))

    dmdt = llg.solve(0.0)  # xxx component-blocked, vertex-ordered

    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)

    m_xyz = llg._m_field.get_ordered_numpy_array_xyz().reshape(nv, 3)

    Hf = LegacyField(S3)
    Hf.f.vector().set_local(llg.effective_field.H_eff)
    Hf.f.vector().apply("insert")
    H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(nv, 3)

    # per-node alpha, vertex-ordered (same view as m/H). Legacy ``llg.alpha``
    # is a raw ``df.Function`` on S1; wrap it in a Field to reuse the ordering.
    alpha_field = LegacyField(S1)
    alpha_field.f.vector().set_local(llg.alpha.vector().get_local())
    alpha_field.f.vector().apply("insert")
    alpha_xyz = alpha_field.get_ordered_numpy_array()

    dmdt_xyz = dmdt.reshape(3, nv).T

    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))
    coords3 = coords3[order]
    m_xyz = m_xyz[order]
    H_xyz = H_xyz[order]
    alpha_xyz = alpha_xyz[order]
    dmdt_xyz = dmdt_xyz[order]

    doc = {
        "schema_version": 1,
        "oracle": {
            "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
            "command": [
                "dev/bin/run-legacy-oracle", "--", "pixi", "run", "--locked",
                "env", "PYTHONPATH=src", "python",
                "<abs-path-to>/gen_spatially_varying_alpha_rhs.py",
            ],
            "generator": (
                "src/finmag/tests/fixtures/gen_spatially_varying_alpha_rhs.py"),
            "generator_note": (
                "Committed driver run at the oracle by absolute path via "
                "dev/bin/run-legacy-oracle. The RHS is computed by the frozen "
                "finmag LLG.solve with a spatially varying alpha; the driver "
                "only sorts and serialises coordinate-paired nodal values."),
        },
        "description": (
            "Deterministic LLG dm/dt on a 1D interval mesh with spatially "
            "varying 3-component m, spatially varying alpha(x), Exchange + "
            "Zeeman. Per-node alpha enters the damping term and gamma_LL."),
        "mesh": {
            "recipe": "dolfin.IntervalMesh(4, 0.0, 4.0)",
            "parameters": {"cells": CELLS, "x0": 0.0, "x1": L},
        },
        "physical_parameters": {
            "Ms": {"value": MS, "unit": "A/m"},
            "A": {"value": A, "unit": "J/m"},
            "alpha_expression": {"value": ALPHA_EXPR, "unit": "1"},
            "gamma": {"value": llg.gamma, "unit": "m/(A*s)"},
            "c": {"value": llg.c, "unit": "1/s"},
            "do_precession": {"value": True, "unit": "1"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "H_zeeman": {"value": list(H_ZEEMAN), "unit": "A/m"},
            "m_expression": {"value": list(M_EXPR), "unit": "1"},
        },
        "coordinates": {
            "unit": "mesh_coordinate", "ordering": "lexicographic_xyz",
            "values": coords3.tolist(),
        },
        "quantities": [
            {"name": "m", "unit": "1", "value_shape": [3],
             "values": m_xyz.tolist(),
             "tolerances": {"absolute": 1e-12, "relative": 1e-12}},
            {"name": "alpha", "unit": "1", "value_shape": [],
             "values": alpha_xyz.tolist(),
             "tolerances": {"absolute": 1e-12, "relative": 1e-12}},
            {"name": "effective_field", "unit": "A/m", "value_shape": [3],
             "values": H_xyz.tolist(),
             "tolerances": {"absolute": 1e-3, "relative": 1e-9}},
            {"name": "dmdt", "unit": "1/s", "value_shape": [3],
             "values": dmdt_xyz.tolist(),
             "tolerances": {"absolute": 1.0, "relative": 1e-9}},
        ],
    }
    print(json.dumps(doc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
