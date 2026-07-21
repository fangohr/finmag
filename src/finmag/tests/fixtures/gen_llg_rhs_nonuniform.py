"""Generate a coordinate-ordered legacy LLG RHS reference (nonuniform case).

Runs at the immutable FEniCS-2019 oracle. Builds a 1D interval mesh (identical
tetrahedral/edge connectivity in dolfin and dolfinx, unlike a 3D box) carrying a
spatially varying 3-component magnetisation, adds Exchange + Zeeman, evaluates
LLG.solve(0.0), and dumps coordinate-sorted m, H_eff and dm/dt to JSON on stdout.
"""
import json
import numpy as np
import dolfin as df

from finmag.physics.llg import LLG
from finmag.energies import Exchange, Zeeman

# ---- parameters -----------------------------------------------------------
L = 4.0            # mesh-coordinate length
cells = 4          # -> vertices at 0,1,2,3,4
unit_length = 1e-9
Ms = 8.6e5
A = 1.3e-11
alpha = 0.1
H_zeeman = (1.0e4, 2.0e4, -5.0e3)
m_expr = ("cos(0.4*x[0])", "sin(0.4*x[0])", "0.5")

mesh = df.IntervalMesh(cells, 0.0, L)
S1 = df.FunctionSpace(mesh, "Lagrange", 1)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

llg = LLG(S1, S3, unit_length=unit_length)
llg.Ms = Ms
llg.set_alpha(alpha)
llg.do_precession = True
llg.set_m(m_expr, normalise=True)

llg.effective_field.add(Exchange(A))
llg.effective_field.add(Zeeman(H_zeeman))

dmdt = llg.solve(0.0)  # xxx component-blocked, vertex-ordered

nv = mesh.num_vertices()
coords = mesh.coordinates().reshape(nv, -1)  # vertex order, (nv, gdim)

# vertex-ordered nodal values
m_xyz = llg._m_field.get_ordered_numpy_array_xyz().reshape(nv, 3)

from finmag.field import Field as LegacyField
Hf = LegacyField(S3)
Hf.f.vector().set_local(llg.effective_field.H_eff)
Hf.f.vector().apply("insert")
H_xyz = Hf.get_ordered_numpy_array_xyz().reshape(nv, 3)

dmdt_xyz = dmdt.reshape(3, nv).T  # (nv, 3), vertex order

# pad coordinates to 3D and lexicographically sort
coords3 = np.zeros((nv, 3))
coords3[:, : coords.shape[1]] = coords
order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))

coords3 = coords3[order]
m_xyz = m_xyz[order]
H_xyz = H_xyz[order]
dmdt_xyz = dmdt_xyz[order]

doc = {
    "schema_version": 1,
    "oracle": {
        "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
        "command": [
            "dev/bin/run-legacy-oracle",
            "--",
            "pixi",
            "run",
            "--locked",
            "env",
            "PYTHONPATH=src",
            "python",
            "<abs-path-to>/gen_llg_rhs_nonuniform.py",
        ],
        "generator": "src/finmag/tests/fixtures/gen_llg_rhs_nonuniform.py",
        "generator_note": (
            "Committed driver run at the oracle by absolute path via "
            "dev/bin/run-legacy-oracle (the script does not exist inside the "
            "detached oracle checkout, so it is supplied by absolute path). "
            "The RHS itself is computed by the frozen finmag LLG.solve; the "
            "driver only sorts and serialises coordinate-paired nodal values."
        ),
    },
    "description": (
        "Deterministic LLG dm/dt on a 1D interval mesh with spatially varying "
        "3-component m, Exchange + Zeeman. 1D connectivity is identical in "
        "dolfin and dolfinx, isolating the RHS physics from tetrahedralisation."
    ),
    "mesh": {
        "recipe": "dolfin.IntervalMesh(4, 0.0, 4.0)",
        "parameters": {"cells": cells, "x0": 0.0, "x1": L},
    },
    "physical_parameters": {
        "Ms": {"value": Ms, "unit": "A/m"},
        "A": {"value": A, "unit": "J/m"},
        "alpha": {"value": alpha, "unit": "1"},
        "gamma": {"value": llg.gamma, "unit": "m/(A*s)"},
        "c": {"value": llg.c, "unit": "1/s"},
        "do_precession": {"value": True, "unit": "1"},
        "unit_length": {"value": unit_length, "unit": "m"},
        "H_zeeman": {"value": list(H_zeeman), "unit": "A/m"},
        "m_expression": {"value": list(m_expr), "unit": "1"},
    },
    "coordinates": {
        "unit": "mesh_coordinate",
        "ordering": "lexicographic_xyz",
        "values": coords3.tolist(),
    },
    "quantities": [
        {
            "name": "m",
            "unit": "1",
            "value_shape": [3],
            "values": m_xyz.tolist(),
            "tolerances": {"absolute": 1e-12, "relative": 1e-12},
        },
        {
            "name": "effective_field",
            "unit": "A/m",
            "value_shape": [3],
            "values": H_xyz.tolist(),
            "tolerances": {"absolute": 1e-3, "relative": 1e-9},
        },
        {
            "name": "dmdt",
            "unit": "1/s",
            "value_shape": [3],
            "values": dmdt_xyz.tolist(),
            "tolerances": {"absolute": 1.0, "relative": 1e-9},
        },
    ],
}

print(json.dumps(doc, indent=2, sort_keys=True))
