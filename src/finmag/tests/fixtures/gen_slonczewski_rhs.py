"""Generate a coordinate-ordered legacy Slonczewski STT dm/dt reference.

Runs at the immutable FEniCS-2019 oracle. Builds a 1D interval mesh (identical
tetrahedral/edge connectivity in dolfin and dolfinx, unlike a 3D box) carrying a
spatially uniform 3-component magnetisation held at an angle to the fixed
polarisation direction p, activates the Slonczewski/Xiao spin-transfer torque
via ``LLG.use_slonczewski``, adds a Zeeman field so precession/damping are also
exercised, evaluates ``LLG.solve(0.0)``, and dumps coordinate-sorted m, H_eff
and dm/dt to JSON on stdout. The dm/dt therefore pins the exact native
``slonczewski_xiao_i`` prefactors (aJ / (1 + ...) structure, m x p and
m x (m x p) terms, epsilonprime secondary torque).
"""
import json
import numpy as np
import dolfin as df

from finmag.physics.llg import LLG
from finmag.energies import Zeeman

# ---- parameters -----------------------------------------------------------
L = 4.0            # mesh-coordinate length
cells = 4          # -> vertices at 0,1,2,3,4
unit_length = 1e-9
Ms = 8.6e5
alpha = 0.1
H_zeeman = (1.0e4, 2.0e4, -5.0e3)
# uniform m held at an angle to p (m . p = 0.8, not parallel)
m_uniform = (0.6, 0.0, 0.8)
# Slonczewski parameters
J = 1.0e12          # current density A/m^2
P = 0.4             # polarisation
d = 2.0e-9          # free-layer thickness m
p_dir = (0.0, 0.0, 1.0)
Lambda = 2.0
epsilonprime = 0.1  # nonzero secondary (field-like) torque

mesh = df.IntervalMesh(cells, 0.0, L)
S1 = df.FunctionSpace(mesh, "Lagrange", 1)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

llg = LLG(S1, S3, unit_length=unit_length)
llg.Ms = Ms
llg.set_alpha(alpha)
llg.do_precession = True
llg.set_m(m_uniform, normalise=True)

llg.effective_field.add(Zeeman(H_zeeman))

llg.use_slonczewski(J, P, d, p_dir, Lambda=Lambda, epsilonprime=epsilonprime)

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
            "<abs-path-to>/gen_slonczewski_rhs.py",
        ],
        "generator": "src/finmag/tests/fixtures/gen_slonczewski_rhs.py",
        "generator_note": (
            "Committed driver run at the oracle by absolute path via "
            "dev/bin/run-legacy-oracle (the script does not exist inside the "
            "detached oracle checkout, so it is supplied by absolute path). "
            "The RHS itself is computed by the frozen finmag LLG.solve with "
            "do_slonczewski active (native calc_llg_slonczewski_dmdt / "
            "slonczewski_xiao_i); the driver only sorts and serialises "
            "coordinate-paired nodal values."
        ),
    },
    "description": (
        "Deterministic LLG dm/dt with the Slonczewski/Xiao spin-transfer torque "
        "on a 1D interval mesh: spatially uniform m at an angle to the "
        "polarisation p, plus Zeeman. 1D connectivity is identical in dolfin "
        "and dolfinx, isolating the STT physics from tetrahedralisation."
    ),
    "mesh": {
        "recipe": "dolfin.IntervalMesh(4, 0.0, 4.0)",
        "parameters": {"cells": cells, "x0": 0.0, "x1": L},
    },
    "physical_parameters": {
        "Ms": {"value": Ms, "unit": "A/m"},
        "alpha": {"value": alpha, "unit": "1"},
        "gamma": {"value": llg.gamma, "unit": "m/(A*s)"},
        "c": {"value": llg.c, "unit": "1/s"},
        "do_precession": {"value": True, "unit": "1"},
        "unit_length": {"value": unit_length, "unit": "m"},
        "H_zeeman": {"value": list(H_zeeman), "unit": "A/m"},
        "m_uniform": {"value": list(m_uniform), "unit": "1"},
        "J": {"value": J, "unit": "A/m^2"},
        "P": {"value": P, "unit": "1"},
        "d": {"value": d, "unit": "m"},
        "p": {"value": list(p_dir), "unit": "1"},
        "Lambda": {"value": Lambda, "unit": "1"},
        "epsilonprime": {"value": epsilonprime, "unit": "1"},
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
