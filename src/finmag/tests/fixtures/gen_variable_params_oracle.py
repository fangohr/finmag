"""Generate coordinate-ordered legacy references for field-valued parameters (Task 16).

Runs at the immutable FEniCS-2019 oracle via ``dev/bin/run-legacy-oracle``.
Every case uses a 1D interval mesh: 1D connectivity is identical in dolfin and
dolfinx (no tetrahedralisation ambiguity), so coordinate-paired nodal values
compare to floating-point noise. Four cases pin the legacy placement/behaviour
of spatially varying material parameters and region energy accounting:

- ``variable_Ms_exchange``: Exchange with a spatially varying ``Ms`` placed in
  DG0 (the ``test_energy_creation_with_variable_Ms`` contract, strengthened to
  a genuinely varying Ms so the DG0 placement is actually exercised).
- ``nonuniform_A_exchange``: Exchange with spatially varying ``A`` in DG0.
- ``spatially_varying_anisotropy``: UniaxialAnisotropy with a spatially varying
  easy axis (the legacy ``test_spatially_varying_anisotropy`` cos/sin pattern)
  placed in CG1, constant ``K1``.
- ``two_region_zeeman``: Zeeman energy integrated over each of two regions plus
  the whole mesh (the ``test_energies_in_regions`` additivity contract).

The magnetisation ``m = (0.6, 0.8 cos(2 pi x0 / L), 0.8 sin(2 pi x0 / L))`` has
exact unit norm everywhere and varies along the mesh, so the DG0-vs-CG1
placement of the coefficients genuinely changes the assembled result.

No iterative linear solve is involved (box-assemble / direct energy assembly),
so tolerances reflect floating-point summation-order noise between the two FEM
stacks, not solver residual.
"""
import json
import os
import sys

import numpy as np
import dolfin as df

from finmag.field import Field
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman

CELLS = 8
L = 8.0
UNIT_LENGTH = 1e-9
MS = 8.6e5
A = 1.3e-11
K1 = 6.0e5

M_EXPR = ("0.6", "0.8*cos(2*pi*x[0]/8.0)", "0.8*sin(2*pi*x[0]/8.0)")
# spatially varying DG0 coefficients (sampled at cell midpoints)
MS_EXPR = "8.6e5*(1.0 + 0.3*x[0]/8.0)"
A_EXPR = "1.3e-11*(1.0 + 0.5*x[0]/8.0)"
# spatially varying unit easy axis in CG1 (the legacy cos/sin pattern)
AXIS_EXPR = ("cos(0.5*x[0])", "sin(0.5*x[0])", "0.0")


def _mesh():
    return df.IntervalMesh(CELLS, 0.0, L)


def _sorted(mesh, *vertex_arrays):
    nv = mesh.num_vertices()
    coords = mesh.coordinates().reshape(nv, -1)
    coords3 = np.zeros((nv, 3))
    coords3[:, : coords.shape[1]] = coords
    order = np.lexsort((coords3[:, 2], coords3[:, 1], coords3[:, 0]))
    return (coords3[order],) + tuple(a[order] for a in vertex_arrays)


def _field_xyz(S3, flat):
    Hf = Field(S3)
    Hf.f.vector().set_local(flat)
    Hf.f.vector().apply("insert")
    return Hf.get_ordered_numpy_array_xyz().reshape(-1, 3)


def _energy_field_case(mesh, interaction, m, Ms_field):
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    interaction.setup(m, Ms_field, UNIT_LENGTH)
    E = interaction.compute_energy()
    H = _field_xyz(S3, interaction.compute_field())
    m_xyz = m.get_ordered_numpy_array_xyz().reshape(-1, 3)
    coords, H_s, m_s = _sorted(mesh, H, m_xyz)
    return {"energy": float(E), "coordinates": coords,
            "H_vertex": H_s, "m_vertex": m_s}


def _quantities(res, h_atol, h_rtol):
    return [
        {"name": "m_vertex", "unit": "1", "value_shape": [3],
         "values": res["m_vertex"].tolist(),
         "tolerances": {"absolute": 1e-12, "relative": 1e-12}},
        {"name": "H_vertex", "unit": "A/m", "value_shape": [3],
         "values": res["H_vertex"].tolist(),
         "tolerances": {"absolute": h_atol, "relative": h_rtol}},
    ]


def _oracle_block(generator_argv):
    return {
        "commit": "ba9280934e188d7f3800e7b9865e70a9422f7687",
        "command": generator_argv,
        "generator": "src/finmag/tests/fixtures/gen_variable_params_oracle.py",
        "generator_note": (
            "Committed driver run at the oracle by absolute path via "
            "dev/bin/run-legacy-oracle. Energies/fields are computed by the "
            "frozen legacy finmag energy classes (box-assemble); the driver "
            "only sorts and serialises coordinate-paired nodal values."
        ),
    }


def main():
    generator_argv = [
        "dev/bin/run-legacy-oracle", "--", "pixi", "run", "--locked",
        "env", "PYTHONPATH=src", "python", os.path.abspath(__file__),
    ]
    oracle = _oracle_block(generator_argv)
    mesh_recipe = "dolfin.IntervalMesh(8, 0.0, 8.0)"
    mesh_params = {"cells": CELLS, "x0": 0.0, "x1": L}
    cases = {}

    # --- variable Ms (DG0) exchange --------------------------------------
    mesh = _mesh()
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)
    m = Field(S3, df.Expression(M_EXPR, degree=1), name="m")
    Ms_var = Field(DG, df.Expression(MS_EXPR, degree=0))
    res = _energy_field_case(
        mesh, Exchange(A, method="box-assemble"), m, Ms_var)
    scale = float(np.abs(res["H_vertex"]).max())
    cases["variable_Ms_exchange"] = {
        "oracle": oracle,
        "description": (
            "Exchange with spatially varying Ms in DG0 "
            "(Ms=8.6e5*(1+0.3 x0/8)), A=1.3e-11, m unit-norm cos/sin, "
            "unit_length=1e-9, IntervalMesh(8,0,8)."),
        "mesh": {"recipe": mesh_recipe, "parameters": mesh_params},
        "physical_parameters": {
            "A": {"value": A, "unit": "J/m"},
            "Ms_expression": {"value": MS_EXPR, "unit": "A/m"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "m_expression": {"value": list(M_EXPR), "unit": "1"},
        },
        "coordinates": {"unit": "mesh_coordinate",
                        "ordering": "lexicographic_xyz",
                        "values": res["coordinates"].tolist()},
        "quantities": _quantities(res, 1e-9 * scale, 1e-9),
        "scalar_quantities": [
            {"name": "energy", "unit": "J", "value_shape": [],
             "value": res["energy"],
             "tolerances": {"absolute": 0.0, "relative": 1e-9}}],
    }

    # --- nonuniform A (DG0) exchange -------------------------------------
    mesh = _mesh()
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)
    m = Field(S3, df.Expression(M_EXPR, degree=1), name="m")
    Ms_field = Field(DG, MS)
    A_var = df.Expression(A_EXPR, degree=0)
    res = _energy_field_case(
        mesh, Exchange(A_var, method="box-assemble"), m, Ms_field)
    scale = float(np.abs(res["H_vertex"]).max())
    cases["nonuniform_A_exchange"] = {
        "oracle": oracle,
        "description": (
            "Exchange with spatially varying A in DG0 "
            "(A=1.3e-11*(1+0.5 x0/8)), Ms=8.6e5, m unit-norm cos/sin, "
            "unit_length=1e-9, IntervalMesh(8,0,8)."),
        "mesh": {"recipe": mesh_recipe, "parameters": mesh_params},
        "physical_parameters": {
            "A_expression": {"value": A_EXPR, "unit": "J/m"},
            "Ms": {"value": MS, "unit": "A/m"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "m_expression": {"value": list(M_EXPR), "unit": "1"},
        },
        "coordinates": {"unit": "mesh_coordinate",
                        "ordering": "lexicographic_xyz",
                        "values": res["coordinates"].tolist()},
        "quantities": _quantities(res, 1e-9 * scale, 1e-9),
        "scalar_quantities": [
            {"name": "energy", "unit": "J", "value_shape": [],
             "value": res["energy"],
             "tolerances": {"absolute": 0.0, "relative": 1e-9}}],
    }

    # --- spatially varying anisotropy axis (CG1) -------------------------
    mesh = _mesh()
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)
    m = Field(S3, df.Expression(M_EXPR, degree=1), name="m")
    Ms_field = Field(DG, MS)
    axis = df.Expression(AXIS_EXPR, degree=1)
    res = _energy_field_case(
        mesh, UniaxialAnisotropy(K1, axis, method="box-assemble"),
        m, Ms_field)
    scale = float(np.abs(res["H_vertex"]).max())
    cases["spatially_varying_anisotropy"] = {
        "oracle": oracle,
        "description": (
            "UniaxialAnisotropy with spatially varying unit easy axis in CG1 "
            "(axis=(cos(0.5 x0), sin(0.5 x0), 0)), K1=6e5, Ms=8.6e5, "
            "m unit-norm cos/sin, unit_length=1e-9, IntervalMesh(8,0,8)."),
        "mesh": {"recipe": mesh_recipe, "parameters": mesh_params},
        "physical_parameters": {
            "K1": {"value": K1, "unit": "J/m**3"},
            "axis_expression": {"value": list(AXIS_EXPR), "unit": "1"},
            "Ms": {"value": MS, "unit": "A/m"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "m_expression": {"value": list(M_EXPR), "unit": "1"},
        },
        "coordinates": {"unit": "mesh_coordinate",
                        "ordering": "lexicographic_xyz",
                        "values": res["coordinates"].tolist()},
        "quantities": _quantities(res, 1e-9 * scale, 1e-9),
        "scalar_quantities": [
            {"name": "energy", "unit": "J", "value_shape": [],
             "value": res["energy"],
             "tolerances": {"absolute": 0.0, "relative": 1e-9}}],
    }

    # --- two-region Zeeman energy accounting -----------------------------
    mesh = _mesh()
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    DG = df.FunctionSpace(mesh, "DG", 0)
    m = Field(S3, df.Expression(M_EXPR, degree=1), name="m")
    Ms_field = Field(DG, MS)
    tdim = mesh.topology().dim()
    domains = df.MeshFunction("size_t", mesh, tdim)
    domains.set_all(0)
    # region 1 for cell midpoint x < L/2, region 2 otherwise
    for cell in df.cells(mesh):
        domains[cell] = 1 if cell.midpoint().x() < 0.5 * L else 2
    dx = df.Measure("dx")[domains]
    zeeman = Zeeman(1e6 * np.array([1.0, 0.0, 0.0]))
    zeeman.setup(m, Ms_field, UNIT_LENGTH)
    E_total = float(zeeman.compute_energy(df.dx))
    E1 = float(zeeman.compute_energy(dx=dx(1)))
    E2 = float(zeeman.compute_energy(dx=dx(2)))
    m_xyz = m.get_ordered_numpy_array_xyz().reshape(-1, 3)
    coords, m_s = _sorted(mesh, m_xyz)
    cases["two_region_zeeman"] = {
        "oracle": oracle,
        "description": (
            "Zeeman(H=1e6 x) energy on two regions (cell midpoint x<4 vs x>=4) "
            "and the whole mesh; Ms=8.6e5, m unit-norm cos/sin, "
            "unit_length=1e-9, IntervalMesh(8,0,8). Region energies sum to "
            "the total (additivity contract)."),
        "mesh": {"recipe": mesh_recipe, "parameters": mesh_params},
        "physical_parameters": {
            "H_zeeman": {"value": [1e6, 0.0, 0.0], "unit": "A/m"},
            "Ms": {"value": MS, "unit": "A/m"},
            "unit_length": {"value": UNIT_LENGTH, "unit": "m"},
            "m_expression": {"value": list(M_EXPR), "unit": "1"},
            "region_split_x": {"value": 0.5 * L, "unit": "mesh_coordinate"},
        },
        "coordinates": {"unit": "mesh_coordinate",
                        "ordering": "lexicographic_xyz",
                        "values": coords.tolist()},
        "quantities": [
            {"name": "m_vertex", "unit": "1", "value_shape": [3],
             "values": m_s.tolist(),
             "tolerances": {"absolute": 1e-12, "relative": 1e-12}}],
        "scalar_quantities": [
            {"name": "energy_total", "unit": "J", "value_shape": [],
             "value": E_total,
             "tolerances": {"absolute": 0.0, "relative": 1e-9}},
            {"name": "energy_region_1", "unit": "J", "value_shape": [],
             "value": E1,
             "tolerances": {"absolute": 0.0, "relative": 1e-9}},
            {"name": "energy_region_2", "unit": "J", "value_shape": [],
             "value": E2,
             "tolerances": {"absolute": 0.0, "relative": 1e-9}}],
    }

    doc = {
        "schema_version": 1,
        "note": (
            "Coordinate-ordered (lexicographic xyz) legacy FEniCS-2019 "
            "references for field-valued material parameters and region energy "
            "accounting (Task 16), generated at the frozen oracle commit via "
            "dev/bin/run-legacy-oracle on 1D interval meshes. Each case is a "
            "schema-v1 fixture (oracle/mesh/physical_parameters/coordinates/"
            "quantities); 'scalar_quantities' carries whole-domain / per-region "
            "energies. No iterative solve is involved, so tolerances reflect "
            "floating-point summation-order noise between the two FEM stacks."
        ),
        "cases": cases,
    }
    json.dump(doc, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
