"""Direct DOLFINx FK demag port (Task 11b).

Validates the ported ``finmag.energies.demag.fk_demag.FKDemag`` and its
coordinate-driven boundary-node ordering against:

- the Task 11a golden BEM matrix, reproduced *bit-for-bit* through the new
  DOLFINx boundary extraction after the coordinate-induced permutation (the
  silent-permutation guard the physics contract demands);
- the analytic uniformly-magnetised-cube demag factor (average H = -Ms/3,
  energy = mu0 Ms^2 V / 6);
- coordinate-ordered legacy FEniCS-2019 oracle references (energy, average
  field, and full pointwise field) generated at the frozen oracle commit
  ``ba9280934e188d7f3800e7b9865e70a9422f7687`` for a unit cube and the barmini
  bar (``src/finmag/tests/fixtures/fk_demag_oracle.json``).

[Claude Opus 4.8]
"""

import json
import os
from math import pi

import numpy as np
import pytest
from mpi4py import MPI

import basix.ufl
import dolfinx.fem as fem
import dolfinx.mesh as dm

from finmag.field import Field, associated_scalar_space
from finmag.energies import Demag
from finmag.energies.demag import Demag2D, MacroGeometry
from finmag.energies.demag.fk_demag import FKDemag, boundary_bem_arrays
from finmag.native.bem_arrays import compute_bem_fk_from_arrays
from finmag.tests.test_native_bem_arrays_dolfinx import (
    CUBE_COORDS, CUBE_CELLS, GOLDEN_BEM_FK)

mu0 = 4.0 * pi * 1e-7

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                        "fk_demag_oracle.json")
ORACLE = json.load(open(_FIXTURE))

# Declared tolerances (serial). Pointwise/average agreement is limited by the
# Krylov solver tolerance (1e-6), not the discretisation; energy is a smoother
# integrated functional and agrees far tighter.
TOL_BEM_GOLDEN = 1e-13          # bit-for-bit BEM reproduction
TOL_ENERGY_REL = 1e-5           # energy vs legacy oracle (same mesh)
TOL_AVG_REL = 1e-3              # average field vs legacy oracle
TOL_POINTWISE_REL = 1e-4        # pointwise field vs legacy oracle, /|H|_max


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _kuhn_cube():
    """DOLFINx unit cube with the exact 6-tet main-diagonal (Kuhn) split that
    legacy ``UnitCubeMesh(1, 1, 1)`` uses, so its boundary triangulation is the
    same surface the Task 11a golden matrix was captured on."""
    tets = np.array([
        [0, 1, 2, 7], [0, 1, 3, 7], [0, 4, 2, 7],
        [0, 4, 5, 7], [0, 6, 3, 7], [0, 6, 5, 7],
    ], dtype=np.int64)
    el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    return dm.create_mesh(MPI.COMM_WORLD, tets, el, CUBE_COORDS.copy())


def _coord_permutation(coords, reference):
    """Index array ``perm`` s.t. ``coords[perm[k]]`` matches ``reference[k]``."""
    perm = np.empty(len(reference), dtype=int)
    for k, rc in enumerate(reference):
        perm[k] = int(np.argmin(np.linalg.norm(coords - rc, axis=1)))
    assert len(set(perm.tolist())) == len(reference)
    return perm


def _demag_on(mesh, m_vec, Ms, unit_length):
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))
    mv = np.asarray(m_vec, dtype=float)
    mv = mv / np.linalg.norm(mv)
    m = Field(S3, tuple(mv))
    Ms_field = Field(DG, Ms)
    demag = FKDemag()
    demag.setup(m, Ms_field, unit_length)
    return demag, S3


# --------------------------------------------------------------------------
# boundary-ordering contract, pinned to the 11a golden matrix
# --------------------------------------------------------------------------

def test_extraction_reproduces_golden_bem_kuhn():
    """The DOLFINx boundary extraction reproduces the 11a golden matrix
    bit-for-bit (up to the coordinate-induced permutation) on the exact legacy
    Kuhn triangulation."""
    mesh = _kuhn_cube()
    S1 = fem.functionspace(mesh, ("Lagrange", 1))
    coords, cells, b2g = boundary_bem_arrays(mesh, S1)
    bem, _ = compute_bem_fk_from_arrays(coords, cells,
                                        np.asarray(b2g, dtype=np.int64))
    perm = _coord_permutation(coords, CUBE_COORDS)
    permuted = bem[np.ix_(perm, perm)]
    np.testing.assert_allclose(permuted, GOLDEN_BEM_FK, rtol=0,
                               atol=TOL_BEM_GOLDEN)


def test_extraction_reproduces_golden_bem_create_unit_cube():
    """The realistic ``create_unit_cube(1,1,1)`` path also reproduces golden
    (DOLFINx uses the same Kuhn split), proving the extraction is not tied to a
    hand-built mesh."""
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    S1 = fem.functionspace(mesh, ("Lagrange", 1))
    coords, cells, b2g = boundary_bem_arrays(mesh, S1)
    bem, _ = compute_bem_fk_from_arrays(coords, cells,
                                        np.asarray(b2g, dtype=np.int64))
    perm = _coord_permutation(coords, CUBE_COORDS)
    permuted = bem[np.ix_(perm, perm)]
    np.testing.assert_allclose(permuted, GOLDEN_BEM_FK, rtol=0,
                               atol=TOL_BEM_GOLDEN)


def test_boundary_triangles_are_outward_oriented():
    """Every extracted boundary triangle winds so its normal points out of the
    cube (the winding the Lindholm double-layer kernel assumes)."""
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    S1 = fem.functionspace(mesh, ("Lagrange", 1))
    coords, cells, _ = boundary_bem_arrays(mesh, S1)
    centre = coords.mean(axis=0)
    for tri in cells:
        p0, p1, p2 = coords[tri]
        normal = np.cross(p1 - p0, p2 - p0)
        outward = ((p0 + p1 + p2) / 3.0) - centre
        assert np.dot(normal, outward) > 0.0


def test_bem_row_sum_identity_through_extraction():
    """FK BEM row sums equal -1 through the DOLFINx extraction (closed-surface
    solid-angle identity), a triangulation-independent sanity check."""
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    S1 = fem.functionspace(mesh, ("Lagrange", 1))
    coords, cells, b2g = boundary_bem_arrays(mesh, S1)
    bem, _ = compute_bem_fk_from_arrays(coords, cells,
                                        np.asarray(b2g, dtype=np.int64))
    np.testing.assert_allclose(bem.sum(axis=1), -np.ones(len(coords)),
                               rtol=0, atol=1e-10)


def test_b2g_map_indexes_scalar_dofs_by_coordinate():
    """``b2g`` maps each BEM-local node to the S1 dof at the same coordinate."""
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    S1 = fem.functionspace(mesh, ("Lagrange", 1))
    coords, _, b2g = boundary_bem_arrays(mesh, S1)
    dof_coords = S1.tabulate_dof_coordinates()
    np.testing.assert_allclose(dof_coords[b2g], coords, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# analytic uniformly-magnetised cube
# --------------------------------------------------------------------------

def test_uniform_cube_average_demag_factor():
    """A uniformly magnetised cube has average demag field -Ms/3 (each axis
    demag factor 1/3). The coarse FK mesh converges toward it from above."""
    Ms = 8.0e5
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    demag, _ = _demag_on(mesh, (1, 0, 0), Ms, 1e-9)
    avg = demag.average_field()
    # dominant component negative and within ~10% of the analytic -Ms/3
    assert avg[0] < 0.0
    assert abs(avg[0] / (-Ms / 3.0) - 1.0) < 0.1
    # transverse averages are small (Kuhn split breaks exact cubic symmetry)
    assert abs(avg[1]) < 0.05 * Ms
    assert abs(avg[2]) < 0.05 * Ms


def test_uniform_cube_energy_matches_analytic():
    """E = mu0 Ms^2 V / 6 for a uniformly magnetised cube."""
    Ms = 8.0e5
    unit_length = 1e-9
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    demag, _ = _demag_on(mesh, (1, 0, 0), Ms, unit_length)
    E = demag.compute_energy()
    V = (1.0 * unit_length) ** 3
    E_expected = (1.0 / 6.0) * mu0 * Ms ** 2 * V
    assert abs(E - E_expected) / E_expected < 0.1


# --------------------------------------------------------------------------
# coordinate-ordered legacy oracle comparisons
# --------------------------------------------------------------------------

def _oracle_compare(name, mesh, m_vec, Ms, unit_length):
    ref = ORACLE[name]
    demag, S3 = _demag_on(mesh, m_vec, Ms, unit_length)
    E = demag.compute_energy()
    avg = demag.average_field()
    H = demag.compute_field().reshape(-1, 3)
    coords = S3.tabulate_dof_coordinates()
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    coords_s, H_s = coords[order], H[order]

    ref_coords = np.asarray(ref["coordinates"])
    ref_H = np.asarray(ref["H_vertex"])
    ref_avg = np.asarray(ref["average_field"])

    # meshes coincide vertex-for-vertex (create_box == legacy BoxMesh)
    assert coords_s.shape == ref_coords.shape
    np.testing.assert_allclose(coords_s, ref_coords, rtol=0, atol=1e-9)

    assert abs(E - ref["energy"]) / abs(ref["energy"]) < TOL_ENERGY_REL
    denom = np.abs(ref_avg) + 1.0
    assert np.max(np.abs(avg - ref_avg) / denom) < TOL_AVG_REL
    scale = np.abs(ref_H).max()
    assert np.abs(H_s - ref_H).max() / scale < TOL_POINTWISE_REL


def test_oracle_cube_field_energy():
    _oracle_compare("cube", dm.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4),
                    (1, 0, 0), 8.0e5, 1e-9)


def test_oracle_barmini_field_energy():
    """barmini-class workflow: 3x3x10 nm bar, m=(1,0,1), compiled FK demag,
    compared to the frozen legacy oracle on the coordinate-identical mesh."""
    mesh = dm.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (3.0, 3.0, 10.0)],
        [2, 2, 4], dm.CellType.tetrahedron)
    _oracle_compare("barmini", mesh, (1, 0, 1), 0.86e6, 1e-9)


# --------------------------------------------------------------------------
# interaction contract + energy_density
# --------------------------------------------------------------------------

def test_interaction_contract():
    demag = FKDemag(name="MyDemag")
    assert demag.name == "MyDemag"
    assert demag.in_jacobian is False
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    demag, S3 = _demag_on(mesh, (1, 0, 0), 8.0e5, 1e-9)
    # compute_field returns a flat owned array matching m's owned dofs
    m = Field(S3, (1.0, 0.0, 0.0))
    assert demag.compute_field().size == m.as_array().size


def test_energy_density_integrates_to_total_energy():
    Ms = 8.0e5
    unit_length = 1e-9
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    demag, _ = _demag_on(mesh, (1, 0, 0), Ms, unit_length)
    rho = demag.energy_density()
    total = demag.compute_energy()
    # sum(rho_i * V_i) ~ total energy; nodal volumes are mesh-unit, scaled here
    V_nodal = demag._nodal_volumes_S1 * unit_length ** demag.dim
    assert abs(np.dot(rho, V_nodal) - total) / abs(total) < 1e-6
    fn = demag.energy_density_function()
    assert fn.x.array.size >= rho.size


# --------------------------------------------------------------------------
# deferred variants raise by name
# --------------------------------------------------------------------------

def test_demag_factory_returns_fk_demag():
    d = Demag()
    assert isinstance(d, FKDemag)
    d2 = Demag(solver="FK")
    assert isinstance(d2, FKDemag)


def test_deferred_demag_solvers_raise_by_name():
    with pytest.raises(NotImplementedError, match="GCR"):
        Demag(solver="GCR")
    with pytest.raises(NotImplementedError, match="Treecode"):
        Demag(solver="Treecode")
    with pytest.raises(NotImplementedError, match="not implemented"):
        Demag(solver="Nonexistent")


def test_demag2d_and_macrogeometry_raise_by_name():
    with pytest.raises(NotImplementedError, match="Demag2D"):
        Demag2D()
    with pytest.raises(NotImplementedError, match="MacroGeometry"):
        MacroGeometry()


def test_lu_solver_type_raises_by_name():
    with pytest.raises(NotImplementedError, match="LU"):
        FKDemag(solver_type="LU")


def test_macrogeometry_argument_raises_by_name():
    with pytest.raises(NotImplementedError, match="macrogeometry"):
        FKDemag(macrogeometry=object())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
