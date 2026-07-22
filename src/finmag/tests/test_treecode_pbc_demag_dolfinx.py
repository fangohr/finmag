"""Treecode and periodic (MacroGeometry) demag, ported to DOLFINx (Task 23).

The native ``finmag.native.treecode_bem`` Cython extension (pure-C octree
fast-summation + Lindholm boundary-element kernels, no libdolfin, no
Boost.Python) is rebuilt for the DOLFINx pixi lane and consumed by two ported
demag surfaces on top of the Task 11b FK foundation:

- ``TreecodeBEM`` (``Demag(solver='Treecode')``): replaces the dense
  Fredkin-Koehler BEM matrix-vector product with the treecode fast-summation
  approximation of the *same* double-layer operator;
- ``MacroGeometry`` / ``FKDemag(macrogeometry=...)``: the periodic
  boundary-element demag, summing Lindholm image contributions over a lattice
  of in-plane translation vectors.

Validation basis (documented in ``transition-notes.org`` -- the oracle env
never built ``treecode_bem`` so there are NO treecode oracle fixtures; this is
the plan's accepted cross-method + analytic evidence exception):

1. the single-tile periodic BEM reproduces the Task 11a golden dense FK BEM
   *bit-for-bit* (same Lindholm convention/orientation);
2. the treecode field cross-checks the ported dense FK demag on the same
   geometry, in the direct-sum limit *and* in a genuine multipole-approximation
   regime;
3. periodic-image convergence and the analytic thin-film demag limit for the
   MacroGeometry path;
4. one ``Simulation.add(Demag(...))`` end-to-end smoke for each of the
   Treecode solver and the non-coincident ``MacroGeometry`` path.

*What the accuracy tests actually prove* (round-1 review correction -- the
approximation regime is geometry-dependent, not a knob-only property):
``test_treecode_field_matches_fk_demag_cube`` and
``test_treecode_default_knobs_machine_precision_on_compact_box`` exercise
*compact, convex* boundaries (cubes/spheres). There, treecode-vs-FK agreement
is machine precision (``~1e-14``) at *every* ``mac``/``p``/``num_limit``
combination, including the legacy default ``mac=0.3, p=3`` -- swept up to
1178 boundary nodes. This is NOT the multipole-approximation regime; it is a
kernel/orientation equivalence check. The reason (cited from
``native/src/treecode_bem/``): the octree's multipole-acceptance test in
``treecode_bem_I.c`` (``mac_square * R > tree->radius_square``, e.g. line 18)
compares the target-to-cluster distance ``R`` against the cluster's own
bounding radius; for a compact convex boundary, evaluation points sit ON the
same surface whose diameter bounds every cluster's separation, so no cluster
is ever "far enough" at reasonable ``mac`` -- the near-field sparse matrix
built by ``bulid_indices_I`` covers the whole sum and multipole evaluation is
never invoked. ``test_treecode_approximation_regime_reachable_on_elongated_bar``
uses a 50:1 aspect-ratio bar instead (well-separated octree clusters at the
two ends), which DOES enter the approximation regime and reproduces a genuine,
monotonic ``mac``-accuracy trend (measured, not asserted from memory); see that
test's docstring and ``transition-notes.org`` for the numbers.
``test_treecode_p_order_does_not_affect_accuracy`` confirms from the C source
(and empirically) that the multipole order ``p`` is a dead parameter in this
kernel.

[Claude Opus 4.8] [Claude Sonnet 5]
"""

import sys
from math import pi

import numpy as np
import pytest
from mpi4py import MPI

import basix.ufl
import dolfinx.fem as fem
import dolfinx.mesh as dm

from finmag.field import Field
from finmag.sim.sim import Simulation
from finmag.energies import Demag
from finmag.energies.demag import MacroGeometry, Demag2D
from finmag.energies.demag.fk_demag import (
    FKDemag, boundary_bem_arrays, boundary_solid_angles)
from finmag.energies.demag.treecode_bem import TreecodeBEM
from finmag.energies.demag.fk_demag_pbc import BMatrixPBC
from finmag.native.bem_arrays import compute_bem_fk_from_arrays
from finmag.tests.test_native_bem_arrays_dolfinx import (
    CUBE_COORDS, GOLDEN_BEM_FK)

mu0 = 4.0 * pi * 1e-7


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _kuhn_cube():
    tets = np.array([
        [0, 1, 2, 7], [0, 1, 3, 7], [0, 4, 2, 7],
        [0, 4, 5, 7], [0, 6, 3, 7], [0, 6, 5, 7],
    ], dtype=np.int64)
    el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    return dm.create_mesh(MPI.COMM_WORLD, tets, el, CUBE_COORDS.copy())


def _coord_permutation(coords, reference):
    perm = np.empty(len(reference), dtype=int)
    for k, rc in enumerate(reference):
        perm[k] = int(np.argmin(np.linalg.norm(coords - rc, axis=1)))
    assert len(set(perm.tolist())) == len(reference)
    return perm


def _box(nx, ny, nz, lx, ly, lz):
    return dm.create_box(
        MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([lx, ly, lz])],
        [nx, ny, nz], cell_type=dm.CellType.tetrahedron)


def _fields(mesh, m_vec, Ms):
    S3 = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(mesh, ("DG", 0))
    mv = np.asarray(m_vec, float)
    mv = mv / np.linalg.norm(mv)
    return Field(S3, tuple(mv)), Field(DG, Ms)


# --------------------------------------------------------------------------
# 1. native module
# --------------------------------------------------------------------------

def test_native_treecode_imports_dolfin_free():
    import finmag.native.treecode_bem as tb
    for sym in ("FastSum", "compute_solid_angle_single",
                "compute_boundary_element", "build_boundary_matrix"):
        assert hasattr(tb, sym)
    assert "dolfin" not in sys.modules


# --------------------------------------------------------------------------
# 2. periodic BEM reproduces the golden dense FK BEM (single tile)
# --------------------------------------------------------------------------

def test_single_tile_pbc_matches_golden_bem():
    mesh = _kuhn_cube()
    S1 = fem.functionspace(mesh, ("Lagrange", 1))
    coords, cells, b2g = boundary_bem_arrays(mesh, S1)
    pbc = BMatrixPBC(mesh, Ts=[(0.0, 0.0, 0.0)])
    golden, _ = compute_bem_fk_from_arrays(
        coords, cells, np.asarray(b2g, np.int64))
    # BMatrixPBC extracts its own boundary ordering; align by coordinate.
    perm_g = _coord_permutation(coords, CUBE_COORDS)
    perm_p = _coord_permutation(pbc.coords, CUBE_COORDS)
    g = golden[np.ix_(perm_g, perm_g)]
    p = pbc.bm[np.ix_(perm_p, perm_p)]
    np.testing.assert_allclose(p, GOLDEN_BEM_FK, rtol=0, atol=1e-12)
    np.testing.assert_allclose(g, p, rtol=0, atol=1e-12)


def test_single_tile_pbc_row_sum_is_minus_one():
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    pbc = BMatrixPBC(mesh, Ts=[(0.0, 0.0, 0.0)])
    np.testing.assert_allclose(
        pbc.bm.sum(axis=1), -np.ones(len(pbc.coords)), rtol=0, atol=1e-10)


# --------------------------------------------------------------------------
# 3. MacroGeometry translation vectors + validation
# --------------------------------------------------------------------------

def test_macrogeometry_requires_odd_positive_tiles():
    with pytest.raises(ValueError):
        MacroGeometry(nx=2, ny=1)
    with pytest.raises(ValueError):
        MacroGeometry(nx=1, ny=4)
    # legacy quirk preserved: nx/ny of 0 fall back to 1 via `nx or 1` (valid).
    MacroGeometry(nx=1, ny=0)


def test_macrogeometry_ts_lattice():
    mg = MacroGeometry(nx=3, ny=1, dx=10.0, dy=5.0)
    Ts = np.array(mg.compute_Ts(None))
    expected = np.array([[-10.0, 0, 0], [0.0, 0, 0], [10.0, 0, 0]])
    np.testing.assert_allclose(np.sort(Ts[:, 0]), np.sort(expected[:, 0]))
    assert Ts.shape == (3, 3)
    assert np.allclose(Ts[:, 2], 0.0)


def test_macrogeometry_ts_from_mesh_extent():
    mesh = _box(4, 4, 1, 20.0, 12.0, 2.0)
    mg = MacroGeometry(nx=3, ny=3)
    Ts = np.array(mg.compute_Ts(mesh))
    assert Ts.shape == (9, 3)
    # dx/dy inferred from the mesh bounding box.
    assert np.isclose(mg.dx, 20.0) and np.isclose(mg.dy, 12.0)


# --------------------------------------------------------------------------
# 4. treecode cross-checks the dense FK demag (same operator)
# --------------------------------------------------------------------------

def test_treecode_field_matches_fk_demag_cube():
    mesh = _box(6, 6, 6, 30.0, 30.0, 30.0)
    m, Ms = _fields(mesh, (1.0, 0.2, -0.3), 8.6e5)
    fk = FKDemag()
    fk.setup(m, Ms, unit_length=1e-9)
    f_fk = fk.compute_field()
    # direct-sum limit (mac large -> no multipole acceptance): near-exact.
    tc = TreecodeBEM(mac=0.7, p=5, num_limit=100)
    tc.setup(m, Ms, unit_length=1e-9)
    f_tc = tc.compute_field()
    rel = np.max(np.abs(f_tc - f_fk)) / np.max(np.abs(f_fk))
    assert rel < 1e-6, rel


def test_treecode_default_knobs_machine_precision_on_compact_box():
    # Round-1 review finding: on a *compact convex* boundary (a cube), the
    # legacy-default knobs (mac=0.3, p=3) do NOT enter the multipole
    # approximation regime -- the octree's near-field sparse matrix (built by
    # `bulid_indices_I`, native/src/treecode_bem/treecode_bem_I.c) covers the
    # entire sum because no cluster is ever "far enough" from a query point on
    # the same compact surface (see the module docstring and
    # `test_treecode_approximation_regime_reachable_on_elongated_bar` below for
    # the genuine-approximation-regime witness and the C-source citation).
    # This test therefore proves kernel/orientation machine-precision
    # equivalence at these knobs/geometry -- NOT the documented treecode
    # approximation accuracy, which the old 5e-3 bound implied. [Claude Sonnet 5]
    mesh = _box(8, 8, 8, 40.0, 40.0, 40.0)
    m, Ms = _fields(mesh, (1.0, 0.0, 0.0), 8.6e5)
    fk = FKDemag()
    fk.setup(m, Ms, unit_length=1e-9)
    f_fk = fk.compute_field()
    tc = TreecodeBEM(mac=0.3, p=3, num_limit=100)  # legacy defaults
    tc.setup(m, Ms, unit_length=1e-9)
    f_tc = tc.compute_field()
    rel = np.max(np.abs(f_tc - f_fk)) / np.max(np.abs(f_fk))
    assert rel < 1e-10, rel


def test_treecode_approximation_regime_reachable_on_elongated_bar():
    # Round-1 review fix: construct a genuine multipole-approximation-regime
    # witness.  A 50:1 aspect-ratio bar (6nm x 6nm x 300nm) gives the octree
    # well-separated clusters (e.g. one end of the bar vs. a query point at the
    # other), unlike a compact cube -- so the multipole-acceptance test in
    # `treecode_bem_I.c` (`mac_square * R > tree->radius_square`) actually
    # fires, and treecode-vs-FK agreement rises well above machine precision.
    # Numbers measured on this exact geometry (mac sweep at p=3, num_limit=100,
    # correct_factor=10 -- the legacy defaults except mac):
    #   mac=0.7 -> 8.818e-04   mac=0.5 -> 3.230e-04
    #   mac=0.3 -> 4.628e-05   mac=0.1 -> 3.059e-07
    # This reproduces (on the correct geometry) the previously-unreproducible
    # "~4e-5 at mac=0.3" figure, and the sweep is monotonic: tightening `mac`
    # (smaller value -> stricter far-field acceptance) strictly improves
    # accuracy over this range.  NOTE (documented, not asserted, since it is a
    # separate C-source subtlety): `mac` also gates the octree's *subdivision*
    # stopping criterion (`common.c`, `create_tree`, line ~569:
    # `r2*(1-plan->mac) >= plan->r_eps*plan->mac`); as mac -> 1 that condition
    # stops the tree from subdividing at all, so the near-field path again
    # covers everything and accuracy jumps back to machine precision (mac=0.9
    # measured at 9.377e-15 here) -- a degenerate case outside the
    # monotonic-approximation range asserted below, not a counterexample to it.
    mesh = _box(2, 2, 60, 6.0, 6.0, 300.0)
    m, Ms = _fields(mesh, (1.0, 0.1, 0.05), 8.6e5)
    fk = FKDemag()
    fk.setup(m, Ms, unit_length=1e-9)
    f_fk = fk.compute_field()

    macs = (0.7, 0.5, 0.3, 0.1)
    errs = []
    for mac in macs:
        tc = TreecodeBEM(mac=mac, p=3, num_limit=100)
        tc.setup(m, Ms, unit_length=1e-9)
        f_tc = tc.compute_field()
        errs.append(np.max(np.abs(f_tc - f_fk)) / np.max(np.abs(f_fk)))
    errs = np.array(errs)

    # genuinely in the approximation regime (far above machine precision),
    # and within the method's documented accuracy band.
    assert 1e-6 < errs[macs.index(0.3)] < 1e-3, errs[macs.index(0.3)]
    # tightening mac (decreasing it) strictly improves accuracy over this range.
    assert np.all(np.diff(errs) < 0.0), errs


def test_treecode_p_order_does_not_affect_accuracy():
    # The multipole expansion in the C kernel is hard-coded to a fixed 35-term
    # (4th-order) array regardless of the `p` argument threaded through the
    # API: `compute_coefficient_directly`/`compute_moment_directly`
    # (native/src/treecode_bem/common.c, ~lines 916-993) literally comment
    # "Always suppose N=35, n=5, size=n*(n+1)*(n+2)/6" and never index past
    # a[20]/mom[35] using `p`. So `p` cannot move the accuracy of `fast_sum_I`;
    # this is confirmed empirically here (not merely asserted from the
    # source), on the same approximation-regime witness geometry as above.
    mesh = _box(2, 2, 60, 6.0, 6.0, 300.0)
    m, Ms = _fields(mesh, (1.0, 0.1, 0.05), 8.6e5)
    fk = FKDemag()
    fk.setup(m, Ms, unit_length=1e-9)
    f_fk = fk.compute_field()

    errs = []
    for p in (1, 3, 5, 8):
        tc = TreecodeBEM(mac=0.3, p=p, num_limit=100)
        tc.setup(m, Ms, unit_length=1e-9)
        f_tc = tc.compute_field()
        errs.append(np.max(np.abs(f_tc - f_fk)) / np.max(np.abs(f_fk)))
    errs = np.array(errs)
    np.testing.assert_allclose(errs, errs[0], rtol=1e-9, atol=0)


def test_treecode_uniform_sphere_demag_factor():
    from finmag.util.meshes import sphere
    mesh = sphere(r=10.0, maxh=2.0, directory="/tmp/finmag-treecode-meshes")
    m, Ms = _fields(mesh, (0.0, 0.0, 1.0), 8.6e5)
    tc = TreecodeBEM(mac=0.4, p=5, num_limit=100)
    tc.setup(m, Ms, unit_length=1e-9)
    H = tc.compute_field().reshape((-1, 3))
    Hz_avg = H[:, 2].mean()
    assert abs(Hz_avg / Ms.f.x.array[0] + 1.0 / 3.0) < 0.05


def test_demag_factory_treecode_returns_treecodebem():
    d = Demag(solver="Treecode")
    assert isinstance(d, TreecodeBEM)


# --------------------------------------------------------------------------
# 5. periodic (MacroGeometry) FK demag physics
# --------------------------------------------------------------------------

def _pbc_field_avg(mesh, m_vec, Ms, nx=1, ny=1):
    m, Ms_f = _fields(mesh, m_vec, Ms)
    demag = FKDemag(macrogeometry=MacroGeometry(nx=nx, ny=ny))
    demag.setup(m, Ms_f, unit_length=1e-9)
    H = demag.compute_field().reshape((-1, 3))
    return H.mean(axis=0)


def test_pbc_end_to_end_runs():
    mesh = _box(4, 4, 2, 20.0, 20.0, 4.0)
    avg = _pbc_field_avg(mesh, (0, 0, 1), 8.6e5, nx=3, ny=3)
    assert np.all(np.isfinite(avg))


def _field_at_origin(mesh, m_vec, Ms, nx=1, ny=1):
    m, Ms_f = _fields(mesh, m_vec, Ms)
    tol = {"absolute_tolerance": 1e-10, "relative_tolerance": 1e-10,
           "maximum_iterations": int(1e5)}
    demag = FKDemag(macrogeometry=MacroGeometry(nx=nx, ny=ny),
                    parameters={"phi_1": tol, "phi_2": tol})
    demag.setup(m, Ms_f, unit_length=1e-9)
    S3 = m.functionspace
    H = demag.compute_field().reshape((-1, 3))
    coords = S3.tabulate_dof_coordinates()[: H.shape[0]]
    i = int(np.argmin(np.linalg.norm(coords, axis=1)))
    return H[i] / Ms


def _centred_box(n, half):
    return dm.create_box(
        MPI.COMM_WORLD,
        [np.array([-half, -half, -half]), np.array([half, half, half])],
        [n, n, n], cell_type=dm.CellType.tetrahedron)


def test_pbc_image_sum_converges_1d():
    # Physics sanity (the plan's accepted PBC evidence): the 1D periodic image
    # sum converges monotonically to a stable limit as the image count grows.
    # A cube tiled along x approaches an infinite chain of cubes; |Hx| decreases
    # monotonically and the successive increments shrink toward a fixed limit
    # (the single-cube Nx~1/3 relaxes as neighbours are added along x).
    #
    # This test is DELIBERATELY trend-only (round-1 review item): it does not
    # pin the nx->infinity plateau value, because that limit has no known
    # closed-form/independent expression for a periodic array of finite cubes
    # -- pinning it would mean asserting a number computed by this same method
    # at higher image counts, which is not independent validation.  The
    # independent analytic anchor for this MacroGeometry path is the
    # out-of-plane thin-film test below (Nz -> 1 is a textbook result).
    # For documentation only (not asserted, and not encoded as a fast test):
    # extending this same 1D sweep offline to nx=51..351 on this mesh gives
    # -0.10101, -0.10089, -0.10083, -0.10080, -0.10078, -0.10077, consistent
    # with a slow ~1/nx approach to a plateau near -0.1007 (a-la-1/n
    # least-squares fit); this is an extrapolation of the same numerical
    # method, not an independently-derived limit, so it is reported here as
    # context rather than pinned as an assertion. See transition-notes.org.
    # [Claude Sonnet 5]
    Ms = 8.6e5
    mesh = _centred_box(10, 10.0)
    hx = np.array([_field_at_origin(mesh, (1, 0, 0), Ms, nx=n, ny=1)[0]
                   for n in (1, 3, 5, 9)])
    assert np.all(np.diff(np.abs(hx)) < 0.0)            # monotone decrease
    incr = np.abs(np.diff(hx))
    assert np.all(np.diff(incr) < 0.0)                  # increments shrink
    assert abs(hx[0] + 1.0 / 3.0) < 0.02               # nx=1 is the cube 1/3


def test_pbc_out_of_plane_thin_film_analytic_limit():
    # A thin film tiled periodically in-plane and magnetised out of plane has
    # demag factor Nz -> 1, i.e. <Hz> -> -Ms.  The periodic image sum reproduces
    # this analytic limit to high accuracy already at a modest tile count.
    Ms = 8.6e5
    mesh = _box(6, 6, 1, 40.0, 40.0, 2.0)
    e1 = abs(_pbc_field_avg(mesh, (0, 0, 1), Ms, nx=1, ny=1)[2] / Ms + 1.0)
    e3 = abs(_pbc_field_avg(mesh, (0, 0, 1), Ms, nx=3, ny=3)[2] / Ms + 1.0)
    assert e3 < e1              # tiling drives Nz toward the analytic 1
    assert e3 < 1e-3, e3       # converged to the thin-film out-of-plane limit


def test_pbc_translation_symmetry_of_ts():
    # The tile lattice is symmetric about the origin (contains +T and -T).
    mg = MacroGeometry(nx=5, ny=3, dx=7.0, dy=4.0)
    Ts = [tuple(t) for t in mg.compute_Ts(None)]
    for t in Ts:
        assert (-t[0], -t[1], -t[2]) in Ts


# --------------------------------------------------------------------------
# 6. Simulation.add end-to-end (round-1 review item): both ported demag
#    surfaces actually work through the public Simulation API, not just via
#    the FKDemag.setup()/compute_field() plumbing exercised above.
# --------------------------------------------------------------------------

def test_simulation_add_treecode_demag_runs_end_to_end():
    box = _box(4, 4, 4, 20.0, 20.0, 20.0)
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name="treecode_sim_add")
    sim.set_m((1.0, 0.2, -0.3))
    sim.add(Demag(solver="Treecode"))

    H = sim.effective_field()
    assert np.all(np.isfinite(H))
    assert np.max(np.abs(H)) > 0.0

    sim.run_until(1e-12)
    assert np.all(np.isfinite(sim.m))
    m_nodal = sim.m.reshape((3, -1))
    np.testing.assert_allclose(np.linalg.norm(m_nodal, axis=0), 1.0, atol=1e-5)


def test_simulation_add_macrogeometry_demag_runs_end_to_end():
    # Non-coincident tile spacing (dx=30 > the mesh's 20nm extent along x):
    # the coincident-node case (dx == extent) is ill-conditioned (see
    # transition-notes.org / the fk_demag_pbc module) and is deliberately
    # avoided here. [Claude Sonnet 5]
    box = _box(4, 4, 2, 20.0, 20.0, 4.0)
    sim = Simulation(box, 8.6e5, unit_length=1e-9, name="macrogeometry_sim_add")
    sim.set_m((0.0, 0.0, 1.0))
    sim.add(Demag(macrogeometry=MacroGeometry(nx=3, ny=1, dx=30.0)))

    H = sim.effective_field()
    assert np.all(np.isfinite(H))
    assert np.max(np.abs(H)) > 0.0

    sim.run_until(1e-12)
    assert np.all(np.isfinite(sim.m))
    m_nodal = sim.m.reshape((3, -1))
    np.testing.assert_allclose(np.linalg.norm(m_nodal, axis=0), 1.0, atol=1e-5)


# --------------------------------------------------------------------------
# 7. by-name deferrals unchanged
# --------------------------------------------------------------------------

def test_demag2d_still_deferred():
    with pytest.raises(NotImplementedError):
        Demag2D()


def test_gcr_still_deferred():
    with pytest.raises(NotImplementedError):
        Demag(solver="GCR")
