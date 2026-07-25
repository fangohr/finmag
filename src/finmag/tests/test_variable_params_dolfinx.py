"""Field-valued material parameters, spatially varying alpha, and regions (Task 16).

Validates the direct DOLFINx port's acceptance of spatially varying material
coefficients (``Ms``, ``A``, ``K1``/axis, ``D``, cubic ``K1``/``K2``/``K3``) and
Gilbert damping ``alpha``, and the region energy/magnetisation accounting, against:

- coordinate-ordered legacy oracle fixtures on 1D interval meshes
  (``variable_params_oracle.json``, ``spatially_varying_alpha_rhs.json``) whose
  1D connectivity is identical in dolfin and dolfinx, so the DG0-vs-CG1
  placement of each coefficient is pinned node-for-node;
- the legacy behavioural contracts ported from
  ``test_energy_creation_with_variable_Ms.py`` (Ms as number vs DG0 Function
  agree), ``test_spatially_varying_alpha.py`` (scalar -> uniform CG1 vector),
  ``test_spatially_varying_anisotropy.py`` (cos/sin easy axis) and
  ``test_energies_in_regions.py`` (region energies sum to the total);
- an explicit K2 native-typo DIVERGENCE PIN for spatially varying K2
  (``cubic_k2_varying_oracle.json``): the ported (correct) field differs from
  the legacy native field in exactly the ``hz`` component the ``energy.cc:116``
  ``K2[2]`` typo predicts, while the box-assembled energy matches;
- ``test_tier1_composed_physics`` (whole-branch review finding 3): every
  Tier 1 slice (Tasks 13-16) composed on one ``Simulation`` -- varying-A
  Exchange, varying-D DMI, constant-axis uniaxial anisotropy, an
  ``assemble=False`` cubic anisotropy with varying K2, an auto-connected
  ``OscillatingZeeman``, spatially varying ``alpha``, and ``mark_regions`` --
  asserting every interaction's field is nonzero, ``H_eff`` equals the sum of
  the parts, per-region energies sum to the total, a short ``run_until``
  advances with the oscillating field tracking ``cos(2*pi*f*t)``, and
  ``|m|`` stays 1.

By-name deferrals kept: legacy string Expressions (pass a callable) and
spatially varying cubic axes.

MASTER -> PORT MAPPING HEADER (SR1 P5.2 test-suite audit restoration). Every
master test function across the four ancestor files, and where it lands here:

``test_energy_creation_with_variable_Ms.py`` (``b5015c5a``):
  - test_can_create_energy_object[Exchange]           -> test_variable_ms_number_matches_dg0_function[factory0], atol 1e-12 (same)
  - test_can_create_energy_object[UniaxialAnisotropy]  -> test_variable_ms_number_matches_dg0_function[factory1], atol 1e-12 (same)
  - test_can_create_energy_object[Zeeman]              -> test_variable_ms_number_matches_dg0_function[factory2], atol 1e-12 (same)
  - test_can_create_energy_object[Demag]               -> test_variable_ms_number_matches_dg0_function[factory3] --
    RESTORED (audit finding: this parametrize case was dropped by the port;
    added back verbatim, atol 1e-12 unchanged, on the same box mesh/Field
    setup as the other three cases).

``test_spatially_varying_alpha.py`` (``b5015c5a``):
  - test_spatially_varying_alpha_using_Simulation_class -> test_scalar_alpha_via_simulation_fills_uniform_nodal_vector --
    RESTORED (audit finding: the port had swapped master's scalar alpha=1 /
    exact-array assertion for a varying-alpha / shape-only check, now
    test_spatially_varying_alpha_is_accepted_via_simulation below, kept as
    genuinely new coverage). Exact ``np.ones(11)`` assertion restored against
    ``sim.llg._alpha_field.as_array()`` (``sim.alpha`` itself returns a plain
    ``float`` for a constant value -- a DELIBERATE, already-documented
    divergence, see test_scalar_alpha_preserved_as_float -- so the array
    check targets the underlying per-node field master's
    ``sim.alpha.vector().array()`` exposed directly).
  - test_spatially_varying_alpha_using_LLG_class -> test_scalar_alpha_fills_uniform_nodal_vector,
    exact ``np.ones(11)`` (unchanged, already ported faithfully).

``test_spatially_varying_anisotropy.py`` (``b5015c5a``):
  - test_spatially_varying_anisotropy_axis -> test_spatially_varying_anisotropy_axis_relaxation_tracks_easy_axis --
    RESTORED (audit finding: the relaxation-tracking case was dropped by the
    port, which kept only the oracle energy/field comparison,
    test_oracle_spatially_varying_anisotropy, retained alongside it as
    covering different ground -- static field pin vs. dynamic relax()
    outcome). Tolerance ``diff.max() < 0.02`` restored VERBATIM; measured
    DOLFINx value ~0.0175, passes unchanged.

``test_energies_in_regions.py`` (``b5015c5a``):
  - test_energies_in_separated_subdomains -> COVERED-ELSEWHERE, same file:
    the additivity contract (``MultiDomainTest.check_energy_consistency``:
    sum of per-subdomain energies == whole-mesh energy) is re-tested via
    ``sim.mark_regions`` + ``sim.compute_energy(name, region=...)`` instead of
    master's ``pair_of_disks`` mesh / ``df.SubMesh`` / ``CellFunction``
    machinery, in test_region_energies_sum_to_total_zeeman_oracle (Zeeman,
    pinned against a legacy oracle), test_region_energies_sum_to_total_for_exchange
    (Exchange) and test_total_energy_over_region_sums_all_interactions
    (multi-interaction "total"). The ``SubMesh``/geometry-separation mechanism
    itself is NOT ported: ``get_submesh`` stays a documented
    ``NotImplementedError`` deferral (test_region_restricted_field_output_still_deferred_by_name),
    since ``mark_regions``/region ``dx`` measures replace it for every
    behavioural need exercised here.
  - test_energies_in_touching_subdomains -> NOT PORTED, deferred+reason:
    ``@pytest.mark.xfail`` IN MASTER ITSELF ("fails for touching subdomains
    for some reason... need to investigate"; never a validated behavioural
    contract even in legacy), so there is no passing legacy behaviour to
    restore.
  - MultiDomainTest (helper class, not a test) -> not a coverage target;
    its region/``dx(region)`` energy-splitting mechanic is what
    ``mark_regions``/``compute_energy(..., region=...)`` replaces above.

[Claude Opus 4.8]; composed-physics test [Claude Sonnet 5]; restoration of
dropped Demag/relaxation/exact-alpha coverage [Claude Sonnet 5]
"""

import json
import os

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

import finmag.util.consts as consts
from finmag.energies import (
    CubicAnisotropy, Demag, DMI, Exchange, OscillatingZeeman,
    UniaxialAnisotropy, Zeeman,
)
from finmag.energies.energy_base import mu0
from finmag.field import Field
from finmag.physics.effective_field import EffectiveField
from finmag.physics.llg import LLG
from finmag.sim.sim import Simulation

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")
VP = json.load(open(os.path.join(_FIX, "variable_params_oracle.json")))["cases"]
ALPHA_FIX = json.load(open(os.path.join(_FIX, "spatially_varying_alpha_rhs.json")))
K2_FIX = json.load(
    open(os.path.join(_FIX, "cubic_k2_varying_oracle.json")))["cases"]["k2_varying"]
MS_FIX = json.load(
    open(os.path.join(_FIX, "cubic_varying_ms_oracle.json")))["cases"]["ms_varying"]


# -- shared callables (vectorized, matching the oracle Expression strings) --

def _m_cossin_L8(x):
    n = x.shape[1]
    return np.vstack((0.6 * np.ones(n), 0.8 * np.cos(2 * np.pi * x[0] / 8.0),
                      0.8 * np.sin(2 * np.pi * x[0] / 8.0)))


def _m_linear_L8(x):
    # Globally linear (CG1-exact) vector field: over any cell interval the FEM
    # volume average equals the value at the midpoint, so region averages are
    # exact by hand. Region 1 = [0, 4] -> value at x=2; region 2 = [4, 8] -> x=6.
    n = x.shape[1]
    return np.vstack((x[0] / 8.0, 1.0 - x[0] / 8.0, 0.5 * np.ones(n)))


def _ms_var(x):
    return 8.6e5 * (1.0 + 0.3 * x[0] / 8.0)


def _a_var(x):
    return 1.3e-11 * (1.0 + 0.5 * x[0] / 8.0)


def _axis_var(x):
    n = x.shape[1]
    return np.vstack((np.cos(0.5 * x[0]), np.sin(0.5 * x[0]), np.zeros(n)))


def _interval8():
    return mesh.create_interval(MPI.COMM_WORLD, 8, [0.0, 8.0])


def _spaces(domain):
    return (fem.functionspace(domain, ("Lagrange", 1, (3,))),
            fem.functionspace(domain, ("DG", 0)))


def _quantity(case, name):
    for q in case["quantities"]:
        if q["name"] == name:
            return q
    raise KeyError(name)


def _scalar(case, name):
    for q in case["scalar_quantities"]:
        if q["name"] == name:
            return q
    raise KeyError(name)


def _blocked_to_xyz(blocked):
    """Component-blocked (``xxx``) flat -> node-interleaved (``xyz``) flat.

    Task 31: ``compute_field()`` now returns component-blocked order; this
    converts it to the per-node-interleaved layout ``_sorted_field`` expects.
    """
    return blocked.reshape((3, -1)).T.reshape(-1)


def _sorted_field(flat_xyz, S3):
    # Task 31: sort a node-interleaved (``xyz``) flat by the matching
    # owned-vertex coordinates (geometry order), so blocked ``compute_field``
    # (converted via _blocked_to_xyz) and ``get_ordered_numpy_array_xyz`` both
    # pair correctly with the coordinates.
    domain = S3.mesh
    n_owned = domain.geometry.index_map().size_local
    coords = domain.geometry.x[:n_owned, :3]
    H = flat_xyz.reshape(-1, 3)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order], H[order]


# --------------------------------------------------------------------------
# coefficient acceptance (constructs, sets up, computes finite)
# --------------------------------------------------------------------------

def _basic_fields():
    domain = _interval8()
    S3, DG = _spaces(domain)
    m = Field(S3, _m_cossin_L8, name="m")
    m.normalise()
    Ms = Field(DG, 8.6e5, name="Ms")
    return domain, S3, DG, m, Ms


@pytest.mark.parametrize("factory", [
    lambda cb: Exchange(cb),
    lambda cb: DMI(cb),
    lambda cb: UniaxialAnisotropy(cb, (0.0, 0.0, 1.0)),
    lambda cb: UniaxialAnisotropy(1.0e5, (0.0, 0.0, 1.0), K2=cb),
])
def test_scalar_coefficients_accept_callables_fields_functions(factory):
    domain, S3, DG, m, Ms = _basic_fields()
    # callable
    energy = factory(_a_var)
    energy.setup(m, Ms, unit_length=1e-9)
    assert np.isfinite(energy.compute_energy())
    assert np.all(np.isfinite(energy.compute_field()))
    # Field coefficient
    coeff_field = Field(DG, _a_var)
    energy2 = factory(coeff_field)
    energy2.setup(m, Ms, unit_length=1e-9)
    assert np.all(np.isfinite(energy2.compute_field()))
    # dolfinx Function coefficient
    fn = fem.Function(DG)
    fn.interpolate(lambda x: _a_var(x))
    energy3 = factory(fn)
    energy3.setup(m, Ms, unit_length=1e-9)
    assert np.all(np.isfinite(energy3.compute_field()))


def test_anisotropy_accepts_spatially_varying_axis():
    domain, S3, DG, m, Ms = _basic_fields()
    an = UniaxialAnisotropy(6.0e5, _axis_var)
    an.setup(m, Ms, unit_length=1e-9)
    assert np.all(np.isfinite(an.compute_field()))
    # a varying axis is used as given (not renormalised); this cos/sin axis is
    # already unit-norm, so the axis Field values are unit vectors.
    axis_vals = an.axis.as_array().reshape(-1, 3)
    assert np.allclose(np.linalg.norm(axis_vals, axis=1), 1.0, atol=1e-12)


def test_string_expression_coefficients_are_deferred_by_name():
    for factory in (Exchange, DMI):
        with pytest.raises(NotImplementedError, match="string Expression"):
            factory("x[0]")
    with pytest.raises(NotImplementedError, match="string Expression"):
        UniaxialAnisotropy("x[0]", (0.0, 0.0, 1.0))
    with pytest.raises(NotImplementedError, match="string Expression"):
        UniaxialAnisotropy(1.0, ("0", "0", "1"))


# --------------------------------------------------------------------------
# variable-Ms behavioural contract (test_energy_creation_with_variable_Ms)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda: Exchange(1.3e-11),
    lambda: UniaxialAnisotropy(1e5, (0.0, 0.0, 1.0)),
    lambda: Zeeman((0.0, 0.0, 1e6)),
    lambda: Demag(),
])
def test_variable_ms_number_matches_dg0_function(factory):
    """Ported from ``test_energy_creation_with_variable_Ms``: energy computed
    with Ms as a plain number equals energy with Ms as a DG0 Function.

    Master's ``test_can_create_energy_object`` parametrizes over ``Exchange``,
    ``UniaxialAnisotropy``, ``Zeeman`` AND ``Demag``; the port originally
    dropped the ``Demag`` case from this sweep (audit finding) -- restored
    here using the same box mesh/Field setup as the other three cases (the
    ``FKDemag`` BEM path accepts a plain ``dolfinx``-native box mesh, verified
    against a netgen-generated box in ``test_fk_demag_dolfinx.py``)."""
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (10e-9, 10e-9, 10e-9)],
        [5, 5, 5], mesh.CellType.tetrahedron)
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(domain, ("DG", 0))
    m = Field(S3, lambda x: np.vstack((1e-9 * np.ones(x.shape[1]),
                                       x[0] / 10.0, np.zeros(x.shape[1]))))
    m.normalise()

    e1 = factory()
    e1.setup(m, Field(DG, 8.6e5), unit_length=1e-9)
    ms_fn = fem.Function(DG)
    ms_fn.interpolate(lambda x: np.full(x.shape[1], 8.6e5))
    e2 = factory()
    e2.setup(m, Field(DG, ms_fn), unit_length=1e-9)
    assert abs(e1.compute_energy() - e2.compute_energy()) < 1e-12


# --------------------------------------------------------------------------
# oracle comparisons: energy + field for varying Ms / A / anisotropy axis
# --------------------------------------------------------------------------

def _compare_energy_field(case_name, energy_factory, coeff_ms):
    case = VP[case_name]
    domain = _interval8()
    S3, DG = _spaces(domain)
    m = Field(S3, _m_cossin_L8, name="m")
    Ms = coeff_ms(DG)
    energy = energy_factory()
    energy.setup(m, Ms, unit_length=1e-9)

    E = energy.compute_energy()
    coords_s, H_s = _sorted_field(_blocked_to_xyz(energy.compute_field()), S3)

    ref_coords = np.asarray(case["coordinates"]["values"])
    np.testing.assert_allclose(coords_s, ref_coords, rtol=0, atol=1e-9)

    # confirm the input m matches the oracle's recorded m
    m_q = _quantity(case, "m_vertex")
    _, m_s = _sorted_field(m.get_ordered_numpy_array_xyz(), S3)
    np.testing.assert_allclose(
        m_s, np.asarray(m_q["values"]),
        atol=m_q["tolerances"]["absolute"], rtol=m_q["tolerances"]["relative"])

    e_q = _scalar(case, "energy")
    np.testing.assert_allclose(
        E, e_q["value"], atol=e_q["tolerances"]["absolute"],
        rtol=e_q["tolerances"]["relative"])
    H_q = _quantity(case, "H_vertex")
    np.testing.assert_allclose(
        H_s, np.asarray(H_q["values"]),
        atol=H_q["tolerances"]["absolute"], rtol=H_q["tolerances"]["relative"])


def test_oracle_variable_ms_exchange():
    _compare_energy_field(
        "variable_Ms_exchange", lambda: Exchange(1.3e-11),
        lambda DG: Field(DG, _ms_var, name="Ms"))


def test_oracle_nonuniform_a_exchange():
    _compare_energy_field(
        "nonuniform_A_exchange", lambda: Exchange(_a_var),
        lambda DG: Field(DG, 8.6e5, name="Ms"))


def test_oracle_spatially_varying_anisotropy():
    _compare_energy_field(
        "spatially_varying_anisotropy",
        lambda: UniaxialAnisotropy(6.0e5, _axis_var),
        lambda DG: Field(DG, 8.6e5, name="Ms"))


def test_spatially_varying_anisotropy_axis_relaxation_tracks_easy_axis():
    """Ported from ``test_spatially_varying_anisotropy_axis``
    (``test_spatially_varying_anisotropy.py``): a spatially varying easy axis
    rotating from (0, 1, 0) at x=0 to (1, 0, 0) at x=Lx, relaxed from
    m=(1, 1, 0). Master's own comment notes the fit is imperfect near x=0;
    the aggregate ``diff.max() < 0.02`` tolerance is restored VERBATIM
    (audit finding -- this relaxation-tracking case was dropped by the port,
    which kept only the oracle energy/field comparison above).

    dolfin -> dolfinx: ``df.Expression`` -> vectorized callable; probing a
    point uses ``Field.__call__`` (``a(x)``, ``sim.m_field(x)``), which
    restores the same point-in-cell mechanic dolfin's ``Function.__call__``
    gave master for free (see ``finmag.field.evaluate_at_point``)."""
    Ms = 1e6
    A = 1.3e-11
    K1 = 6e5
    lb = consts.bloch_parameter(A, K1)

    unit_length = 1e-9
    nx = 20
    Lx = nx * lb / unit_length
    domain = mesh.create_interval(MPI.COMM_WORLD, nx, [0.0, Lx])

    # anisotropy axis goes from (0, 1, 0) at x=0 to (1, 0, 0) at x=Lx
    def _axis_expr(x):
        n = x.shape[1]
        denom = np.sqrt(x[0] ** 2 + (Lx - x[0]) ** 2)
        return np.vstack((x[0] / denom, (Lx - x[0]) / denom, np.zeros(n)))

    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    a = Field(S3, _axis_expr)

    sim = Simulation(domain, Ms, unit_length=unit_length, name="sva_relax")
    sim.set_m((1.0, 1.0, 0.0))
    sim.add(UniaxialAnisotropy(K1, a))
    sim.relax()

    # probe the easy axis and the magnetisation along the interval
    points = 100
    xs = np.linspace(0, Lx, points)
    axis_xs = np.zeros((points, 3))
    m_xs = np.zeros((points, 3))
    for i, x in enumerate(xs):
        axis_xs[i] = a(x)
        m_xs[i] = sim.m_field(x)

    # we want the magnetisation to follow the easy axis; measured max diff
    # ~0.0175 (master tolerance 0.02, passes verbatim).
    diff = np.abs(m_xs - axis_xs)
    assert diff.max() < 0.02


# --------------------------------------------------------------------------
# spatially varying alpha
# --------------------------------------------------------------------------

def _alpha_spaces(domain):
    return (fem.functionspace(domain, ("Lagrange", 1)),
            fem.functionspace(domain, ("Lagrange", 1, (3,))))


def test_scalar_alpha_preserved_as_float():
    domain = _interval8()
    S1, S3 = _alpha_spaces(domain)
    llg = LLG(S1, S3)
    llg.set_alpha(0.3)
    assert isinstance(llg.alpha, float)
    assert llg.alpha == 0.3


def test_scalar_alpha_fills_uniform_nodal_vector():
    """Ported from ``test_spatially_varying_alpha``: a scalar sets every nodal
    alpha value (uniform CG1 vector)."""
    domain = mesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 20.0])
    S1, S3 = _alpha_spaces(domain)
    llg = LLG(S1, S3)
    llg.set_alpha(1)
    assert np.array_equal(llg._alpha_field.as_array(), np.ones(11))


def test_scalar_alpha_via_simulation_fills_uniform_nodal_vector():
    """Ported from ``test_spatially_varying_alpha_using_Simulation_class``:
    scalar alpha=1 set via ``sim.alpha`` fills a uniform per-node array of
    ones (master asserted ``sim.alpha.vector().array() == np.ones(11)``
    exactly).

    Restore note (audit finding): the port had swapped this scalar/exact-array
    case for a varying-alpha/shape-only check
    (``test_spatially_varying_alpha_is_accepted_via_simulation`` below, kept
    as genuinely new coverage). ``sim.alpha`` itself returns a plain ``float``
    for a constant value here (the documented divergence pinned by
    ``test_scalar_alpha_preserved_as_float``: legacy's ``df.Function`` had no
    such fast path), so master's exact-array assertion is restored against
    the underlying per-node field (``sim.llg._alpha_field.as_array()``) that
    ``sim.alpha`` would collapse to a float from -- same mesh/parameters as
    master (``length=20``, ``simplices=10`` -> 11 nodes)."""
    domain = mesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 20.0])
    sim = Simulation(domain, 1.0, unit_length=1e-9, name="varalpha_scalar")
    sim.alpha = 1
    assert np.array_equal(sim.llg._alpha_field.as_array(), np.ones(11))


def test_spatially_varying_alpha_is_accepted_via_simulation():
    domain = mesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 20.0])
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="varalpha")
    sim.set_m((1.0, 0.0, 0.0))
    sim.alpha = lambda x: 0.1 + 0.05 * x[0]
    assert isinstance(sim.alpha, np.ndarray)
    assert sim.alpha.shape == (11,)


def test_oracle_spatially_varying_alpha_rhs():
    params = ALPHA_FIX["physical_parameters"]
    cells = ALPHA_FIX["mesh"]["parameters"]["cells"]
    x1 = ALPHA_FIX["mesh"]["parameters"]["x1"]
    domain = mesh.create_interval(MPI.COMM_WORLD, cells, [0.0, x1])
    S1, S3 = _alpha_spaces(domain)
    llg = LLG(S1, S3, do_precession=params["do_precession"]["value"],
              unit_length=params["unit_length"]["value"])
    llg.Ms = params["Ms"]["value"]
    llg.set_alpha(lambda x: 0.05 + 0.1 * x[0] / 4.0)
    llg.set_m(
        lambda x: np.vstack((np.cos(0.4 * x[0]), np.sin(0.4 * x[0]),
                             0.5 * np.ones(x.shape[1]))),
        normalise=True)
    llg.effective_field.add(Exchange(params["A"]["value"], name="Exchange"))
    llg.effective_field.add(
        Zeeman(tuple(params["H_zeeman"]["value"]), name="Zeeman"))

    dmdt = llg.solve(0.0)

    coords = S3.tabulate_dof_coordinates()
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))

    # alpha placement
    a_q = _quantity(ALPHA_FIX, "alpha")
    alpha_sorted = llg._alpha_field.get_ordered_numpy_array()
    a_coords = S1.tabulate_dof_coordinates()
    a_order = np.lexsort((a_coords[:, 2], a_coords[:, 1], a_coords[:, 0]))
    np.testing.assert_allclose(
        alpha_sorted[a_order], np.asarray(a_q["values"]),
        atol=a_q["tolerances"]["absolute"], rtol=a_q["tolerances"]["relative"])

    dmdt_q = _quantity(ALPHA_FIX, "dmdt")
    dmdt_sorted = dmdt.reshape(3, -1).T[order]
    np.testing.assert_allclose(
        dmdt_sorted, np.asarray(dmdt_q["values"]),
        atol=dmdt_q["tolerances"]["absolute"],
        rtol=dmdt_q["tolerances"]["relative"])


def test_per_node_alpha_enters_gamma_LL_and_damping():
    """A macrospin-like uniform state on a mesh with two-valued alpha: the two
    nodes evolve with distinct gamma_LL = gamma/(1+alpha^2), proving alpha is
    per-node in BOTH precession and damping (not a scalar)."""
    domain = mesh.create_interval(MPI.COMM_WORLD, 1, [0.0, 1.0])
    S1, S3 = _alpha_spaces(domain)
    llg = LLG(S1, S3)
    llg.Ms = 8.6e5
    # alpha 0.1 at x=0, 0.5 at x=1 (raw nodal set, matched to coordinates)
    coords = S1.tabulate_dof_coordinates()
    alpha_vals = np.where(coords[:, 0] < 0.5, 0.1, 0.5)
    llg._alpha_field.from_array(alpha_vals.astype(np.float64))
    llg._alpha_node = llg._alpha_field.get_ordered_numpy_array()
    llg.set_m((1.0, 0.0, 0.0))
    llg.effective_field.add(Zeeman((0.0, 0.0, 1.0e5)))
    dmdt = llg.solve(0.0).reshape(3, -1)
    # dm_z from damping = -alpha gamma_LL (m x (m x H))_z; for m=x_hat, H=z_hat
    # (m x (m x H)) = -H_perp so dm_z(damping) = alpha*gamma_LL*Hz. Distinct per node.
    node_alpha = llg._alpha_node
    gamma_LL = consts.gamma / (1.0 + node_alpha ** 2)
    expected_dmz = node_alpha * gamma_LL * 1.0e5
    np.testing.assert_allclose(dmdt[2], expected_dmz, rtol=1e-10)
    assert not np.isclose(expected_dmz[0], expected_dmz[1])


# --------------------------------------------------------------------------
# cubic K2 native-typo DIVERGENCE PIN (deliberate deviation, now LIVE)
# --------------------------------------------------------------------------

def _k2_oracle_setup():
    physical = K2_FIX["physical_parameters"]
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (0.7, 0.7, 0.7)], [4, 4, 4],
        mesh.CellType.tetrahedron)
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    DG = fem.functionspace(domain, ("DG", 0))
    m = Field(S3, lambda x: np.vstack(
        (0.6 * np.ones(x.shape[1]), 0.8 * np.cos(2 * np.pi * x[0]),
         0.8 * np.sin(2 * np.pi * x[0]))), name="m")
    Ms = Field(DG, physical["Ms"]["value"], name="Ms")
    ca = CubicAnisotropy(
        physical["u1"]["value"], physical["u2"]["value"], K1=0,
        K2=lambda x: -1.3e7 * (1.0 + 0.4 * x[0] / 0.7), K3=0)
    ca.setup(m, Ms, unit_length=physical["unit_length"]["value"])
    # Task 31: order by owned-vertex coords so blocked compute_field() rows
    # (reshape((3, -1)).T) sort consistently.
    domain = S3.mesh
    n_owned = domain.geometry.index_map().size_local
    coords = domain.geometry.x[:n_owned, :3]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return ca, S3, order


def test_k2_varying_port_matches_correct_field_and_energy():
    """The ported (correct) per-node field reproduces the fixture's
    independently-derived correct field, and the box-assembled energy (which
    never uses the native field path) matches the legacy energy."""
    ca, S3, order = _k2_oracle_setup()
    H_port = ca.compute_field().reshape((3, -1)).T[order]
    H_correct = np.asarray(_quantity(K2_FIX, "H_correct")["values"])
    scale = np.abs(H_correct).max()
    np.testing.assert_allclose(H_port, H_correct, rtol=1e-9, atol=1e-9 * scale)

    e_q = _scalar(K2_FIX, "energy")
    np.testing.assert_allclose(
        ca.compute_energy(), e_q["value"], rtol=e_q["tolerances"]["relative"],
        atol=1e-25)


def test_k2_varying_diverges_from_legacy_native_in_hz_only():
    """DIVERGENCE PIN (DELIBERATE DEVIATION, USER ACCEPTANCE PENDING): for
    spatially varying K2 the legacy native routine's ``hz`` uses ``K2`` at the
    fixed native array index 2 (the ``energy.cc:116`` ``K2[2]`` typo), so the
    ported *correct* per-node field agrees with the legacy native ``hx``/``hy``
    to floating-point noise but its ``hz`` diverges by exactly
    ``-2/(mu0 Ms)(K2[v*] - K2[i]) termz[i]``. Predicted: ``hx``/``hy`` identical,
    ``hz`` differs, and the implied per-node K2 that reproduces the legacy
    ``hz`` is a single constant ``K2[v*]``."""
    ca, S3, order = _k2_oracle_setup()
    H_port = ca.compute_field().reshape((3, -1)).T[order]
    H_legacy = np.asarray(_quantity(K2_FIX, "H_vertex")["values"])
    H_correct = np.asarray(_quantity(K2_FIX, "H_correct")["values"])
    K2_nodal = np.asarray(_quantity(K2_FIX, "K2_nodal")["values"])
    scale = np.abs(H_legacy).max()

    # hx, hy: correct == legacy native to noise
    np.testing.assert_allclose(H_port[:, 0], H_legacy[:, 0], atol=1e-6 * scale)
    np.testing.assert_allclose(H_port[:, 1], H_legacy[:, 1], atol=1e-6 * scale)

    # hz: genuinely diverges (typo is LIVE for varying K2)
    hz_gap = np.abs(H_port[:, 2] - H_legacy[:, 2]).max()
    assert hz_gap > 1e-3 * scale, hz_gap

    # the legacy hz corresponds to a single fixed-index K2[v*]
    m_hz = np.abs(H_correct[:, 2]) > 1e-6 * scale
    implied_K2 = H_legacy[m_hz, 2] / H_correct[m_hz, 2] * K2_nodal[m_hz]
    assert implied_K2.std() < 1e-3 * abs(implied_K2.mean()), implied_K2
    K2_star = implied_K2.mean()
    # v* is a genuine node whose K2 is used everywhere (K2 varies -> divergence)
    assert K2_nodal.min() <= K2_star <= K2_nodal.max()
    assert K2_nodal.max() - K2_nodal.min() > 1e5  # K2 really does vary


# --------------------------------------------------------------------------
# cubic assemble=False: per-node Ms (fix round 1, Finding 1)
# --------------------------------------------------------------------------

def _cubic_varying_ms_setup():
    """Ms on the CG1 scalar space matching m's nodes -- the placement
    legacy's own native routine requires (see
    ``CubicAnisotropy._ms_per_node``); K2=0 sidesteps the documented
    ``energy.cc:116`` ``K2[2]`` typo so this isolates Ms handling."""
    physical = MS_FIX["physical_parameters"]
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (0.7, 0.7, 0.7)], [4, 4, 4],
        mesh.CellType.tetrahedron)
    S3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    S1 = fem.functionspace(domain, ("Lagrange", 1))
    m = Field(S3, lambda x: np.vstack(
        (0.6 * np.ones(x.shape[1]), 0.8 * np.cos(2 * np.pi * x[0]),
         0.8 * np.sin(2 * np.pi * x[0]))), name="m")
    Ms = Field(S1, lambda x: 876626.0 * (1.0 + 0.35 * x[0] / 0.7), name="Ms")
    ca = CubicAnisotropy(
        physical["u1"]["value"], physical["u2"]["value"],
        K1=physical["K1"]["value"], K2=physical["K2"]["value"],
        K3=physical["K3"]["value"])  # assemble=False default
    ca.setup(m, Ms, unit_length=physical["unit_length"]["value"])
    # Task 31: order by owned-vertex coords (matches blocked compute_field rows).
    domain = S3.mesh
    n_owned = domain.geometry.index_map().size_local
    coords = domain.geometry.x[:n_owned, :3]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return ca, S3, order


def test_cubic_varying_ms_matches_native_oracle():
    """Quantitative pin (Finding 1, fix round 1): the ported native analytic
    field for a spatially varying Ms must reproduce the legacy native
    ``compute_cubic_field`` output node-for-node, at the same tolerance as
    the other native-oracle per-term cases."""
    ca, S3, order = _cubic_varying_ms_setup()
    H_port = ca.compute_field().reshape((3, -1)).T[order]
    H_q = _quantity(MS_FIX, "H_vertex")
    ref_H = np.asarray(H_q["values"])
    np.testing.assert_allclose(
        H_port, ref_H, atol=H_q["tolerances"]["absolute"],
        rtol=H_q["tolerances"]["relative"])

    e_q = _scalar(MS_FIX, "energy")
    np.testing.assert_allclose(
        ca.compute_energy(), e_q["value"], rtol=e_q["tolerances"]["relative"],
        atol=1e-25)


def test_cubic_varying_ms_field_is_finite_and_nonzero():
    ca, _, _ = _cubic_varying_ms_setup()
    H = ca.compute_field()
    assert np.all(np.isfinite(H))
    assert np.max(np.abs(H)) > 0.0


def test_cubic_varying_ms_participates_in_dynamics():
    """Finding 1(a): a spatially varying Ms with the legacy-default
    ``assemble=False`` constructor must actually drive the LLG right-hand
    side, not merely compute a finite field in isolation. Ms lives on the
    CG1 space matching ``m`` (``Simulation``/``LLG`` always place ``Ms`` in
    DG0, which ``CubicAnisotropy._ms_per_node`` documents as unsupported for
    a *varying* Ms -- see
    ``test_cubic_varying_ms_on_dg0_space_raises_documented_error`` below), so
    the ``EffectiveField`` registry is built directly against ``LLG``'s
    internals rather than through ``Simulation``."""
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [2, 2, 2],
        mesh.CellType.tetrahedron)
    S1, S3 = _alpha_spaces(domain)
    llg = LLG(S1, S3, unit_length=1e-9)
    m0 = np.array((1.0, 0.3, 0.1))
    m0 /= np.linalg.norm(m0)
    llg.set_m(tuple(m0))
    llg.set_alpha(0.5)

    Ms_varying = Field(S1, lambda x: 8.6e5 * (1.0 + 0.2 * x[0] / 5.0), name="Ms")
    llg.effective_field = EffectiveField(llg._m_field, Ms_varying, 1e-9)
    llg.effective_field.add(
        CubicAnisotropy((1, 0, 0), (0, 1, 0), 1.0e4))  # assemble=False default

    dmdt = llg.solve(0.0)
    assert np.all(np.isfinite(dmdt))
    assert np.max(np.abs(dmdt)) > 0.0


def test_cubic_varying_ms_on_dg0_space_raises_documented_error():
    """A varying Ms on a space that does not align node-for-node with m (e.g.
    the DG0 space ``Simulation``/``LLG`` always use for Ms) cannot be used by
    the native analytic path -- exactly as legacy's own native routine
    requires (``Ms_arr.check_shape(nodes, ...)``; verified empirically
    against the oracle: legacy raises ``ValueError: compute_cubic_field: Ms:
    Expected array of shape (nodes), got (cells)`` in this scenario). The
    port raises a clear, documented error instead of silently misindexing or
    surfacing a confusing NumPy broadcast failure."""
    box = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [2, 2, 2],
        mesh.CellType.tetrahedron)
    sim = Simulation(
        box, lambda x: 8.6e5 * (1.0 + 0.1 * x[0] / 5.0), unit_length=1e-9,
        name="cubic_ms_dg0_mismatch")
    sim.set_m((0.3, 0.4, np.sqrt(1 - 0.09 - 0.16)))
    sim.add(CubicAnisotropy((1, 0, 0), (0, 1, 0), 1.0e4))  # assemble=False default
    with pytest.raises(ValueError, match="per-node"):
        sim.effective_field()


# --------------------------------------------------------------------------
# regions: mark_regions + per-region energy / m_average
# --------------------------------------------------------------------------

def _region_sim():
    domain = _interval8()
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="regions")
    sim.set_m(_m_cossin_L8, normalise=True)
    return sim


def test_mark_regions_builds_contiguous_ids_and_markers():
    sim = _region_sim()
    ids = sim.mark_regions(lambda pt: "left" if pt[0] < 4.0 else "right")
    assert set(ids) == {"left", "right"}
    assert sorted(ids.values()) == [0, 1]
    assert sim.region_markers.values.min() >= 0


def test_region_energies_sum_to_total_zeeman_oracle():
    """Ported ``test_energies_in_regions`` additivity, pinned against the legacy
    two-region Zeeman oracle."""
    case = VP["two_region_zeeman"]
    sim = _region_sim()
    sim.add(Zeeman((1e6, 0.0, 0.0)))
    sim.mark_regions(lambda pt: 1 if pt[0] < 4.0 else 2)

    E_total = sim.compute_energy("Zeeman")
    E1 = sim.compute_energy("Zeeman", region=1)
    E2 = sim.compute_energy("Zeeman", region=2)

    np.testing.assert_allclose(E1 + E2, E_total, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(
        E_total, _scalar(case, "energy_total")["value"], rtol=1e-9)
    np.testing.assert_allclose(
        E1, _scalar(case, "energy_region_1")["value"], rtol=1e-9)
    np.testing.assert_allclose(
        E2, _scalar(case, "energy_region_2")["value"], rtol=1e-9)


def test_region_energies_sum_to_total_for_exchange():
    """Additivity holds for a box-energy too (EnergyBase.compute_energy(dx))."""
    sim = _region_sim()
    sim.add(Exchange(1.3e-11))
    sim.mark_regions(lambda pt: 1 if pt[0] < 4.0 else 2)
    total = sim.compute_energy("Exchange")
    parts = sim.compute_energy("Exchange", region=1) \
        + sim.compute_energy("Exchange", region=2)
    np.testing.assert_allclose(parts, total, rtol=1e-12, atol=1e-18)


def test_total_energy_over_region_sums_all_interactions():
    sim = _region_sim()
    sim.add(Exchange(1.3e-11))
    sim.add(Zeeman((1e6, 0.0, 0.0)))
    sim.mark_regions(lambda pt: 1 if pt[0] < 4.0 else 2)
    tot = sim.compute_energy("total", region=1) \
        + sim.compute_energy("total", region=2)
    whole = sim.compute_energy("Exchange") + sim.compute_energy("Zeeman")
    np.testing.assert_allclose(tot, whole, rtol=1e-12, atol=1e-18)


# --------------------------------------------------------------------------
# Tier 1 whole-branch review, finding 3: composed-physics test. No existing
# test exercises every Tier 1 slice (Tasks 13-16) composed on one Simulation;
# this mirrors the experiment the whole-branch reviewer ran by hand. Kept in
# this file (not a new file) since it is a direct extension of the Task 16
# variable-parameters/regions coverage above, and is wired into the same
# ``dolfinx-src-varparams-pytest`` gate as every other test here.
# --------------------------------------------------------------------------

def _composed_physics_sim():
    domain = mesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)], [3, 3, 3],
        mesh.CellType.tetrahedron)
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="tier1_composed")

    def m0(x):
        return np.vstack((
            0.6 * np.ones(x.shape[1]),
            0.8 * np.cos(2.0 * np.pi * x[0] / 5.0),
            0.8 * np.sin(2.0 * np.pi * x[0] / 5.0),
        ))

    sim.set_m(m0, normalise=True)
    sim.alpha = lambda x: 0.3 + 0.2 * x[0] / 5.0  # spatially varying alpha

    sim.add(Exchange(lambda x: 1.3e-11 * (1.0 + 0.3 * x[0] / 5.0)))  # varying A
    sim.add(DMI(lambda x: 1.0e-3 * (1.0 + 0.2 * x[0] / 5.0)))  # varying D
    sim.add(UniaxialAnisotropy(1.0e4, (0.0, 0.0, 1.0)))  # constant axis
    sim.add(CubicAnisotropy(  # assemble=False (default) with varying K2
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), K1=1.0e3,
        K2=lambda x: 5.0e3 * (1.0 + 0.3 * x[0] / 5.0), K3=1.0e2,
        assemble=False))
    osc = OscillatingZeeman(H0=(0.0, 0.0, 5.0e4), freq=1.0e8, phase=0.0, t_off=None)
    sim.add(osc)  # no explicit with_time_update: exercises the Task 6 auto-connect

    sim.mark_regions(lambda pt: 1 if pt[0] < 2.5 else 2)
    return sim, osc


def test_tier1_composed_physics():
    """Every Tier 1 slice composed on one Simulation: varying-A Exchange,
    varying-D DMI, constant-axis uniaxial anisotropy, ``assemble=False``
    cubic anisotropy with varying K2, an auto-connected ``OscillatingZeeman``,
    spatially varying ``alpha``, and ``mark_regions``."""
    sim, osc = _composed_physics_sim()

    # each interaction's field is finite and genuinely nonzero.
    for name in sim.interactions():
        H = sim.get_interaction(name).compute_field()
        assert np.all(np.isfinite(H))
        assert np.max(np.abs(H)) > 0.0, name

    # H_eff equals the sum of the parts (t=0.0, needed for the auto-connected
    # OscillatingZeeman).
    H_eff = sim.llg.effective_field.compute(t=0.0)
    parts_sum = sum(
        sim.get_interaction(name).compute_field() for name in sim.interactions())
    np.testing.assert_allclose(H_eff, parts_sum, rtol=1e-10, atol=1e-10)

    # per-region energies sum to the total.
    total = sim.compute_energy("total")
    region_sum = (
        sim.compute_energy("total", region=1)
        + sim.compute_energy("total", region=2))
    np.testing.assert_allclose(region_sum, total, rtol=1e-11, atol=1e-18)

    # a short run_until advances, with the oscillating field tracking
    # cos(2*pi*f*t), and |m| stays 1.
    freq = 1.0e8
    t_end = 2.0e-10
    sim.run_until(t_end)

    expected_scale = np.cos(2.0 * np.pi * freq * sim.t)
    expected = np.array([0.0, 0.0, 5.0e4]) * expected_scale
    H_osc = osc.compute_field().reshape((3, -1)).T
    np.testing.assert_allclose(
        H_osc, np.broadcast_to(expected, H_osc.shape), atol=1.0, rtol=1e-6)

    m_xxx = sim.m.reshape(3, -1)
    norms = np.sqrt(np.sum(m_xxx ** 2, axis=0))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_region_m_average_is_restricted():
    sim = _region_sim()
    sim.mark_regions(lambda pt: 1 if pt[0] < 4.0 else 2)
    avg1 = sim.m_average_in_region(1)
    assert avg1.shape == (3,)
    assert np.all(np.isfinite(avg1))


def test_region_measure_unknown_region_raises_by_name():
    sim = _region_sim()
    sim.mark_regions(lambda pt: 1 if pt[0] < 4.0 else 2)
    with pytest.raises(KeyError, match="unknown region"):
        sim.region_measure(99)


def test_region_measure_before_mark_regions_raises():
    sim = _region_sim()
    with pytest.raises(RuntimeError, match="mark_regions"):
        sim.region_measure(1)


def test_save_m_in_region_registers_ndt_column_with_region_average(tmp_path):
    """SR1 P4-region: faithful port of legacy ``save_m_in_region``
    (``b5015c5a:src/finmag/sim/sim.py``). It does NOT write a field to file; it
    registers a per-region ``<m>`` column in the .ndt table whose value is the
    volume-averaged magnetisation over the region (legacy set
    ``tablewriter.entities[name]`` to ``m_average_fun(dx=self.dx(region_id))``).

    m is globally linear, so the FEM average over a region equals the value at
    that region's midpoint (exact by hand)."""
    from finmag.util.fileio import Tablereader
    domain = _interval8()
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="regions_ndt")
    sim.set_m(_m_linear_L8, normalise=False)
    sim.mark_regions(lambda pt: 1 if pt[0] < 4.0 else 2)
    sim.ndtfilename = str(tmp_path / "regions.ndt")

    sim.save_m_in_region(1, name="left")
    sim.save_m_in_region(2, name="right")
    sim.save_averages()

    reader = Tablereader(sim.ndtfilename)
    left = np.array(
        [reader["left_m_x"][0], reader["left_m_y"][0], reader["left_m_z"][0]])
    right = np.array(
        [reader["right_m_x"][0], reader["right_m_y"][0], reader["right_m_z"][0]])
    # Region 1 = [0, 4], midpoint x=2 -> (2/8, 1-2/8, 0.5) = (0.25, 0.75, 0.5).
    np.testing.assert_allclose(left, [0.25, 0.75, 0.5], atol=1e-12)
    # Region 2 = [4, 8], midpoint x=6 -> (6/8, 1-6/8, 0.5) = (0.75, 0.25, 0.5).
    np.testing.assert_allclose(right, [0.75, 0.25, 0.5], atol=1e-12)
    # The column reproduces the ported, tested region-average computation.
    np.testing.assert_allclose(left, sim.m_average_in_region(1), atol=1e-14)
    np.testing.assert_allclose(right, sim.m_average_in_region(2), atol=1e-14)


def test_save_m_in_region_two_regions_do_not_mix(tmp_path):
    """Two regions carrying DIFFERENT magnetisation: each region's column must
    reflect ONLY its own cells (region-mixup guard). Also pins the legacy
    default column name ``region_<internal_id>``."""
    from finmag.util.fileio import Tablereader
    domain = _interval8()
    sim = Simulation(domain, 8.6e5, unit_length=1e-9, name="regions_mix")
    sim.set_m(_m_linear_L8, normalise=False)
    ids = sim.mark_regions(lambda pt: 1 if pt[0] < 4.0 else 2)
    sim.ndtfilename = str(tmp_path / "regions_mix.ndt")

    sim.save_m_in_region(1)  # default name region_<region_ids[1]>
    sim.save_m_in_region(2)
    sim.save_averages()

    reader = Tablereader(sim.ndtfilename)
    n1 = "region_{}".format(ids[1])
    n2 = "region_{}".format(ids[2])
    left = np.array(
        [reader[n1 + "_m_x"][0], reader[n1 + "_m_y"][0], reader[n1 + "_m_z"][0]])
    right = np.array(
        [reader[n2 + "_m_x"][0], reader[n2 + "_m_y"][0], reader[n2 + "_m_z"][0]])
    np.testing.assert_allclose(left, sim.m_average_in_region(1), atol=1e-14)
    np.testing.assert_allclose(right, sim.m_average_in_region(2), atol=1e-14)
    np.testing.assert_allclose(left, [0.25, 0.75, 0.5], atol=1e-12)
    np.testing.assert_allclose(right, [0.75, 0.25, 0.5], atol=1e-12)
    # Region magnetisations genuinely differ -> a mixup would be caught.
    assert not np.allclose(left, right)


def test_region_restricted_field_output_still_deferred_by_name():
    """``save_m_in_region`` is now ported (per-region .ndt column; see
    ``test_save_m_in_region_*``). The region-restricted *field / submesh* paths
    (``get_submesh``, ``get_field_as_dolfin_function(region=...)``) stay deferred
    by name: legacy gated them behind ``mark_regions``, which itself raised on
    modern dolfin (>= 1.5), so they were never functional there."""
    sim = _region_sim()
    sim.mark_regions(lambda pt: 0)
    sim.save_m_in_region(0)  # registers an ndt column; no longer raises
    with pytest.raises(NotImplementedError, match="get_submesh"):
        sim.get_submesh(0)
    with pytest.raises(NotImplementedError):
        sim.get_field_as_dolfin_function("m", region=0)


# --------------------------------------------------------------------------
# spatially varying cubic axes (SR1 P3.5): accepted (was by-name deferral)
# --------------------------------------------------------------------------

def test_spatially_varying_cubic_axis_is_accepted():
    """SR1 P3.5: spatially varying cubic axes (callable/Field/Function) are now
    supported, lifting the historical by-name deferral. This is a greenfield
    extension: legacy ``CubicAnisotropy.__init__`` forms ``u3 = np.cross(u1,
    u2)`` from the raw axes and crashes on any varying axis, so there is no
    legacy oracle -- the axis path is pinned by the W1 constant-reduction and W2
    per-region witnesses in ``test_cubic_anisotropy_dolfinx.py``. Spatially
    varying cubic K's remain supported here too."""
    domain = _interval8()
    S3, DG = _spaces(domain)
    m = Field(S3, _m_cossin_L8); m.normalise()
    Ms = Field(DG, 8.6e5)

    # callable u1 axis (spatially varying) -- accepted, finite field.
    ca = CubicAnisotropy(_axis_var, (0.0, 1.0, 0.0), K1=1.0e4)
    ca.setup(m, Ms, unit_length=1e-9)
    assert np.all(np.isfinite(ca.compute_field()))

    # callable u2 axis (spatially varying) -- accepted, finite field.
    ca2 = CubicAnisotropy((1.0, 0.0, 0.0), _axis_var, K1=1.0e4)
    ca2.setup(m, Field(DG, 8.6e5), unit_length=1e-9)
    assert np.all(np.isfinite(ca2.compute_field()))

    # spatially varying cubic K's remain supported (unchanged behaviour).
    ca3 = CubicAnisotropy((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                          K1=lambda x: 1.0e4 * (1.0 + x[0]))
    ca3.setup(m, Field(DG, 8.6e5), unit_length=1e-9)
    assert np.all(np.isfinite(ca3.compute_field()))
