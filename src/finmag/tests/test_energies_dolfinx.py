"""DOLFINx energy-interaction tests: Exchange, Zeeman (static), UniaxialAnisotropy.

BUCKET-B aggregate port. This file does not literally transcribe any single
master file; it AGGREGATES coverage from five master ancestors (all at git
``b5015c5a``) under renamed, restructured tests, plus one master test
restored verbatim under its master name (see RESTORE below). The table
below is the MASTER -> PORT MAPPING HEADER required by the SR1 P5.2
bucket-B convention (exemplars: ``git show 02e2e3d8`` for the accounting
style, ``test_fk_demag_dolfinx.py`` for the minimal-diff-transcription +
NEW-under-DOLFINx split). Every master function across all five ancestors
is listed exactly once, with its fate: a covering port function, an
explicit "not covered", a "deferred + reason", or "covered elsewhere + file".

1) ``src/finmag/energies/exchange_test.py`` (6 functions)

  - ``test_interaction_accepts_name`` -> covered-elsewhere (weaker):
    ``test_exchange_uniform_field_is_exactly_zero`` asserts the *default*
    ``exchange.name == "Exchange"``; the custom-name kwarg is exercised for
    ``Zeeman`` in this file (``zeeman.name == "Applied"``, same
    ``EnergyBase`` code path) but not re-asserted for ``Exchange`` with a
    custom string. Not independently fixed (out of this audit's named-restore
    scope; low risk, shared implementation).
  - ``test_there_should_be_no_exchange_for_uniform_m`` -> covered:
    ``test_exchange_uniform_field_is_exactly_zero``. Tolerance TIGHTENED,
    not loosened: master ``FIELD_TOLERANCE=6e-7`` / ``ENERGY_TOLERANCE=0.0``
    -> port asserts field ``atol=1e-12`` and energy ``abs=0.0`` exactly
    (measured: both are exact/near-exact zero for box-assemble on a uniform
    field, so the tighter bound still passes).
  - ``test_exchange_energy_analytical`` -> covered:
    ``test_exchange_linear_energy_and_physical_length_scaling``. Different
    ``m`` profile (linear ``(x, y, 0)`` vs. master's ``(x, z, -y)``) and
    generalised across 2D/3D + explicit ``unit_length``. Tolerance TIGHTENED:
    master ``rel=1e-7`` -> port ``rel=1e-13`` (measured comparably tight
    since the port uses an exactly-linear field on an affine mesh).
  - ``test_exchange_energy_analytical_2`` -> covered-elsewhere:
    ``test_exchange_linear_energy_and_physical_length_scaling`` (same
    property -- analytic linear-energy formula scaling with ``A`` and
    ``unit_length`` -- different construction: master used a fine
    ``BoxMesh`` + trig ``m`` at ``rel=5e-5``; port uses an exactly-linear
    ``m`` at ``rel=1e-13``).
  - ``test_exchange_field_supported_methods`` -> deferred + reason:
    covered by ``test_deferred_energy_methods_are_rejected_precisely``.
    Alternate methods (``box-matrix-numpy``, ``box-matrix-petsc``,
    ``direct``) are NOT ported under DOLFINx and raise
    ``NotImplementedError("... not yet ported")`` by name; only
    ``box-assemble`` exists, so a same-vs-alternate-method equivalence test
    is moot by construction. (Master itself excluded ``"project"`` from the
    comparison as "too bad"; DOLFINx additionally rejects ``"project"``
    outright, see ``test_deferred_energy_methods_are_rejected_precisely``.)
  - ``test_exchange_periodic_boundary_conditions`` -> RESTORED this
    session under its master name (see RESTORE section below); was the
    named dropped-coverage item from the P5 test-suite audit.

2) ``src/finmag/energies/anisotropy_test.py`` (5 functions)

  - ``test_interaction_accepts_name`` -> NOT COVERED. No ``.name``
    assertion exists for ``UniaxialAnisotropy`` anywhere in this port file
    (``Exchange``/``Zeeman`` custom-name handling is exercised elsewhere in
    this file via the same shared ``EnergyBase`` mechanism, but
    ``UniaxialAnisotropy`` itself is never checked). Genuine minor gap,
    flagged rather than silently assumed covered; not fixed here (outside
    the audit's named-restore scope).
  - ``test_anisotropy_energy_simple_configurations`` -> covered:
    ``test_anisotropy_parallel_and_perpendicular_legacy_k2_law`` (identical
    parametrize-over-``m`` shape as master, extended with a non-zero ``K2``
    term and a different ``K1``). Tolerance TIGHTENED: master
    ``rtol=1e-12`` -> port ``rel=1e-13, abs=1e-15``.
  - ``test_anisotropy_energy_analytical`` -> covered-elsewhere:
    ``test_anisotropy_density_integrates_to_energy_and_refreshes`` exercises
    the ``K1``-only path (``K2`` defaults to 0) but with a spatially
    *uniform* ``m`` rather than master's spatially-varying
    ``(0, sqrt(1-x^2), x)`` analytic-integral case; the port test is
    stronger in a different dimension (checks the energy-density integral
    identity and refresh-after-``m``-change instead).
  - ``test_anisotropy_field`` -> covered, STRONGER method:
    ``test_anisotropy_k1_k2_field_direction_ms_and_length_scaling``. Master
    re-derives ``H`` by re-assembling the same UFL weak form with a
    ``TestFunction`` (a self-consistency check against the same machinery
    under test); the port instead computes ``H`` from the closed-form
    analytic ``K1``/``K2`` field formula, an independent oracle.
  - ``test_anisotropy_field_supported_methods`` -> deferred + reason:
    same as exchange's supported-methods test above --
    ``test_deferred_energy_methods_are_rejected_precisely`` plus
    ``test_invalid_or_deferred_material_inputs_fail_explicitly``
    (``assemble=False`` raises ``NotImplementedError("... native/direct ...")``).
    Only ``box-assemble`` is ported.

3) ``src/finmag/energies/zeeman_test.py`` (14 functions total; STATIC-Zeeman
   subset only per this file's scope -- 5 functions. The remaining 9
   time-varying functions are covered in ``test_timezeeman_dolfinx.py``,
   whose own docstring cross-references back here for the static subset.)

  - ``test_interaction_accepts_name`` (``Zeeman`` part only; the
    ``TimeZeeman``/``DiscreteTimeZeeman`` parts are covered in
    ``test_timezeeman_dolfinx.py``) -> covered, stronger:
    ``test_zeeman_field_average_and_analytic_energy`` asserts a *custom*
    name (``zeeman.name == "Applied"``), stronger than master's bare
    ``hasattr`` check.
  - ``test_compute_energy`` -> covered:
    ``test_zeeman_field_average_and_analytic_energy`` (generalised to
    2D/3D, arbitrary ``H`` vector). Tolerance TIGHTENED: master
    ``rtol=1e-12`` -> port ``rel=1e-13``.
  - ``test_energy_density_function`` -> covered-elsewhere (weaker):
    ``test_zeeman_callable_set_value_keeps_live_function_and_density``.
    Master integrates ``energy_density_function()`` over the mesh
    (``df.assemble(edf * dx) * unit_length``) and compares to the
    independent analytic total ``-mu0 * H``; the port instead checks the
    raw density array pointwise against the analytic density formula and
    that ``energy_density_function() is density.f`` (a live-reference
    identity check). The INTEGRAL-equals-energy property master checked
    for ``Zeeman`` specifically is not reproduced here (it IS reproduced
    for ``Exchange``/``UniaxialAnisotropy`` via the local
    ``_density_integral`` helper, just not wired up for ``Zeeman`` too) --
    flagged as a minor gap, not fixed (outside the audit's named-restore
    scope).
  - ``test_compute_energy_in_regions`` -> covered-elsewhere (weaker):
    ``test_zeeman_compute_energy_preserves_restricted_measure_argument``.
    Master splits the mesh into two SEPARATE physical subdomains
    (``df.SubMesh``) with DIFFERENT ``m`` per domain and compares each
    against an independent per-region analytic formula at ``rtol=5e-3``;
    the port instead uses ONE spatially-uniform ``m`` over a single mesh
    split by ``meshtags``/restricted ``dx(1)``, verifying that the
    restricted half-energy equals half the total at ``rel=1e-13``. The
    measure-restriction PLUMBING is verified tightly; master's
    "different ``m`` per region matches an independent analytic per-region
    formula" property is not reproduced (``df.SubMesh``/``CellFunction``/
    ``Measure("dx")[domains]`` are removed dolfin APIs -- ``meshtags`` +
    ``subdomain_data`` is the DOLFINx equivalent already used here; see
    also ``test_energies_in_regions.py`` below, same relationship).
  - ``test_value_set_update`` -> covered, stronger:
    ``test_zeeman_callable_set_value_keeps_live_function_and_density``
    additionally asserts the live ``dolfinx.fem.Function`` object identity
    (``zeeman.H.f is function``) survives ``set_value()``, not just the
    ``.value`` attribute as master did.

   Time-varying subset (9 functions, NOT covered in this file by design --
   see ``test_timezeeman_dolfinx.py``): ``test_time_zeeman_init``,
   ``test_time_dependent_field_update``,
   ``test_time_dependent_field_switched_off``,
   ``test_discrete_time_zeeman_updates_in_intervals``,
   ``test_discrete_time_zeeman_check_arguments_are_sane``,
   ``test_discrete_time_zeeman_switchoff_only``, ``test_oscillating_zeeman``,
   ``test_dipolar_field_class``,
   ``test_compare_stray_field_of_sphere_with_dipolar_field`` (still
   out-of-scope/xfail-in-master per that file's own docstring, pending the
   deferred airbox/example machinery).

4) ``src/finmag/energies/magnetostatic_field_test.py`` (2 functions) -> NOT
   COVERED. ``MagnetostaticField`` (``finmag/energies/magnetostatic_field.py``)
   is a standalone macrospin-style class with NO ``dolfin``/FEM dependency
   at all (pure numpy), still present unchanged in the tree, but it is not
   wired into ``finmag.energies``'s lazy export table and has no DOLFINx
   test coverage. Flagged here rather than silently assumed covered; a
   cheap future slice, but out of this audit's named-restore scope (only
   ``test_exchange_periodic_boundary_conditions`` was named).
   - ``test_magnetostatic_field_for_uniformly_magnetised_sphere`` -> not covered.
   - ``test_magnetostatic_energy_density_for_uniformly_magnetised_sphere`` -> not covered.

5) ``src/finmag/energies/test_energies_in_regions.py`` (2 functions)

  - ``test_energies_in_separated_subdomains`` -> covered-elsewhere, TWO
    locations, of increasing strength: (a) in THIS file, weakly, via
    ``test_zeeman_compute_energy_preserves_restricted_measure_argument``
    (restricted-``dx`` summation property only, spatially-uniform ``m``);
    (b) more strongly, in ``test_variable_params_dolfinx.py`` via
    ``test_region_energies_sum_to_total_zeeman_oracle``,
    ``test_region_energies_sum_to_total_for_exchange`` and
    ``test_total_energy_over_region_sums_all_interactions``, which use
    ``sim.mark_regions`` + ``compute_energy(..., region=...)`` as the
    DOLFINx replacement for master's ``pair_of_disks``/``df.SubMesh``/
    ``CellFunction`` region machinery (that file's own mapping header has
    the full accounting; ``get_submesh`` itself stays a documented
    ``NotImplementedError`` deferral there since region ``dx`` measures
    cover every behavioural need exercised by ``MultiDomainTest``).
  - ``test_energies_in_touching_subdomains`` -> NOT COVERED / deferred:
    already ``@pytest.mark.xfail`` in master itself, investigating an
    unresolved touching-subdomain bug ("fails for some reason... need to
    investigate"; master's own comment). Not reproduced under DOLFINx:
    low value (master never got this working either) and depends on the
    same unported ``SubMesh`` machinery as above. (Cross-checked: also
    NOT PORTED in ``test_variable_params_dolfinx.py`` for the same reason.)

RESTORE (P5 test-suite audit): ``test_exchange_periodic_boundary_conditions``
was dropped from the initial DOLFINx aggregation of ``exchange_test.py``
without being flagged. It is restored below under its master name,
immediately after the plain-exchange tests.

PBC IS A NO-OP UNDER THIS PORT -- DOCUMENTED, NOT HIDDEN. Master's test
uses ``finmag.util.pbc2d.PeriodicBoundary2D`` (a *different* class from the
locally-defined ``PeriodicBoundary`` investigated in the sibling
``test_field_dolfinx.py`` PBC-no-op finding -- that one's ``inside()`` body,
``x[0] < DOLFIN_EPS and x[0] > DOLFIN_EPS``, is self-contradictory and hence
always ``False``; ``PeriodicBoundary2D.inside()`` uses genuinely different,
non-vacuous boundary-matching logic and is not shown here to share that
specific defect). Regardless of whether master's own periodicity detection
was correct, DOLFINx's ``dolfinx.fem.functionspace`` has **no**
``constrained_domain=`` argument at all: periodicity support lives only in
the separate ``dolfinx_mpc`` package, which nothing else in this port
depends on, and ``finmag.util.pbc2d`` itself still imports legacy
``dolfin`` (importing it here would violate
``test_ported_energy_exports_do_not_load_legacy_dolfin`` above). So under
DOLFINx there is structurally no way to build a genuinely periodic "_pbc"
function space in this test: it is reproduced below as the SAME plain CG1
space as the "_normal" variant. The restored test therefore -- like
master's own ``field_test.py`` PBC sweep -- does not actually exercise
periodicity; it degenerates to running the same uniform-``m``
zero-exchange-field/energy check twice on the same space. Master's
tolerances (``FIELD_TOLERANCE = 6e-7``, ``ENERGY_TOLERANCE = 0.0``) are kept
VERBATIM and pass (measured: both computed quantities are exact/near-exact
zero for box-assemble exchange on a uniform field).
"""

import sys

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from ufl import Measure

from finmag.energies import (
    EnergyBase,
    Exchange,
    TimeZeeman,
    UniaxialAnisotropy,
    Zeeman,
)
from finmag.energies.energy_base import mu0
from finmag.field import Field, owned_raw_to_blocked


def _node_rows(blocked_field):
    """Reshape a legacy component-blocked (``xxx``) field to per-node rows.

    Task 31: ``compute_field()`` now returns component-blocked
    ``[x0..xN, y0..yN, z0..zN]``; per-node ``(N, 3)`` rows (owned-vertex order)
    are ``reshape((3, -1)).T``, not the old raw-interleaved ``reshape((-1, 3))``.
    """
    return blocked_field.reshape((3, -1)).T


def _domain(dimension, cells=2):
    if dimension == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, cells, cells)
    if dimension == 3:
        return mesh.create_unit_cube(MPI.COMM_WORLD, cells, cells, cells)
    raise ValueError("test domains are two- or three-dimensional")


def _fields(dimension=2, m=(1.0, 0.0, 0.0), Ms=2.5, cells=2):
    domain = _domain(dimension, cells)
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    scalar_space = fem.functionspace(domain, ("DG", 0))
    return domain, Field(vector_space, m, name="m"), Field(
        scalar_space, Ms, name="Ms"
    )


def _density_integral(interaction, density):
    local_value = np.dot(density, interaction.nodal_volume_S1)
    mesh_value = interaction.m.mesh().comm.allreduce(local_value, op=MPI.SUM)
    return mesh_value * interaction.unit_length**interaction.dim


def test_ported_energy_exports_do_not_load_legacy_dolfin():
    assert EnergyBase.__module__ == "finmag.energies.energy_base"
    assert Exchange.__module__ == "finmag.energies.exchange"
    assert Zeeman.__module__ == "finmag.energies.zeeman"
    assert UniaxialAnisotropy.__module__ == "finmag.energies.anisotropy"
    assert "dolfin" not in sys.modules


@pytest.mark.parametrize(
    "method", ("box-matrix-numpy", "box-matrix-petsc", "project", "direct")
)
def test_deferred_energy_methods_are_rejected_precisely(method):
    with pytest.raises(NotImplementedError, match="not yet ported"):
        EnergyBase(method=method)
    with pytest.raises(NotImplementedError, match="not yet ported"):
        Exchange(1.0, method=method)


def test_unknown_energy_method_is_not_silently_mapped():
    with pytest.raises(ValueError, match="unsupported energy method"):
        EnergyBase(method="mystery")


def test_box_energy_rejects_unsupported_higher_order_space_explicitly():
    domain = _domain(2)
    quadratic_space = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    scalar_space = fem.functionspace(domain, ("DG", 0))
    m = Field(quadratic_space, (1.0, 0.0, 0.0))
    Ms = Field(scalar_space, 2.5)

    with pytest.raises(NotImplementedError, match="CG1"):
        Exchange(1.0).setup(m, Ms)


@pytest.mark.parametrize("dimension", (2, 3))
def test_zeeman_field_average_and_analytic_energy(dimension):
    _, m, Ms = _fields(dimension, m=(0.6, 0.0, 0.8), Ms=8.0e5)
    H = np.array((1.0e6, -2.0e5, 3.0e5))
    unit_length = 2.0e-9
    zeeman = Zeeman(H, name="Applied")
    zeeman.setup(m, Ms, unit_length)

    assert zeeman.name == "Applied"
    assert zeeman.in_jacobian is False
    assert np.allclose(_node_rows(zeeman.compute_field()), H)
    assert np.allclose(zeeman.average_field(), H)
    expected = -mu0 * 8.0e5 * np.dot((0.6, 0.0, 0.8), H)
    expected *= unit_length**dimension
    assert zeeman.compute_energy() == pytest.approx(expected, rel=1e-13)


def test_zeeman_callable_set_value_keeps_live_function_and_density():
    _, m, Ms = _fields(2, m=(1.0, 0.0, 0.0), Ms=4.0)
    zeeman = Zeeman(
        lambda x: np.vstack(
            (1.0 + x[0], np.full(x.shape[1], 2.0), np.full(x.shape[1], 3.0))
        )
    )
    zeeman.setup(m, Ms)
    function = zeeman.H.f

    assert np.allclose(zeeman.average_field(), (1.5, 2.0, 3.0))
    initial_density = zeeman.energy_density()
    coordinates, density_values = initial_density.coords_and_values()
    assert np.allclose(density_values, -4.0 * mu0 * (1.0 + coordinates[:, 0]))
    zeeman.set_value((-2.0, 0.0, 0.0))
    assert zeeman.H.f is function
    assert np.allclose(_node_rows(zeeman.compute_field()), (-2.0, 0.0, 0.0))
    assert zeeman.compute_energy() == pytest.approx(8.0 * mu0)

    density = zeeman.energy_density()
    assert isinstance(density, Field)
    assert np.allclose(density.as_array(), 8.0 * mu0)
    assert zeeman.energy_density_function() is density.f


def test_zeeman_setup_rebuilds_field_for_a_new_mesh():
    _, first_m, first_ms = _fields(2, m=(1.0, 0.0, 0.0), Ms=4.0, cells=1)
    zeeman = Zeeman((2.0, 0.0, 0.0))
    zeeman.setup(first_m, first_ms)
    assert np.isfinite(zeeman.compute_energy())

    second_domain, second_m, second_ms = _fields(
        3, m=(1.0, 0.0, 0.0), Ms=4.0, cells=1
    )
    zeeman.setup(second_m, second_ms)
    assert zeeman.H.functionspace.mesh is second_domain
    assert zeeman.compute_energy() == pytest.approx(-8.0 * mu0)


def test_zeeman_compute_energy_preserves_restricted_measure_argument():
    domain, m, Ms = _fields(2, m=(1.0, 0.0, 0.0), Ms=4.0, cells=4)
    zeeman = Zeeman((2.0, 0.0, 0.0))
    zeeman.setup(m, Ms)

    cell_dim = domain.topology.dim
    left_cells = mesh.locate_entities(
        domain, cell_dim, lambda x: x[0] <= 0.5 + 1.0e-12
    )
    tags = mesh.meshtags(
        domain,
        cell_dim,
        np.sort(left_cells),
        np.ones(left_cells.size, dtype=np.int32),
    )
    restricted_dx = Measure("dx", domain=domain, subdomain_data=tags)(1)

    assert zeeman.compute_energy(dx=restricted_dx) == pytest.approx(
        0.5 * zeeman.compute_energy(), rel=1e-13
    )


def test_exchange_uniform_field_is_exactly_zero():
    _, m, Ms = _fields(2, m=(1.0, 0.0, 0.0), Ms=8.0e5)
    exchange = Exchange(13.0e-12)
    exchange.setup(m, Ms, unit_length=1.0e-9)

    assert exchange.method == "box-assemble"
    assert exchange.name == "Exchange"
    assert exchange.in_jacobian is True
    assert exchange.compute_energy() == pytest.approx(0.0, abs=0.0)
    assert np.allclose(exchange.compute_field(), 0.0, atol=1e-12)


@pytest.mark.parametrize("dimension", (2, 3))
def test_exchange_linear_energy_and_physical_length_scaling(dimension):
    _, m, Ms = _fields(dimension, Ms=2.5)
    m.set(
        lambda x: np.vstack(
            (x[0], x[1], np.zeros(x.shape[1], dtype=np.float64))
        )
    )
    A = 5.0
    unit_length = 0.25
    exchange = Exchange(A, name="Ex")
    exchange.setup(m, Ms, unit_length)

    expected = 2.0 * A * unit_length ** (dimension - 2)
    assert exchange.compute_energy() == pytest.approx(expected, rel=1e-13)

    density = exchange.energy_density()
    assert _density_integral(exchange, density) == pytest.approx(
        exchange.compute_energy(), rel=1e-13
    )
    assert exchange.energy_density_function().function_space is exchange.S1


def test_exchange_field_scales_as_inverse_unit_length_squared():
    _, m, Ms = _fields(2, Ms=2.5, cells=3)
    m.set(
        lambda x: np.vstack(
            (x[0], 2.0 * x[1], np.zeros(x.shape[1], dtype=np.float64))
        )
    )
    first = Exchange(5.0)
    second = Exchange(5.0)
    first.setup(m, Ms, unit_length=1.0)
    second.setup(m, Ms, unit_length=2.0)
    H1 = first.compute_field()
    H2 = second.compute_field()

    assert np.max(np.abs(H1)) > 0.0
    assert np.allclose(H2, H1 / 4.0)
    # H1 is component-blocked (Task 31); pair it with the blocked nodal volumes
    # so each component's volume-weighted sum (the exchange field integrates to
    # zero) contracts matching nodes.
    vol_blocked = owned_raw_to_blocked(
        first.m.functionspace, first.nodal_volume_S3
    )
    weighted = (H1 * vol_blocked).reshape((3, -1)).sum(1)
    assert np.allclose(weighted, 0.0, atol=1e-8)


def test_exchange_periodic_boundary_conditions():
    """Restored (P5 test-suite audit): master
    ``exchange_test.py::test_exchange_periodic_boundary_conditions``.

    See the module docstring's "PBC IS A NO-OP UNDER THIS PORT" section for
    the full explanation. In short: DOLFINx has no ``constrained_domain=``
    argument, so the "_pbc" function space below is necessarily the SAME
    plain CG1 space as the "_normal" one -- this test does not actually
    exercise periodicity, mirroring master's own (differently-caused)
    ``field_test.py`` PBC no-op. Master's tolerances are kept verbatim.
    """
    mesh1 = mesh.create_box(
        MPI.COMM_WORLD,
        [(0.0, 0.0, 0.0), (1.0, 1.0, 0.1)],
        [2, 2, 1],
        mesh.CellType.tetrahedron,
    )
    mesh2 = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

    for domain in (mesh1, mesh2):
        S3_normal = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        # DOLFINx has no `constrained_domain=`; `finmag.util.pbc2d` (which
        # provides master's `PeriodicBoundary2D`) is dolfin-only and is
        # deliberately not imported here (see module docstring). So the
        # "_pbc" space is, of necessity, identical to S3_normal.
        S3_pbc = S3_normal

        for S3 in (S3_normal, S3_pbc):
            FIELD_TOLERANCE = 6e-7  # master verbatim
            ENERGY_TOLERANCE = 0.0  # master verbatim

            Ms_space = fem.functionspace(domain, ("DG", 0))
            m_field = Field(S3, (0.0, 0.0, 1.0), name="m")
            Ms_field = Field(Ms_space, 1.0)

            exch = Exchange(1.0)
            exch.setup(m_field, Ms_field)
            field = exch.compute_field()
            energy = exch.compute_energy()

            # measured: both ~0 (box-assemble exchange field/energy of an
            # exactly uniform m is exact up to floating-point round-off).
            assert np.max(np.abs(field)) < FIELD_TOLERANCE
            assert abs(energy) <= ENERGY_TOLERANCE


def test_exchange_and_anisotropy_can_rebind_to_a_new_mesh():
    _, first_m, first_ms = _fields(2, m=(1.0, 0.0, 0.0), Ms=2.5, cells=1)
    exchange = Exchange(5.0)
    anisotropy = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0), K2=1.5)
    exchange.setup(first_m, first_ms)
    anisotropy.setup(first_m, first_ms)
    assert exchange.compute_energy() == pytest.approx(0.0, abs=0.0)
    assert anisotropy.compute_energy() == pytest.approx(4.0)
    first_density_function = exchange.energy_density_function()

    second_domain, second_m, second_ms = _fields(
        3, m=(0.0, 0.0, 1.0), Ms=2.5, cells=1
    )
    exchange.setup(second_m, second_ms)
    anisotropy.setup(second_m, second_ms)
    assert exchange.A.mesh() is second_domain
    assert anisotropy.axis.mesh() is second_domain
    assert exchange.compute_energy() == pytest.approx(0.0, abs=0.0)
    assert anisotropy.compute_energy() == pytest.approx(-1.5)
    second_density_function = exchange.energy_density_function()
    assert second_density_function is not first_density_function
    assert second_density_function.function_space.mesh is second_domain


@pytest.mark.parametrize(
    ("magnetisation", "expected_density"),
    (((0.0, 0.0, 1.0), -1.5), ((0.0, 0.0, -1.0), -1.5), ((1.0, 0.0, 0.0), 4.0)),
)
def test_anisotropy_parallel_and_perpendicular_legacy_k2_law(
    magnetisation, expected_density
):
    _, m, Ms = _fields(3, m=magnetisation, Ms=2.5)
    anisotropy = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0), K2=1.5)
    anisotropy.setup(m, Ms, unit_length=0.5)

    assert anisotropy.compute_energy() == pytest.approx(
        expected_density * 0.5**3, rel=1e-13, abs=1e-15
    )


def test_anisotropy_k1_k2_field_direction_ms_and_length_scaling():
    _, m, Ms = _fields(2, m=(0.6, 0.0, 0.8), Ms=2.5)
    first = UniaxialAnisotropy(4.0, (0.0, 0.0, 5.0), K2=1.5)
    second = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0), K2=1.5)
    first.setup(m, Ms, unit_length=1.0)
    second.setup(m, Ms, unit_length=3.0)

    expected_z = (2.0 * 4.0 * 0.8 + 4.0 * 1.5 * 0.8**3) / (mu0 * 2.5)
    expected = np.array((0.0, 0.0, expected_z))
    H1 = _node_rows(first.compute_field())
    H2 = _node_rows(second.compute_field())
    assert np.allclose(H1, expected, rtol=1e-13, atol=1e-9)
    assert np.allclose(H2, expected, rtol=1e-13, atol=1e-9)
    assert np.allclose(first.axis.coords_and_values()[1], (0.0, 0.0, 1.0))

    expected_density = 4.0 * (1.0 - 0.8**2) - 1.5 * 0.8**4
    assert first.compute_energy() == pytest.approx(expected_density, rel=1e-13)
    assert second.compute_energy() == pytest.approx(
        expected_density * 3.0**2, rel=1e-13
    )


def test_anisotropy_reassembles_nonlinear_k2_after_m_changes():
    _, m, Ms = _fields(2, m=(0.6, 0.0, 0.8), Ms=2.5)
    anisotropy = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0), K2=1.5)
    anisotropy.setup(m, Ms)
    first = _node_rows(anisotropy.compute_field())

    m.set((0.8, 0.0, 0.6))
    second = _node_rows(anisotropy.compute_field())
    expected_z = (2.0 * 4.0 * 0.6 + 4.0 * 1.5 * 0.6**3) / (mu0 * 2.5)
    assert not np.allclose(first, second)
    assert np.allclose(second, (0.0, 0.0, expected_z), rtol=1e-13, atol=1e-9)


def test_box_field_matches_finite_difference_energy_sign_and_mu0():
    _, m, Ms = _fields(2, m=(0.6, 0.0, 0.8), Ms=2.5)
    anisotropy = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0), K2=1.5)
    anisotropy.setup(m, Ms, unit_length=0.25)
    base = np.array((0.6, 0.0, 0.8))
    perturbation = np.array((0.1, -0.2, 0.3))
    epsilon = 1.0e-7

    m.set(base + epsilon * perturbation)
    plus = anisotropy.compute_energy()
    m.set(base - epsilon * perturbation)
    minus = anisotropy.compute_energy()
    m.set(base)
    numerical_derivative = (plus - minus) / (2.0 * epsilon)

    H = _node_rows(anisotropy.compute_field())
    delta = np.repeat(perturbation[None, :], H.shape[0], axis=0)
    # Blocked field paired with blocked nodal volumes, both as owned-vertex rows.
    volumes = _node_rows(
        owned_raw_to_blocked(
            anisotropy.m.functionspace, anisotropy.nodal_volume_S3
        )
    )
    local_pairing = np.sum(H * delta * volumes)
    pairing = m.mesh().comm.allreduce(local_pairing, op=MPI.SUM)
    expected_derivative = (
        -mu0 * 2.5 * anisotropy.unit_length**anisotropy.dim * pairing
    )
    assert numerical_derivative == pytest.approx(expected_derivative, rel=1e-8)


def test_anisotropy_density_integrates_to_energy_and_refreshes():
    _, m, Ms = _fields(2, m=(1.0, 0.0, 0.0), Ms=2.5)
    anisotropy = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0))
    anisotropy.setup(m, Ms, unit_length=0.5)
    density = anisotropy.energy_density()
    density_function = anisotropy.energy_density_function()
    assert _density_integral(anisotropy, density) == pytest.approx(
        anisotropy.compute_energy(), rel=1e-13
    )

    m.set((0.0, 0.0, 1.0))
    assert np.allclose(anisotropy.energy_density(), 0.0)
    assert anisotropy.energy_density_function() is density_function
    assert np.allclose(density_function.x.array, 0.0)


def test_invalid_or_deferred_material_inputs_fail_explicitly():
    domain, m, Ms = _fields()

    # Task 16: spatially varying A/K1/K2/axis/Ms are now SUPPORTED (see
    # test_variable_params_dolfinx.py). Only legacy string Expressions remain
    # deferred by name (DOLFINx has no Expression object; pass a callable).
    with pytest.raises(NotImplementedError, match="string Expression"):
        Exchange("x[0]")
    with pytest.raises(NotImplementedError, match="string Expression"):
        UniaxialAnisotropy("x[0]", (0.0, 0.0, 1.0))
    with pytest.raises(NotImplementedError, match="string Expression"):
        UniaxialAnisotropy(1.0, ("0", "0", "1"))
    with pytest.raises(ValueError, match="non-zero"):
        UniaxialAnisotropy(1.0, (0.0, 0.0, 0.0))
    with pytest.raises(NotImplementedError, match="native/direct"):
        UniaxialAnisotropy(1.0, (0.0, 0.0, 1.0), assemble=False)
    # Task 15: TimeZeeman is now ported; a constant-array field_expression
    # with no t_off raises ValueError (no time update would ever happen),
    # not the old by-name deferral. See test_timezeeman_dolfinx.py for the
    # full ported-class suite.
    with pytest.raises(ValueError, match="t_off"):
        TimeZeeman((1.0, 0.0, 0.0))

    exchange = Exchange(1.0)
    with pytest.raises(ValueError, match="unit_length"):
        exchange.setup(m, Ms, unit_length=0.0)


def test_nonpositive_ms_is_rejected_collectively():
    _, m, Ms = _fields(Ms=0.0)
    with pytest.raises(ValueError, match="Ms must be positive"):
        Exchange(1.0).setup(m, Ms)
    with pytest.raises(ValueError, match="Ms must be positive"):
        Zeeman((1.0, 0.0, 0.0)).setup(m, Ms)
