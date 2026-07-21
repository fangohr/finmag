"""Focused production tests for the first DOLFINx energy slice."""

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
from finmag.field import Field


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
    assert np.allclose(zeeman.compute_field().reshape((-1, 3)), H)
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
    assert np.allclose(zeeman.compute_field().reshape((-1, 3)), (-2.0, 0.0, 0.0))
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
    weighted = (H1 * first.nodal_volume_S3).reshape((-1, 3)).sum(0)
    assert np.allclose(weighted, 0.0, atol=1e-8)


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
    H1 = first.compute_field().reshape((-1, 3))
    H2 = second.compute_field().reshape((-1, 3))
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
    first = anisotropy.compute_field().copy()

    m.set((0.8, 0.0, 0.6))
    second = anisotropy.compute_field().reshape((-1, 3))
    expected_z = (2.0 * 4.0 * 0.6 + 4.0 * 1.5 * 0.6**3) / (mu0 * 2.5)
    assert not np.allclose(first, second.reshape(-1))
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

    H = anisotropy.compute_field().reshape((-1, 3))
    delta = np.repeat(perturbation[None, :], H.shape[0], axis=0)
    volumes = anisotropy.nodal_volume_S3.reshape((-1, 3))
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
    scalar_space = fem.functionspace(domain, ("DG", 0))
    spatial = Field(scalar_space, 1.0)

    with pytest.raises(NotImplementedError, match="spatially varying A"):
        Exchange(spatial)
    with pytest.raises(NotImplementedError, match="spatially varying A"):
        Exchange("x[0]")
    with pytest.raises(NotImplementedError, match="spatially varying A"):
        Exchange(lambda x: 1.0 + x[0])
    with pytest.raises(NotImplementedError, match="spatially varying K1"):
        UniaxialAnisotropy(spatial, (0.0, 0.0, 1.0))
    with pytest.raises(NotImplementedError, match="spatially varying K1"):
        UniaxialAnisotropy("x[0]", (0.0, 0.0, 1.0))
    with pytest.raises(NotImplementedError, match="spatially varying K2"):
        UniaxialAnisotropy(1.0, (0.0, 0.0, 1.0), K2=lambda x: x[0])
    with pytest.raises(NotImplementedError, match="spatially varying anisotropy"):
        UniaxialAnisotropy(1.0, lambda x: (0.0, 0.0, 1.0))
    with pytest.raises(NotImplementedError, match="spatially varying anisotropy"):
        UniaxialAnisotropy(1.0, ("0", "0", "1"))
    varying_ms_space = fem.functionspace(domain, ("Lagrange", 1))
    varying_ms = Field(varying_ms_space, lambda x: 1.0 + x[0])
    with pytest.raises(NotImplementedError, match="spatially varying Ms"):
        Exchange(1.0).setup(m, varying_ms)
    with pytest.raises(NotImplementedError, match="spatially varying Ms"):
        Zeeman((1.0, 0.0, 0.0)).setup(m, varying_ms)
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
