"""Tests for the first DOLFINx-only micromagnetic prototype helpers.

These tests are not a legacy Finmag regression yet. They provide a small,
documented target for M4 so the future port can grow from checked DOLFINx
semantics rather than from untested exploratory snippets. [Codex gpt-5.5 high]
"""

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from dev.dolfinx.prototype import MU0, constant_vector_function
from dev.dolfinx.prototype import exchange_energy, uniaxial_anisotropy_energy
from dev.dolfinx.prototype import explicit_llg_step, llg_rhs, nodal_vector_values
from dev.dolfinx.prototype import vector_function_space, zeeman_energy


def test_constant_vector_function_sets_all_components():
    """Check that vector field setup produces the requested constant values."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)

    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    values = nodal_vector_values(magnetisation)
    assert np.allclose(values, (1.0, 0.0, 0.0))


def test_constant_vector_function_rejects_wrong_component_count():
    """Keep component-count mistakes explicit during the prototype phase."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)

    with pytest.raises(ValueError, match="function space expects 3"):
        constant_vector_function(function_space, (1.0, 0.0))


def test_zeeman_energy_for_constant_field_on_unit_square():
    """Compare DOLFINx Zeeman assembly with the analytical unit-square value."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = zeeman_energy(
        magnetisation,
        field=(2.0, 0.0, 0.0),
        saturation_magnetisation=3.0,
    )

    assert np.isclose(energy, -6.0 * MU0)


def test_zeeman_energy_applies_unit_length_scaling():
    """DOLFINx integrates over mesh coordinates, so physical length must scale."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = zeeman_energy(
        magnetisation,
        field=(2.0, 0.0, 0.0),
        saturation_magnetisation=3.0,
        unit_length=1e-9,
    )

    assert np.isclose(energy, -6.0 * MU0 * 1e-18)


def test_uniaxial_anisotropy_energy_for_parallel_axis():
    """Parallel magnetisation and easy axis should have zero anisotropy energy."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = uniaxial_anisotropy_energy(
        magnetisation,
        axis=(1.0, 0.0, 0.0),
        anisotropy_constant=7.0,
    )

    assert np.isclose(energy, 0.0)


def test_uniaxial_anisotropy_energy_for_perpendicular_axis():
    """Perpendicular magnetisation and easy axis should integrate to K."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = uniaxial_anisotropy_energy(
        magnetisation,
        axis=(0.0, 1.0, 0.0),
        anisotropy_constant=7.0,
    )

    assert np.isclose(energy, 7.0)


def test_uniaxial_anisotropy_energy_normalises_axis_and_scales_length():
    """Axis normalisation and unit-length scaling are part of the public helper."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = uniaxial_anisotropy_energy(
        magnetisation,
        axis=(2.0, 2.0, 0.0),
        anisotropy_constant=7.0,
        unit_length=1e-9,
    )

    assert np.isclose(energy, 3.5e-18)


def test_uniaxial_anisotropy_energy_rejects_zero_axis():
    """A zero easy axis would silently make the anisotropy definition invalid."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="axis must be non-zero"):
        uniaxial_anisotropy_energy(
            magnetisation,
            axis=(0.0, 0.0, 0.0),
            anisotropy_constant=7.0,
        )


def test_uniaxial_anisotropy_energy_zero_constant_short_circuits():
    """A zero anisotropy constant should not build a degenerate UFL form."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy = uniaxial_anisotropy_energy(
        magnetisation,
        axis=(0.0, 0.0, 0.0),
        anisotropy_constant=0.0,
    )

    assert np.isclose(energy, 0.0)


def test_exchange_energy_zero_for_constant_magnetisation():
    """A constant magnetisation has zero exchange energy."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    assert np.isclose(exchange_energy(magnetisation, exchange_constant=5.0), 0.0)


def test_exchange_energy_for_linear_field_on_unit_square():
    """For m=(x,y,0), integral grad(m):grad(m) over the unit square is 2."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = fem.Function(function_space)
    magnetisation.interpolate(
        lambda x: np.vstack((x[0], x[1], np.zeros(x.shape[1])))
    )

    assert np.isclose(exchange_energy(magnetisation, exchange_constant=5.0), 10.0)


def test_exchange_energy_applies_gradient_unit_length_scaling():
    """A 3D gradient energy scales as one power of the physical unit length."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = fem.Function(function_space)
    magnetisation.interpolate(
        lambda x: np.vstack((x[0], x[1], np.zeros(x.shape[1])))
    )

    energy = exchange_energy(
        magnetisation,
        exchange_constant=5.0,
        unit_length=1e-9,
    )

    assert np.isclose(energy, 10.0e-9)


def test_exchange_energy_rejects_non_positive_unit_length():
    """Invalid physical length scaling should fail before form assembly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="unit_length must be positive"):
        exchange_energy(magnetisation, exchange_constant=5.0, unit_length=0.0)


def test_exchange_energy_zero_constant_short_circuits():
    """A zero exchange constant should not build a degenerate UFL form."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    assert np.isclose(exchange_energy(magnetisation, exchange_constant=0.0), 0.0)


def test_llg_rhs_for_constant_field_has_expected_direction():
    """A damped spin initially along x should precess and relax toward z."""
    magnetisation = np.array([[1.0, 0.0, 0.0]])
    effective_field = np.array([[0.0, 0.0, 1.0]])

    rhs = llg_rhs(magnetisation, effective_field, gamma=1.0, alpha=1.0)

    assert np.allclose(rhs, [[0.0, 0.5, 0.5]])


def test_explicit_llg_step_preserves_norm_and_lowers_zeeman_energy():
    """The first checked M4 time step should keep |m|=1 and reduce energy."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy_before = zeeman_energy(
        magnetisation,
        field=(0.0, 0.0, 1.0),
        saturation_magnetisation=1.0,
    )
    explicit_llg_step(
        magnetisation,
        effective_field=(0.0, 0.0, 1.0),
        dt=1e-3,
        gamma=1.0,
        alpha=1.0,
    )
    energy_after = zeeman_energy(
        magnetisation,
        field=(0.0, 0.0, 1.0),
        saturation_magnetisation=1.0,
    )

    values = nodal_vector_values(magnetisation)
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)
    assert np.all(values[:, 1] > 0.0)
    assert np.all(values[:, 2] > 0.0)
    assert energy_after < energy_before


def test_explicit_llg_step_rejects_invalid_inputs():
    """Invalid timestep and field shape errors should be explicit."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="dt must be positive"):
        explicit_llg_step(magnetisation, effective_field=(0.0, 0.0, 1.0), dt=0.0)

    with pytest.raises(ValueError, match="effective field has 2 components"):
        explicit_llg_step(magnetisation, effective_field=(0.0, 1.0), dt=1e-3)
