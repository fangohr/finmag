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
from dev.dolfinx.prototype import cubic_anisotropy_energy, dmi_energy
from dev.dolfinx.prototype import exchange_energy, uniaxial_anisotropy_energy
from dev.dolfinx.prototype import effective_field_llg_step, effective_field_values
from dev.dolfinx.prototype import explicit_llg_step, llg_rhs, nodal_vector_values
from dev.dolfinx.prototype import nodal_volume
from dev.dolfinx.prototype import vector_function_space, zeeman_energy
from dev.dolfinx.relaxation_example import RelaxationParameters


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


def test_dmi_energy_zero_for_constant_magnetisation():
    """A constant magnetisation has zero curl, so zero DMI energy."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    assert np.isclose(dmi_energy(magnetisation, dmi_constant=5.0), 0.0)


def test_dmi_energy_for_linear_field_on_unit_cube():
    """For m=(z,x,y), curl(m)=(1,1,1) and integral(m.curl(m)) over the unit
    cube is integral(x+y+z) = 1.5, so the DMI energy is D * 1.5."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = fem.Function(function_space)
    magnetisation.interpolate(
        lambda x: np.vstack((x[2], x[0], x[1]))
    )

    assert np.isclose(dmi_energy(magnetisation, dmi_constant=5.0), 7.5)


def test_dmi_energy_applies_unit_length_scaling():
    """The DMI curl term has one derivative, so energy scales as unit_length**2
    in 3D (dim - 1)."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = fem.Function(function_space)
    magnetisation.interpolate(
        lambda x: np.vstack((x[2], x[0], x[1]))
    )

    energy = dmi_energy(magnetisation, dmi_constant=5.0, unit_length=1e-9)

    assert np.isclose(energy, 7.5e-18)


def test_dmi_energy_rejects_non_3d_mesh():
    """The reduced prototype only supports the standard 3D bulk DMI term."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="only supports 3D meshes"):
        dmi_energy(magnetisation, dmi_constant=5.0)


def test_dmi_energy_rejects_non_positive_unit_length():
    """Invalid physical length scaling should fail before form assembly."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="unit_length must be positive"):
        dmi_energy(magnetisation, dmi_constant=5.0, unit_length=0.0)


def test_dmi_energy_zero_constant_short_circuits():
    """A zero DMI constant should not build a degenerate UFL form."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    assert np.isclose(dmi_energy(magnetisation, dmi_constant=0.0), 0.0)


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


def test_nodal_volume_sums_to_total_mesh_volume():
    """Box/lumped nodal volumes should sum to the mesh's total area."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    scalar_space = fem.functionspace(domain, ("Lagrange", 1))
    volume = nodal_volume(scalar_space)

    assert np.isclose(domain.comm.allreduce(volume.sum(), op=MPI.SUM), 1.0)


def test_nodal_volume_vector_space_sums_per_component():
    """Each component block of a vector space's nodal volume should sum to
    the mesh's total area, matching the scalar-space case."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    vector_space = vector_function_space(domain)
    volume = nodal_volume(vector_space).reshape((-1, 3))

    totals = domain.comm.allreduce(volume.sum(axis=0), op=MPI.SUM)
    assert np.allclose(totals, 1.0)


def test_nodal_volume_applies_unit_length_scaling():
    """Nodal volume should scale as unit_length**dim, like the other energies."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    scalar_space = fem.functionspace(domain, ("Lagrange", 1))

    volume = nodal_volume(scalar_space, unit_length=1e-9)

    assert np.isclose(
        domain.comm.allreduce(volume.sum(), op=MPI.SUM), 1.0 * 1e-9**2
    )


def test_nodal_volume_rejects_non_positive_unit_length():
    """Invalid physical length scaling should fail before form assembly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    scalar_space = fem.functionspace(domain, ("Lagrange", 1))

    with pytest.raises(ValueError, match="unit_length must be positive"):
        nodal_volume(scalar_space, unit_length=0.0)


def test_effective_field_matches_applied_field_for_zeeman_only():
    """For a uniform applied field with all other terms off, H_eff == field
    exactly. This is the defining property of the Zeeman effective field, and
    validates the whole box-method pipeline (derivative assembly, ghost
    accumulation, nodal-volume division) before trusting it for less trivial
    terms. [GitHub Copilot / Claude Sonnet 5]
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))
    parameters = RelaxationParameters(
        anisotropy_constant=0.0,
        exchange_constant=0.0,
        field=(2.0, -1.0, 0.5),
        saturation_magnetisation=3.0,
    )

    field = effective_field_values(magnetisation, parameters)

    assert np.allclose(field, (2.0, -1.0, 0.5))


def test_effective_field_zero_for_constant_magnetisation_exchange_only():
    """Exchange field should vanish for a spatially constant magnetisation."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))
    parameters = RelaxationParameters(
        anisotropy_constant=0.0,
        exchange_constant=5.0,
        field=(0.0, 0.0, 0.0),
        saturation_magnetisation=1.0,
    )

    field = effective_field_values(magnetisation, parameters)

    assert np.allclose(field, 0.0, atol=1e-10)


def test_effective_field_rejects_non_positive_saturation_magnetisation():
    """An invalid Ms should fail before any form assembly."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))
    parameters = object.__new__(RelaxationParameters)
    object.__setattr__(parameters, "saturation_magnetisation", 0.0)

    with pytest.raises(ValueError, match="saturation_magnetisation must be positive"):
        effective_field_values(magnetisation, parameters)


def test_effective_field_llg_step_preserves_norm_and_lowers_zeeman_energy():
    """The effective-field stepper should behave like the fixed-field one
    when only Zeeman is active (H_eff == field in that case)."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))
    parameters = RelaxationParameters(
        anisotropy_constant=0.0,
        exchange_constant=0.0,
        field=(0.0, 0.0, 1.0),
        saturation_magnetisation=1.0,
    )

    energy_before = zeeman_energy(
        magnetisation, field=(0.0, 0.0, 1.0), saturation_magnetisation=1.0
    )
    effective_field_llg_step(magnetisation, parameters, dt=1e-3, gamma=1.0, alpha=1.0)
    energy_after = zeeman_energy(
        magnetisation, field=(0.0, 0.0, 1.0), saturation_magnetisation=1.0
    )

    values = nodal_vector_values(magnetisation)
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)
    assert energy_after < energy_before


def test_effective_field_llg_step_is_driven_by_anisotropy_unlike_fixed_field_step():
    """With zero applied field, only effective_field_llg_step should move ``m``
    toward the anisotropy easy axis; explicit_llg_step ignores anisotropy
    entirely since it only precesses toward its fixed field argument. This is
    a direct regression test for the limitation documented after wiring DMI/
    cubic anisotropy into the simulation wrappers. [GitHub Copilot / Claude
    Sonnet 5]
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)

    fixed_field_m = constant_vector_function(function_space, (1.0, 0.0, 0.001))
    explicit_llg_step(
        fixed_field_m, effective_field=(0.0, 0.0, 0.0), dt=1e-2, gamma=1.0, alpha=1.0
    )
    assert np.allclose(nodal_vector_values(fixed_field_m)[:, 2], 0.001, atol=1e-9)

    effective_field_m = constant_vector_function(function_space, (1.0, 0.0, 0.001))
    parameters = RelaxationParameters(
        anisotropy_axis=(0.0, 0.0, 1.0),
        anisotropy_constant=5.0,
        exchange_constant=0.0,
        field=(0.0, 0.0, 0.0),
        saturation_magnetisation=1.0,
    )
    effective_field_llg_step(effective_field_m, parameters, dt=1e-2, gamma=1.0, alpha=1.0)
    assert np.all(nodal_vector_values(effective_field_m)[:, 2] > 0.001)


def test_effective_field_llg_step_rejects_invalid_dt():
    """Invalid timestep should fail before computing the effective field."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))
    parameters = RelaxationParameters()

    with pytest.raises(ValueError, match="dt must be positive"):
        effective_field_llg_step(magnetisation, parameters, dt=0.0)


# Reference constants copied from src/finmag/energies/cubic_anisotropy_test.py
# so this prototype energy is checked against finmag's own legacy analytic
# case, not just a self-derived one. [GitHub Copilot / Claude Sonnet 5]
_CUBIC_K1 = -8608726
_CUBIC_K2 = -13744132
_CUBIC_K3 = 1100269
_CUBIC_U1 = (0, -0.7071, 0.7071)
_CUBIC_U2 = (0, 0.7071, 0.7071)


def _legacy_cubic_anisotropy_energy_density(m):
    """Port of compute_cubic_energy() from cubic_anisotropy_test.py."""
    u3 = np.cross(_CUBIC_U1, _CUBIC_U2)
    a = np.dot(_CUBIC_U1, m)
    b = np.dot(_CUBIC_U2, m)
    c = np.dot(u3, m)
    energy = _CUBIC_K1 * (a**2 * b**2 + a**2 * c**2 + b**2 * c**2)
    energy += _CUBIC_K2 * (a**2 * b**2 * c**2)
    energy += _CUBIC_K3 * (a**4 * b**4 + a**4 * c**4 + b**4 * c**4)
    return energy


def test_cubic_anisotropy_energy_matches_legacy_reference_values():
    """Reproduce the legacy finmag cubic_anisotropy_test.py analytic case
    directly: same K1/K2/K3, axes, and constant magnetisation. [Codex
    gpt-5.5 high]
    """
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (0.0, 0.0, 1.0))

    energy = cubic_anisotropy_energy(
        magnetisation, _CUBIC_U1, _CUBIC_U2, _CUBIC_K1, _CUBIC_K2, _CUBIC_K3
    )

    # The unit cube has volume 1, so the total energy equals the density.
    expected = _legacy_cubic_anisotropy_energy_density((0.0, 0.0, 1.0))
    assert np.isclose(energy, expected, rtol=1e-6)


def test_cubic_anisotropy_energy_applies_unit_length_scaling():
    """No spatial derivatives are involved, so scaling is unit_length**dim,
    matching zeeman_energy/uniaxial_anisotropy_energy."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (0.0, 0.0, 1.0))

    energy = cubic_anisotropy_energy(
        magnetisation,
        _CUBIC_U1,
        _CUBIC_U2,
        _CUBIC_K1,
        _CUBIC_K2,
        _CUBIC_K3,
        unit_length=1e-9,
    )

    expected = _legacy_cubic_anisotropy_energy_density((0.0, 0.0, 1.0)) * 1e-27
    assert np.isclose(energy, expected, rtol=1e-6)


def test_cubic_anisotropy_energy_rejects_wrong_axis_shape():
    """Axis shape mistakes should be explicit, not silently broadcast."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="must be 3-vectors"):
        cubic_anisotropy_energy(magnetisation, (0, 1), _CUBIC_U2, _CUBIC_K1)


def test_cubic_anisotropy_energy_rejects_zero_axis():
    """A zero axis would silently make the cubic-anisotropy definition invalid."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="must be non-zero"):
        cubic_anisotropy_energy(magnetisation, (0, 0, 0), _CUBIC_U2, _CUBIC_K1)


def test_cubic_anisotropy_energy_rejects_non_positive_unit_length():
    """Invalid physical length scaling should fail before form assembly."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="unit_length must be positive"):
        cubic_anisotropy_energy(
            magnetisation, _CUBIC_U1, _CUBIC_U2, _CUBIC_K1, unit_length=0.0
        )


def test_cubic_anisotropy_energy_all_zero_constants_short_circuits():
    """All-zero K1/K2/K3 should not build a degenerate UFL form, and should
    not require valid axes either (mirrors the other energies' short-circuit
    behaviour)."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (0.0, 0.0, 1.0))

    energy = cubic_anisotropy_energy(
        magnetisation, (0, 0, 0), (0, 0, 0), 0.0, 0.0, 0.0
    )
    assert np.isclose(energy, 0.0)
