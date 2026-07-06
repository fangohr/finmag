"""Small DOLFINx-only micromagnetic prototype helpers.

This module is intentionally kept under ``dev/dolfinx`` instead of
``src/finmag``. M4 is still a reduced-scope prototype lane, so these helpers let
us exercise DOLFINx mesh/function/form semantics without changing the green
legacy FEniCS-2019 M2/M3 code paths. [Codex gpt-5.5 high]
"""

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from ufl import curl, dx, grad, inner


MU0 = 4 * np.pi * 1e-7


def vector_function_space(domain, components=3, degree=1):
    """Create a DOLFINx vector-valued Lagrange function space."""
    return fem.functionspace(domain, ("Lagrange", degree, (components,)))


def constant_vector_function(function_space, values):
    """Interpolate a constant vector into a DOLFINx vector function space."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("constant vector values must be one-dimensional")

    value_size = int(np.prod(function_space.element.value_shape))
    if values.size != value_size:
        raise ValueError(
            "constant vector has %d components, but the function space expects %d"
            % (values.size, value_size)
        )

    result = fem.Function(function_space)

    def evaluate_constant(x):
        return np.repeat(values[:, None], x.shape[1], axis=1)

    result.interpolate(evaluate_constant)
    return result


def nodal_vector_values(vector_function):
    """Return vector-valued function dofs as an ``(n_dofs, components)`` array."""
    value_size = int(np.prod(vector_function.function_space.element.value_shape))
    return vector_function.x.array.reshape((-1, value_size))


def average_nodal_vector(vector_function):
    """Return the MPI-reduced average vector over local nodal values.

    This is a prototype output helper, not a replacement for volume-averaged
    finite-element integration. It is sufficient for the first M4 JSON example
    because the example uses a spatially uniform magnetisation. [Codex
    gpt-5.5 high]
    """
    values = nodal_vector_values(vector_function)
    local_sum = np.sum(values, axis=0)
    local_count = np.array([values.shape[0]], dtype=np.float64)

    comm = vector_function.function_space.mesh.comm
    global_sum = np.zeros_like(local_sum)
    global_count = np.zeros_like(local_count)
    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_count, global_count, op=MPI.SUM)

    return global_sum / global_count[0]


def zeeman_energy(magnetisation, field, saturation_magnetisation, unit_length=1.0):
    """Compute ``-mu0 * Ms * integral(m . H)`` for a DOLFINx field.

    This is deliberately the smallest useful energy term: it verifies DOLFINx
    form assembly, MPI reduction, and unit-length scaling before we attempt the
    more invasive exchange/time-integration work. [Codex gpt-5.5 high]
    """
    if unit_length <= 0:
        raise ValueError("unit_length must be positive")

    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** domain.geometry.dim
    applied_field = fem.Constant(domain, np.asarray(field, dtype=np.float64))

    local_energy = fem.assemble_scalar(
        fem.form(
            -MU0
            * float(saturation_magnetisation)
            * physical_measure_scale
            * inner(magnetisation, applied_field)
            * dx
        )
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)


def uniaxial_anisotropy_energy(
    magnetisation, axis, anisotropy_constant, unit_length=1.0
):
    """Compute ``K * integral(1 - (m . axis)^2)`` for a DOLFINx field.

    This second prototype energy keeps the scope intentionally narrow: a
    constant uniaxial axis and the same mesh-measure scaling convention as the
    Zeeman helper. It exercises a nonlinear DOLFINx form before we move to
    exchange gradients or time integration. [Codex gpt-5.5 high]
    """
    if unit_length <= 0:
        raise ValueError("unit_length must be positive")
    if anisotropy_constant == 0:
        return 0.0

    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("anisotropy axis must be non-zero")

    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** domain.geometry.dim
    unit_axis = fem.Constant(domain, axis / axis_norm)

    local_energy = fem.assemble_scalar(
        fem.form(
            float(anisotropy_constant)
            * physical_measure_scale
            * (1 - inner(magnetisation, unit_axis) ** 2)
            * dx
        )
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)


def llg_rhs(magnetisation_values, effective_field_values, gamma=1.0, alpha=0.0):
    """Evaluate the normalized LLG right-hand side at nodal vector values.

    The prototype uses nodal arrays on purpose: this keeps the first M4 time
    integrator transparent and testable before we decide how a Finmag-facing
    DOLFINx time-stepping API should assemble or project effective fields.
    [Codex gpt-5.5 high]
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    magnetisation_values = np.asarray(magnetisation_values, dtype=np.float64)
    effective_field_values = np.asarray(effective_field_values, dtype=np.float64)

    precession = np.cross(magnetisation_values, effective_field_values)
    damping = np.cross(magnetisation_values, precession)
    return -float(gamma) / (1 + float(alpha) ** 2) * (
        precession + float(alpha) * damping
    )


def explicit_llg_step(
    magnetisation, effective_field, dt, gamma=1.0, alpha=0.0, normalise=True
):
    """Advance a DOLFINx magnetisation field by one explicit LLG step.

    This is a deliberately small time-integration workflow rather than a
    production solver. It supports a constant effective field, updates the
    DOLFINx function in place, and optionally renormalises nodal magnetisation
    vectors after the explicit Euler step. [Codex gpt-5.5 high]
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    values = nodal_vector_values(magnetisation)
    effective_field = np.asarray(effective_field, dtype=np.float64)
    if effective_field.shape != (values.shape[1],):
        raise ValueError(
            "effective field has %d components, but magnetisation expects %d"
            % (effective_field.size, values.shape[1])
        )

    field_values = np.repeat(effective_field[None, :], values.shape[0], axis=0)
    updated = values + float(dt) * llg_rhs(
        values,
        field_values,
        gamma=gamma,
        alpha=alpha,
    )

    if normalise:
        norms = np.linalg.norm(updated, axis=1)
        if np.any(norms == 0):
            raise ValueError("cannot normalise zero magnetisation vector")
        updated = updated / norms[:, None]

    values[:] = updated
    magnetisation.x.scatter_forward()
    return magnetisation


def exchange_energy(magnetisation, exchange_constant, unit_length=1.0):
    """Compute ``A * integral(grad(m) : grad(m))`` for a DOLFINx field.

    This is the first gradient-based M4 prototype energy. The unit scaling uses
    mesh-coordinate gradients and the coordinate change ``x_phys = unit_length *
    x_mesh``, so the integral scales as ``unit_length ** (dim - 2)``. Keeping
    this convention explicit is important before any Finmag-facing DOLFINx API
    is designed. [Codex gpt-5.5 high]
    """
    if unit_length <= 0:
        raise ValueError("unit_length must be positive")
    if exchange_constant == 0:
        return 0.0

    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** (domain.geometry.dim - 2)

    local_energy = fem.assemble_scalar(
        fem.form(
            float(exchange_constant)
            * physical_measure_scale
            * inner(grad(magnetisation), grad(magnetisation))
            * dx
        )
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)


def dmi_energy(magnetisation, dmi_constant, unit_length=1.0):
    """Compute ``D * integral(m . curl(m))`` for a DOLFINx field.

    This mirrors the bulk (T-symmetry) 3D Dzyaloshinskii-Moriya term used by
    the legacy ``finmag.energies.dmi.DMI`` class, i.e. ``D * inner(m,
    curl(m))`` (the ``dmi_type='auto'`` case on a 3D mesh in
    ``finmag.util.helpers.times_curl``). Interfacial and 1D/2D DMI variants
    are not covered by this reduced prototype yet. [GitHub Copilot / Claude
    Sonnet 5]

    The scaling convention matches the other prototype energies: DOLFINx
    integrates over mesh coordinates, and the curl introduces a single
    spatial derivative, so the integral scales as
    ``unit_length ** (dim - 1)``.
    """
    if unit_length <= 0:
        raise ValueError("unit_length must be positive")
    if dmi_constant == 0:
        return 0.0

    domain = magnetisation.function_space.mesh
    if domain.geometry.dim != 3:
        raise ValueError("dmi_energy currently only supports 3D meshes")

    physical_measure_scale = float(unit_length) ** (domain.geometry.dim - 1)

    local_energy = fem.assemble_scalar(
        fem.form(
            float(dmi_constant)
            * physical_measure_scale
            * inner(magnetisation, curl(magnetisation))
            * dx
        )
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)
