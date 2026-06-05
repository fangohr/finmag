"""Small DOLFINx-only micromagnetic prototype helpers.

This module is intentionally kept under ``dev/dolfinx`` instead of
``src/finmag``. M4 is still a reduced-scope prototype lane, so these helpers let
us exercise DOLFINx mesh/function/form semantics without changing the green
legacy FEniCS-2019 M2/M3 code paths. [Codex gpt-5.5 high]
"""

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from ufl import dx, grad, inner


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
