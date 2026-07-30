"""Small DOLFINx-only micromagnetic prototype helpers.

This module is intentionally kept under ``dev/dolfinx`` instead of
``src/finmag``. M4 is still a reduced-scope prototype lane, so these helpers let
us exercise DOLFINx mesh/function/form semantics without changing the green
legacy FEniCS-2019 M2/M3 code paths. [Codex gpt-5.5 high]
"""

import numpy as np
import ufl
from dolfinx import fem, la
from mpi4py import MPI
from ufl import TestFunction, curl, dx, grad, inner


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
    """Return the MPI-reduced average vector over globally owned nodal values.

    This is a prototype output helper, not a replacement for volume-averaged
    finite-element integration. It is sufficient for the first M4 JSON example
    because the example uses a spatially uniform magnetisation. [Codex
    gpt-5.5 high]
    """
    # DOLFINx local arrays append ghosts; reducing all rows weights shared
    # vertices once per rank and changes nonuniform averages. [Codex GPT-5.6]
    owned_blocks = vector_function.function_space.dofmap.index_map.size_local
    values = nodal_vector_values(vector_function)[:owned_blocks]
    local_sum = np.sum(values, axis=0)
    local_count = np.array([values.shape[0]], dtype=np.float64)

    comm = vector_function.function_space.mesh.comm
    global_sum = np.zeros_like(local_sum)
    global_count = np.zeros_like(local_count)
    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_count, global_count, op=MPI.SUM)

    return global_sum / global_count[0]


def _zeeman_integrand(magnetisation, field, saturation_magnetisation, unit_length):
    """Return the Zeeman energy density UFL expression (no ``*dx``)."""
    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** domain.geometry.dim
    applied_field = fem.Constant(domain, np.asarray(field, dtype=np.float64))
    return (
        -MU0
        * float(saturation_magnetisation)
        * physical_measure_scale
        * inner(magnetisation, applied_field)
    )


def zeeman_energy(magnetisation, field, saturation_magnetisation, unit_length=1.0):
    """Compute ``-mu0 * Ms * integral(m . H)`` for a DOLFINx field.

    This is deliberately the smallest useful energy term: it verifies DOLFINx
    form assembly, MPI reduction, and unit-length scaling before we attempt the
    more invasive exchange/time-integration work. [Codex gpt-5.5 high]
    """
    if unit_length <= 0:
        raise ValueError("unit_length must be positive")

    domain = magnetisation.function_space.mesh
    local_energy = fem.assemble_scalar(
        fem.form(
            _zeeman_integrand(magnetisation, field, saturation_magnetisation, unit_length)
            * dx
        )
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)


def _uniaxial_anisotropy_integrand(magnetisation, unit_axis, anisotropy_constant, unit_length):
    """Return the uniaxial anisotropy energy density UFL expression.

    ``unit_axis`` must already be a normalised ``fem.Constant``; validation
    and normalisation happen in the caller. [GitHub Copilot / Claude Sonnet 5]
    """
    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** domain.geometry.dim
    return (
        float(anisotropy_constant)
        * physical_measure_scale
        * (1 - inner(magnetisation, unit_axis) ** 2)
    )


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
    unit_axis = fem.Constant(domain, axis / axis_norm)

    local_energy = fem.assemble_scalar(
        fem.form(
            _uniaxial_anisotropy_integrand(
                magnetisation, unit_axis, anisotropy_constant, unit_length
            )
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
    _apply_euler_llg_update(values, field_values, dt, gamma, alpha, normalise)
    magnetisation.x.scatter_forward()
    return magnetisation


def _apply_euler_llg_update(values, field_values, dt, gamma, alpha, normalise):
    """Apply one explicit Euler LLG update in place to nodal ``values``.

    Shared by ``explicit_llg_step`` (fixed field) and
    ``effective_field_llg_step`` (energy-derived field), so both steppers use
    exactly the same update/normalisation math. [GitHub Copilot / Claude
    Sonnet 5]
    """
    updated = values + float(dt) * llg_rhs(values, field_values, gamma=gamma, alpha=alpha)

    if normalise:
        norms = np.linalg.norm(updated, axis=1)
        if np.any(norms == 0):
            raise ValueError("cannot normalise zero magnetisation vector")
        updated = updated / norms[:, None]

    values[:] = updated


def _exchange_integrand(magnetisation, exchange_constant, unit_length):
    """Return the exchange energy density UFL expression (no ``*dx``)."""
    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** (domain.geometry.dim - 2)
    return (
        float(exchange_constant)
        * physical_measure_scale
        * inner(grad(magnetisation), grad(magnetisation))
    )


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
    local_energy = fem.assemble_scalar(
        fem.form(
            _exchange_integrand(magnetisation, exchange_constant, unit_length) * dx
        )
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)


def _dmi_integrand(magnetisation, dmi_constant, unit_length):
    """Return the bulk 3D DMI energy density UFL expression (no ``*dx``)."""
    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** (domain.geometry.dim - 1)
    return (
        float(dmi_constant)
        * physical_measure_scale
        * inner(magnetisation, curl(magnetisation))
    )


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

    local_energy = fem.assemble_scalar(
        fem.form(_dmi_integrand(magnetisation, dmi_constant, unit_length) * dx)
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)


def _cubic_anisotropy_integrand(magnetisation, u1, u2, u3, K1, K2, K3, unit_length):
    """Return the cubic anisotropy energy density UFL expression.

    ``u1``/``u2``/``u3`` must already be ``fem.Constant`` axis vectors;
    validation happens in the caller. [GitHub Copilot / Claude Sonnet 5]
    """
    domain = magnetisation.function_space.mesh
    physical_measure_scale = float(unit_length) ** domain.geometry.dim

    a = inner(magnetisation, u1)
    b = inner(magnetisation, u2)
    c = inner(magnetisation, u3)

    integrand = (
        float(K1) * (a**2 * b**2 + a**2 * c**2 + b**2 * c**2)
        + float(K2) * (a**2 * b**2 * c**2)
        + float(K3) * (a**4 * b**4 + a**4 * c**4 + b**4 * c**4)
    )
    return physical_measure_scale * integrand


def cubic_anisotropy_energy(
    magnetisation, u1, u2, K1, K2=0.0, K3=0.0, unit_length=1.0
):
    """Compute the cubic anisotropy energy for constant axes/constants.

    Mirrors legacy ``finmag.energies.cubic_anisotropy.CubicAnisotropy``: with
    ``a = u1.m``, ``b = u2.m``, ``c = u3.m`` and ``u3 = u1 x u2``,

        E = integral[
            K1 * (a^2 b^2 + a^2 c^2 + b^2 c^2)
            + K2 * (a^2 b^2 c^2)
            + K3 * (a^4 b^4 + a^4 c^4 + b^4 c^4)
        ] dx

    Unlike ``uniaxial_anisotropy_energy``, ``u1``/``u2`` are used as given and
    not renormalised, matching the legacy class, which documents them as
    "should be unit vectors" but does not enforce it. This reduced prototype
    only supports spatially constant ``K1``/``K2``/``K3`` and axes; the legacy
    class also supports spatially varying ``Field`` coefficients, which is out
    of scope here. [GitHub Copilot / Claude Sonnet 5]

    The scaling convention matches ``zeeman_energy``/
    ``uniaxial_anisotropy_energy``: no spatial derivatives are involved, so
    the integral scales as ``unit_length ** dim``.
    """
    if unit_length <= 0:
        raise ValueError("unit_length must be positive")
    if K1 == 0 and K2 == 0 and K3 == 0:
        return 0.0

    u1 = np.asarray(u1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)
    if u1.shape != (3,) or u2.shape != (3,):
        raise ValueError("cubic anisotropy axes must be 3-vectors")
    if np.linalg.norm(u1) == 0 or np.linalg.norm(u2) == 0:
        raise ValueError("cubic anisotropy axes must be non-zero")
    u3 = np.cross(u1, u2)

    domain = magnetisation.function_space.mesh
    axis1 = fem.Constant(domain, u1)
    axis2 = fem.Constant(domain, u2)
    axis3 = fem.Constant(domain, u3)

    local_energy = fem.assemble_scalar(
        fem.form(
            _cubic_anisotropy_integrand(
                magnetisation, axis1, axis2, axis3, K1, K2, K3, unit_length
            )
            * dx
        )
    )
    return domain.comm.allreduce(local_energy, op=MPI.SUM)


def nodal_volume(function_space, unit_length=1.0):
    """Return the lumped ("box") nodal volume for each blocked dof.

    This assembles ``TestFunction . ones`` over the whole domain (matching
    legacy Finmag's ``nodal_volume``/box-method convention), giving each dof
    a share of the surrounding cell volumes. It is scaled by
    ``unit_length ** dim`` to convert from mesh-coordinate volume to physical
    volume, consistent with the other prototype energies. [GitHub Copilot /
    Claude Sonnet 5]
    """
    if unit_length <= 0:
        raise ValueError("unit_length must be positive")

    domain = function_space.mesh
    physical_measure_scale = float(unit_length) ** domain.geometry.dim
    value_size = int(np.prod(function_space.element.value_shape)) or 1
    v = TestFunction(function_space)

    if value_size == 1:
        form = fem.form(physical_measure_scale * v * dx)
    else:
        ones = fem.Constant(domain, np.ones(value_size))
        form = fem.form(physical_measure_scale * inner(ones, v) * dx)

    volume = fem.assemble_vector(form)
    volume.scatter_reverse(la.InsertMode.add)
    return volume.array.copy()


def total_energy_integrand(magnetisation, parameters):
    """Return the summed UFL energy density (no ``*dx``) for nonzero terms.

    Only includes terms whose corresponding constant is non-zero, both to
    avoid unnecessary UFL complexity and because ``dmi_energy``'s 3D-mesh
    guard would otherwise reject a 2D mesh even for the default (inert)
    ``dmi_constant=0.0``. Returns ``None`` if every term is zero. [GitHub
    Copilot / Claude Sonnet 5]
    """
    domain = magnetisation.function_space.mesh
    terms = []

    if parameters.exchange_constant != 0:
        terms.append(
            _exchange_integrand(
                magnetisation, parameters.exchange_constant, parameters.unit_length
            )
        )

    terms.append(
        _zeeman_integrand(
            magnetisation,
            parameters.field,
            parameters.saturation_magnetisation,
            parameters.unit_length,
        )
    )

    if parameters.anisotropy_constant != 0:
        axis = np.asarray(parameters.anisotropy_axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("anisotropy axis must be non-zero")
        unit_axis = fem.Constant(domain, axis / axis_norm)
        terms.append(
            _uniaxial_anisotropy_integrand(
                magnetisation, unit_axis, parameters.anisotropy_constant, parameters.unit_length
            )
        )

    if parameters.dmi_constant != 0:
        if domain.geometry.dim != 3:
            raise ValueError("dmi_energy currently only supports 3D meshes")
        terms.append(
            _dmi_integrand(magnetisation, parameters.dmi_constant, parameters.unit_length)
        )

    if (
        parameters.cubic_anisotropy_K1 != 0
        or parameters.cubic_anisotropy_K2 != 0
        or parameters.cubic_anisotropy_K3 != 0
    ):
        u1 = np.asarray(parameters.cubic_anisotropy_u1, dtype=np.float64)
        u2 = np.asarray(parameters.cubic_anisotropy_u2, dtype=np.float64)
        if np.linalg.norm(u1) == 0 or np.linalg.norm(u2) == 0:
            raise ValueError("cubic anisotropy axes must be non-zero")
        u3 = np.cross(u1, u2)
        terms.append(
            _cubic_anisotropy_integrand(
                magnetisation,
                fem.Constant(domain, u1),
                fem.Constant(domain, u2),
                fem.Constant(domain, u3),
                parameters.cubic_anisotropy_K1,
                parameters.cubic_anisotropy_K2,
                parameters.cubic_anisotropy_K3,
                parameters.unit_length,
            )
        )

    return sum(terms[1:], terms[0]) if terms else None


def effective_field_values(magnetisation, parameters):
    """Compute the nodal effective field ``H_eff = -1/(mu0*Ms) * dE/dm``.

    Uses the standard finite-element "box method": assemble the weak-form
    functional derivative of the total energy with respect to ``m`` (a
    linear form, via ``ufl.derivative``), then divide by the lumped nodal
    volume to recover pointwise field values, matching the same box-method
    convention used by legacy ``finmag.energies.energy_base.EnergyBase``
    (``dE_dm = Constant(-1/mu0) * derivative(E_integrand/Ms * dx, m)``,
    divided by nodal volume).

    Validated directly: for a uniform applied field with all other
    constants zero, this recovers ``H_eff == field`` exactly (the defining
    property of the Zeeman effective field), confirming the box-method
    pipeline (derivative assembly, ghost accumulation, volume division) is
    wired correctly before trusting it for less trivial terms. [GitHub
    Copilot / Claude Sonnet 5]
    """
    if parameters.saturation_magnetisation <= 0:
        raise ValueError("saturation_magnetisation must be positive")

    function_space = magnetisation.function_space
    value_size = int(np.prod(function_space.element.value_shape))
    integrand = total_energy_integrand(magnetisation, parameters)
    if integrand is None:
        return np.zeros((nodal_vector_values(magnetisation).shape[0], value_size))

    v = TestFunction(function_space)
    derivative_form = fem.form(
        -1.0 / MU0 * ufl.derivative(integrand * dx, magnetisation, v)
    )
    derivative_vector = fem.assemble_vector(derivative_form)
    derivative_vector.scatter_reverse(la.InsertMode.add)

    volume = nodal_volume(function_space, parameters.unit_length)
    field = derivative_vector.array / parameters.saturation_magnetisation / volume
    return field.reshape((-1, value_size))


def effective_field_llg_step(magnetisation, parameters, dt, gamma=1.0, alpha=1.0, normalise=True):
    """Advance ``magnetisation`` using the energy-derived effective field.

    Unlike ``explicit_llg_step`` (which takes a fixed field argument),
    this computes ``H_eff`` from the current magnetisation state via
    ``effective_field_values`` on every call, so exchange/anisotropy/DMI/
    cubic-anisotropy contributions genuinely drive the dynamics rather than
    only changing the reported energy. It shares the same explicit-Euler
    update/normalisation math as ``explicit_llg_step``. [GitHub Copilot /
    Claude Sonnet 5]
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    values = nodal_vector_values(magnetisation)
    field_values = effective_field_values(magnetisation, parameters)
    _apply_euler_llg_update(values, field_values, dt, gamma, alpha, normalise)
    magnetisation.x.scatter_forward()
    return magnetisation
