"""DOLFINx energy-interaction foundation using lumped box assembly.

Task 16 adds spatially varying material coefficients (``scalar_coefficient`` /
``axis_coefficient`` accept callables/Fields/Functions), spatially varying
``Ms``, and an optional region measure on ``compute_energy``. [Claude Opus 4.8]
"""

import numbers
from math import pi

import numpy as np
import ufl
from aeon import timer
from dolfinx import fem, la
from mpi4py import MPI
from ufl import TestFunction, dx, inner

from finmag.field import Field, associated_scalar_space, owned_raw_to_blocked


mu0 = 4.0 * pi * 1e-7


class EnergyBase:
    """Base class for DOLFINx energy terms.

    The first direct DOLFINx slice supports the historical lumped
    ``box-assemble`` algorithm only. Setup and every assembly/reduction method
    are collective over the magnetisation mesh communicator.
    """

    _supported_methods = ("box-assemble",)
    _deferred_methods = (
        "box-matrix-numpy",
        "box-matrix-petsc",
        "project",
        "direct",
    )

    def __init__(self, method="box-assemble", in_jacobian=False):
        if method in self._deferred_methods:
            raise NotImplementedError(
                "energy method {!r} is not yet ported to DOLFINx; "
                "use 'box-assemble'".format(method)
            )
        if method not in self._supported_methods:
            raise ValueError(
                "unsupported energy method {!r}; supported methods are {}".format(
                    method, self._supported_methods
                )
            )
        self.in_jacobian = bool(in_jacobian)
        self.method = method

    def setup(self, E_integrand, m, Ms, unit_length=1.0):
        """Bind an energy-density expression to ``m`` and constant positive ``Ms``.

        ``E_integrand`` is expressed per physical volume while coordinates are
        mesh units. Total energy therefore gains ``unit_length**mesh_dim``;
        box-field division uses unscaled mesh-coordinate nodal volumes so that
        physical volume factors cancel exactly.
        """
        if not isinstance(m, Field):
            raise TypeError("m must be a finmag.Field")
        if not isinstance(Ms, Field):
            raise TypeError("Ms must be a finmag.Field")
        if m.is_scalar_field():
            raise ValueError("m must be a vector Field")
        _require_cg1_magnetisation(m)
        if not Ms.is_scalar_field():
            raise ValueError("Ms must be a scalar Field")
        if m.mesh() is not Ms.mesh():
            raise ValueError("m and Ms must use the same mesh")

        unit_length = float(unit_length)
        if not np.isfinite(unit_length) or unit_length <= 0.0:
            raise ValueError("unit_length must be a positive finite number")
        ms_values = Ms.as_array()
        local_invalid_ms = bool(
            np.any(~np.isfinite(ms_values)) or np.any(ms_values <= 0.0)
        )
        invalid_ms = m.mesh().comm.allreduce(local_invalid_ms, op=MPI.LOR)
        if invalid_ms:
            raise ValueError("Ms must be positive")
        # Spatially varying Ms is supported (Task 16): legacy ``EnergyBase``
        # used ``Ms.f`` verbatim in the UFL derivative regardless of its
        # function space, so any positive scalar Ms Field (DG0 or CG1) is
        # accepted here and enters through ``E_integrand / Ms.f`` below.
        if hasattr(self, "E_density_function"):
            del self.E_density_function

        self.E_integrand = E_integrand
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length
        self.dim = m.mesh_dim()
        self.S1 = associated_scalar_space(m.functionspace)

        scalar_test = TestFunction(self.S1)
        vector_test = TestFunction(m.functionspace)
        self.E = E_integrand * dx
        self.nodal_E = E_integrand * scalar_test * dx
        self.dE_dm = (-1.0 / mu0) * ufl.derivative(
            E_integrand / Ms.f * dx,
            m.f,
            vector_test,
        )

        self.nodal_volume_S1 = _nodal_volume_owned(self.S1)
        self.nodal_volume_S3 = _nodal_volume_owned(m.functionspace)
        return self

    @timer.method
    def compute_energy(self, dx=None):
        """Collectively return total energy in joules.

        With ``dx=None`` (default) the energy is integrated over the whole
        mesh. A restricted UFL measure (e.g. ``sim.region_measure(region_id)``)
        integrates only that region; region energies sum to the whole-mesh
        energy. This mirrors the legacy ``Zeeman.compute_energy(dx=...)``
        region-accounting contract, extended consistently to every box-energy
        (``test_energies_in_regions`` is the behavioral invariant).
        """
        if dx is None:
            form = self.E
        else:
            form = self.E_integrand * dx
        mesh_energy = _assemble_scalar(self.m.mesh(), form)
        return mesh_energy * self.unit_length**self.dim

    @timer.method
    def energy_density(self):
        """Collectively return owned lumped nodal energy-density values."""
        nodal_energy = _assemble_vector_owned(self.nodal_E, self.S1)
        return nodal_energy / self.nodal_volume_S1

    def energy_density_function(self):
        """Return the current lumped density as a ghost-refreshed Function."""
        if not hasattr(self, "E_density_function"):
            name = "{}_energy_density".format(
                getattr(self, "name", self.__class__.__name__)
            )
            self.E_density_function = Field(self.S1, name=name)
        self.E_density_function.from_array(self.energy_density())
        return self.E_density_function.f

    @timer.method
    def compute_field(self):
        """Collectively return the field in legacy component-blocked order.

        Public field-array surface (Task 31): the raw owned box-assembled
        coefficients (:meth:`_compute_field_raw`, node-interleaved backend
        order) are converted once here to the legacy component-blocked,
        owned-vertex-coordinate-ordered ``xxx`` view via the shared
        ``owned_raw_to_blocked`` helper -- the same ordering ``sim.m`` returns.
        """
        return owned_raw_to_blocked(
            self.m.functionspace, self._compute_field_raw()
        )

    def _compute_field_raw(self):
        """Flat rank-local owned box-assembled field (backend/interleaved order).

        Internal, pre-conversion layout used by the public blocked
        ``compute_field`` and by ``average_field`` (an order-invariant nodal
        mean). Kept raw because it pairs elementwise with the raw
        ``nodal_volume_S3`` weights.
        """
        derivative = _assemble_vector_owned(self.dE_dm, self.m.functionspace)
        return derivative / self.nodal_volume_S3

    def average_field(self):
        """Collectively return the legacy arithmetic nodal field average."""
        values = self._compute_field_raw().reshape((-1, self.m.value_dim()))
        local_sum = np.sum(values, axis=0)
        global_sum = np.zeros_like(local_sum)
        self.m.mesh().comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        global_count = self.m.mesh().comm.allreduce(values.shape[0], op=MPI.SUM)
        return global_sum / global_count


def _assembled_vector(expression):
    """Assemble a linear form with complete owner and ghost values."""
    vector = fem.assemble_vector(fem.form(expression))
    vector.scatter_reverse(la.InsertMode.add)
    vector.scatter_forward()
    return vector


def _assemble_vector_owned(expression, function_space):
    vector = _assembled_vector(expression)
    return vector.array[: _owned_scalar_dofs(function_space)].copy()


def _assemble_scalar(domain, expression):
    local_value = fem.assemble_scalar(fem.form(expression))
    return domain.comm.allreduce(local_value, op=MPI.SUM)


def _nodal_volume_owned(function_space):
    value_shape = function_space.ufl_element().reference_value_shape
    value_size = int(np.prod(value_shape)) if value_shape else 1
    test = TestFunction(function_space)
    if value_size == 1:
        expression = test * dx
    else:
        ones = fem.Constant(function_space.mesh, np.ones(value_size))
        expression = inner(test, ones) * dx
    volumes = _assemble_vector_owned(expression, function_space)
    local_invalid = bool(np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0))
    invalid = function_space.mesh.comm.allreduce(local_invalid, op=MPI.LOR)
    if invalid:
        raise ValueError("box assembly requires positive owned nodal volumes")
    return volumes


def _owned_scalar_dofs(function_space):
    dofmap = function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def _is_string_expression(value):
    return isinstance(value, str) or (
        isinstance(value, (tuple, list))
        and any(isinstance(item, str) for item in value)
    )


def scalar_coefficient(value, name):
    """Normalise a scalar material coefficient for a coefficient Field.

    Accepts (Task 16) a plain finite number, a ``dolfinx.fem.Constant``, a
    length-1 array, OR a spatially varying coefficient: a Python callable, a
    :class:`~finmag.field.Field`, or a ``dolfinx.fem.Function``. Constant
    numbers are validated finite and returned as ``float``; spatially varying
    values are returned unchanged for :meth:`finmag.field.Field.set` to place
    into the coefficient's function space (DG0 or CG1, per the owning class).

    Legacy string ``Expression`` coefficients raise ``NotImplementedError`` by
    name -- DOLFINx has no ``Expression`` object, so pass a callable instead
    (a documented deviation shared with the rest of the port).
    """
    if _is_string_expression(value):
        raise NotImplementedError(
            "legacy string Expression {} is not supported; pass a "
            "callable".format(name)
        )
    if isinstance(value, (Field, fem.Function)) or callable(value):
        return value
    if isinstance(value, fem.Constant):
        value = value.value
    if isinstance(value, numbers.Real):
        result = float(value)
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                "{} constant array must have a single element; pass a "
                "callable/Field for a spatially varying value".format(name)
            )
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def axis_coefficient(value, name, normalise=True):
    """Normalise a 3-vector axis coefficient for a coefficient Field.

    A constant axis (3-tuple/list/array of numbers) is validated finite and,
    when ``normalise`` is true, returned as a unit vector (restoring the legacy
    cosine contract). A spatially varying axis -- a callable, ``Field`` or
    ``dolfinx.fem.Function`` -- is returned unchanged (legacy interpolated the
    axis field as given, without renormalising it). Legacy string Expressions
    raise ``NotImplementedError`` by name.
    """
    if _is_string_expression(value):
        raise NotImplementedError(
            "legacy string Expression {} is not supported; pass a "
            "callable".format(name)
        )
    if isinstance(value, (Field, fem.Function)) or callable(value):
        return value
    if isinstance(value, fem.Constant):
        value = value.value
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("{} must be a three-component vector".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain finite values".format(name))
    if not normalise:
        return array
    norm = np.linalg.norm(array)
    if norm == 0.0:
        raise ValueError("{} must be non-zero".format(name))
    return array / norm


def _require_cg1_magnetisation(m):
    element = m.functionspace.ufl_element()
    if (
        m.value_dim() != 3
        or m.functionspace.dofmap.index_map_bs != 3
        or element.degree != 1
        or element.family_name not in ("P", "Lagrange")
    ):
        raise NotImplementedError(
            "the first DOLFINx box-energy slice requires a blocked "
            "three-component CG1 magnetisation space"
        )
