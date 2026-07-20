"""DOLFINx energy-interaction foundation using lumped box assembly."""

from math import pi

import numpy as np
import ufl
from aeon import timer
from dolfinx import fem, la
from mpi4py import MPI
from ufl import TestFunction, dx, inner

from finmag.field import Field, associated_scalar_space


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
        if not Ms.is_constant():
            raise NotImplementedError(
                "spatially varying Ms is deferred from the first DOLFINx "
                "energy slice"
            )
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
    def compute_energy(self):
        """Collectively return total energy in joules."""
        mesh_energy = _assemble_scalar(self.m.mesh(), self.E)
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
        """Collectively return flat rank-local owned field coefficients."""
        derivative = _assemble_vector_owned(self.dE_dm, self.m.functionspace)
        return derivative / self.nodal_volume_S3

    def average_field(self):
        """Collectively return the legacy arithmetic nodal field average."""
        values = self.compute_field().reshape((-1, self.m.value_dim()))
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
