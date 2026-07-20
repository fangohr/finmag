"""Serial/MPI evidence for the Task 5 lumped-box energy contract.

This probe deliberately lives outside ``src``.  It freezes the DOLFINx 0.10
assembly ordering needed by the direct production port: reverse-add ghost
contributions, forward-scatter the completed vector, and only then read or
divide owned coefficients.
"""

import json

import numpy as np
import ufl
from dolfinx import fem, la, mesh
from mpi4py import MPI
from ufl import TestFunction, dx, grad, inner


MU0 = 4.0 * np.pi * 1e-7


def _assembled_vector(expression):
    """Assemble a linear form with complete owner and ghost values."""
    vector = fem.assemble_vector(fem.form(expression))
    vector.scatter_reverse(la.InsertMode.add)
    vector.scatter_forward()
    return vector


def _owned_size(function_space):
    dofmap = function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def _nodal_volume(function_space):
    """Return dimensionless lumped volumes, including refreshed ghosts."""
    value_shape = function_space.ufl_element().reference_value_shape
    value_size = int(np.prod(value_shape)) if value_shape else 1
    test = TestFunction(function_space)
    if value_size == 1:
        expression = test * dx
    else:
        ones = fem.Constant(function_space.mesh, np.ones(value_size))
        expression = inner(test, ones) * dx
    return _assembled_vector(expression)


def _box_field(energy_density, magnetisation, saturation_magnetisation):
    """Return a ghost-refreshed Function for ``-dE/(mu0 Ms dm)``."""
    function_space = magnetisation.function_space
    test = TestFunction(function_space)
    derivative = (-1.0 / MU0) * ufl.derivative(
        energy_density / saturation_magnetisation * dx,
        magnetisation,
        test,
    )
    assembled = _assembled_vector(derivative)
    volumes = _nodal_volume(function_space)
    owned = _owned_size(function_space)

    result = fem.Function(function_space)
    result.x.array[:owned] = assembled.array[:owned] / volumes.array[:owned]
    result.x.scatter_forward()
    return result


def _energy(energy_density, unit_length):
    domain = energy_density.ufl_domain().ufl_cargo()
    local_value = fem.assemble_scalar(fem.form(energy_density * dx))
    mesh_value = domain.comm.allreduce(local_value, op=MPI.SUM)
    return mesh_value * float(unit_length) ** domain.topology.dim


def _constant_scalar(function_space, value):
    result = fem.Function(function_space)
    result.interpolate(lambda x: np.full(x.shape[1], float(value)))
    result.x.scatter_forward()
    return result


def _constant_vector(function_space, value):
    value = np.asarray(value, dtype=np.float64)
    result = fem.Function(function_space)
    result.interpolate(lambda x: np.repeat(value[:, None], x.shape[1], axis=1))
    result.x.scatter_forward()
    return result


def run_probe():
    """Validate the Task 5 box contract on one or two MPI ranks."""
    comm = MPI.COMM_WORLD
    if comm.size not in (1, 2):
        raise RuntimeError("energy box probe expects one or two MPI ranks")

    domain = mesh.create_unit_square(comm, 3, 3)
    scalar_space = fem.functionspace(domain, ("Lagrange", 1))
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    owned_blocks = vector_space.dofmap.index_map.size_local
    ghost_blocks = vector_space.dofmap.index_map.num_ghosts

    scalar_volumes = _nodal_volume(scalar_space)
    scalar_owned = _owned_size(scalar_space)
    global_volume = comm.allreduce(
        float(np.sum(scalar_volumes.array[:scalar_owned])), op=MPI.SUM
    )
    assert np.isclose(global_volume, 1.0)

    vector_volumes = _nodal_volume(vector_space)
    vector_owned = _owned_size(vector_space)
    local_component_volumes = vector_volumes.array[:vector_owned].reshape((-1, 3)).sum(0)
    global_component_volumes = np.zeros(3)
    comm.Allreduce(local_component_volumes, global_component_volumes, op=MPI.SUM)
    assert np.allclose(global_component_volumes, 1.0)
    assert np.all(vector_volumes.array > 0.0)

    saturation = _constant_scalar(scalar_space, 2.5)
    magnetisation = _constant_vector(vector_space, (0.6, 0.0, 0.8))

    applied_value = np.array((1.25, -2.5, 3.75))
    applied = fem.Constant(domain, applied_value)
    zeeman_density = -MU0 * saturation * inner(magnetisation, applied)
    zeeman_field = _box_field(zeeman_density, magnetisation, saturation)
    assert np.allclose(zeeman_field.x.array.reshape((-1, 3)), applied_value)

    unit_length = 2.0e-9
    expected_zeeman = (
        -MU0
        * 2.5
        * np.dot((0.6, 0.0, 0.8), applied_value)
        * unit_length**2
    )
    assert np.isclose(
        _energy(zeeman_density, unit_length), expected_zeeman, rtol=1e-13
    )

    axis_value = np.array((0.0, 0.0, 1.0))
    axis = fem.Constant(domain, axis_value)
    K1 = 4.0
    K2 = 1.5
    alignment = 0.8
    anisotropy_density = K1 * (1.0 - inner(axis, magnetisation) ** 2)
    anisotropy_density -= K2 * inner(axis, magnetisation) ** 4
    anisotropy_field = _box_field(
        anisotropy_density, magnetisation, saturation
    )
    expected_anisotropy_field = (
        (2.0 * K1 * alignment + 4.0 * K2 * alignment**3)
        / (MU0 * 2.5)
        * axis_value
    )
    assert np.allclose(
        anisotropy_field.x.array.reshape((-1, 3)),
        expected_anisotropy_field,
        rtol=1e-13,
        atol=1e-10,
    )
    expected_anisotropy_energy = (
        K1 * (1.0 - alignment**2) - K2 * alignment**4
    ) * unit_length**2
    assert np.isclose(
        _energy(anisotropy_density, unit_length),
        expected_anisotropy_energy,
        rtol=1e-13,
    )

    linear_m = fem.Function(vector_space)
    linear_m.interpolate(
        lambda x: np.vstack((x[0], 2.0 * x[1], np.zeros(x.shape[1])))
    )
    linear_m.x.scatter_forward()
    A = 5.0

    def exchange_density(length):
        return A / float(length) ** 2 * inner(grad(linear_m), grad(linear_m))

    exchange_length = 0.25
    assert np.isclose(_energy(exchange_density(exchange_length), exchange_length), 25.0)

    exchange_field_1 = _box_field(exchange_density(1.0), linear_m, saturation)
    exchange_field_2 = _box_field(exchange_density(2.0), linear_m, saturation)
    assert np.max(np.abs(exchange_field_1.x.array)) > 0.0
    assert np.allclose(exchange_field_2.x.array, exchange_field_1.x.array / 4.0)

    weighted = (
        exchange_field_1.x.array[:vector_owned].reshape((-1, 3))
        * vector_volumes.array[:vector_owned].reshape((-1, 3))
    ).sum(0)
    global_weighted = np.zeros(3)
    comm.Allreduce(weighted, global_weighted, op=MPI.SUM)
    assert np.allclose(global_weighted, 0.0, atol=1e-8)

    owned_by_rank = comm.gather(owned_blocks, root=0)
    ghosts_by_rank = comm.gather(ghost_blocks, root=0)
    if comm.rank == 0:
        print(
            json.dumps(
                {
                    "ghost_blocks_by_rank": ghosts_by_rank,
                    "global_volume": global_volume,
                    "mpi_size": comm.size,
                    "owned_blocks_by_rank": owned_by_rank,
                    "status": "ok",
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    run_probe()
