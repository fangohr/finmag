"""Bounded serial/MPI evidence for DOLFINx Field ownership semantics."""

import json

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from dev.dolfinx.field_adapter import DOLFINxField
from dev.dolfinx.prototype import average_nodal_vector, vector_function_space


def _vector_values(x):
    return np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] + x[1]))


def _normalised_vector_values(x):
    values = _vector_values(x).T
    return values / np.linalg.norm(values, axis=1)[:, None]


def run_probe():
    """Assert the owned/ghost contract on one or two MPI ranks."""
    comm = MPI.COMM_WORLD
    if comm.size not in (1, 2):
        raise RuntimeError("field ownership probe expects one or two MPI ranks")

    domain = mesh.create_unit_square(comm, 3, 3)
    function_space = vector_function_space(domain)
    index_map = function_space.dofmap.index_map
    block_size = function_space.dofmap.index_map_bs
    owned_blocks = index_map.size_local
    ghost_blocks = index_map.num_ghosts

    field = DOLFINxField(function_space)
    field.set(_vector_values)

    assert field.f.x.array.size == (owned_blocks + ghost_blocks) * block_size
    assert field.as_array().shape == (owned_blocks * block_size,)
    assert field.nodal_values().shape == (owned_blocks, block_size)
    assert field.local_values_with_ghosts().shape == (
        owned_blocks + ghost_blocks,
        block_size,
    )

    copied = DOLFINxField(function_space)
    copied.from_array(field.as_array())
    assert np.allclose(
        copied.local_values_with_ghosts(), field.local_values_with_ghosts()
    )
    if ghost_blocks:
        try:
            copied.from_array(field.f.x.array.copy())
        except ValueError as error:
            assert "raw dof array (owned)" in str(error)
        else:
            raise AssertionError("from_array accepted ghost-appended storage")

    assert copied.allclose(field)
    if comm.rank == 0:
        copied.f.x.array[0] += 1.0
    copied.f.x.scatter_forward()
    assert not copied.allclose(field)

    # Form assembly integrates owned cells and performs one explicit reduction;
    # ghost values are not counted as additional volume. [Codex GPT-5.6]
    assert np.allclose(field.average(), (1.5, 2.5, 4.0), atol=1e-12)

    nodal_field = DOLFINxField(function_space)
    nodal_field.set(
        lambda x: np.vstack(
            (1.0 + x[0] ** 2, 2.0 + x[1] ** 2, 3.0 + x[0] + x[1])
        )
    )
    expected_nodal_average = np.array((1.0 + 7.0 / 18.0, 2.0 + 7.0 / 18.0, 4.0))
    assert np.allclose(
        average_nodal_vector(nodal_field.f), expected_nodal_average, atol=1e-12
    )

    field.normalise()
    dof_coords = function_space.tabulate_dof_coordinates()
    expected_local = _normalised_vector_values(dof_coords.T)
    assert np.allclose(field.local_values_with_ghosts(), expected_local)
    assert np.allclose(np.linalg.norm(field.nodal_values(), axis=1), 1.0)

    zero_field = DOLFINxField(function_space, value=(1.0, 0.0, 0.0))
    if comm.rank == 0:
        zero_field.f.x.array[:block_size] = 0.0
    zero_field.f.x.scatter_forward()
    try:
        zero_field.normalise()
    except ValueError as error:
        assert "zero vector" in str(error)
    else:
        raise AssertionError("normalise accepted a zero owned vector")

    coords, values = field.coords_and_values()
    num_owned_vertices = domain.geometry.index_map().size_local
    assert coords.shape == (num_owned_vertices, domain.geometry.dim)
    assert values.shape == (num_owned_vertices, block_size)
    assert np.allclose(values, _normalised_vector_values(coords.T))

    xyz = field.get_ordered_numpy_array_xyz()
    xxx = field.get_ordered_numpy_array_xxx()
    assert np.allclose(xyz, values.reshape(-1))
    assert np.allclose(xxx, values.T.reshape(-1))

    component_round_trip = DOLFINxField(function_space)
    component_round_trip.set_with_ordered_numpy_array_xxx(xxx)
    assert np.allclose(
        component_round_trip.local_values_with_ghosts(),
        field.local_values_with_ghosts(),
    )

    gathered = comm.gather((coords, values), root=0)
    owned_blocks_by_rank = comm.gather(owned_blocks, root=0)
    ghost_blocks_by_rank = comm.gather(ghost_blocks, root=0)
    if comm.rank == 0:
        global_coords = np.concatenate([item[0] for item in gathered])
        global_values = np.concatenate([item[1] for item in gathered])
        assert global_coords.shape[0] == domain.geometry.index_map().size_global
        assert np.unique(np.round(global_coords, decimals=12), axis=0).shape == (
            global_coords.shape[0],
            domain.geometry.dim,
        )
        assert np.allclose(global_values, _normalised_vector_values(global_coords.T))
        print(
            json.dumps(
                {
                    "global_owned_vertices": int(global_coords.shape[0]),
                    "mpi_size": comm.size,
                    "owned_blocks_by_rank": owned_blocks_by_rank,
                    "ghost_blocks_by_rank": ghost_blocks_by_rank,
                    "status": "ok",
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    run_probe()
