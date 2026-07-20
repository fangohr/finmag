"""Two-rank production Field gate; run as a module under ``mpiexec``."""

import json

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.field import Field


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        raise RuntimeError("field_dolfinx_mpi_probe requires exactly two ranks")

    domain = mesh.create_unit_square(comm, 3, 3)
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    field = Field(
        vector_space,
        lambda x: np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] + x[1])),
    )

    owned_blocks = vector_space.dofmap.index_map.size_local
    ghost_blocks = vector_space.dofmap.index_map.num_ghosts
    block_size = vector_space.dofmap.index_map_bs
    assert block_size == 3
    assert field.as_array().shape == (owned_blocks * block_size,)
    assert field.local_array_with_ghosts().shape == (
        (owned_blocks + ghost_blocks) * block_size,
    )
    with np.testing.assert_raises_regex(ValueError, "owned"):
        field.from_array(field.local_array_with_ghosts())

    assert np.allclose(field.average(), (1.5, 2.5, 4.0))
    coordinates, values = field.coords_and_values()
    expected = np.column_stack(
        (
            1.0 + coordinates[:, 0],
            2.0 + coordinates[:, 1],
            3.0 + coordinates[:, 0] + coordinates[:, 1],
        )
    )
    assert np.allclose(values, expected)
    assert np.allclose(
        field.get_ordered_numpy_array_xyz().reshape((-1, 3)), expected
    )
    assert np.allclose(
        field.get_ordered_numpy_array_xxx().reshape((3, -1)), expected.T
    )

    copied = Field(vector_space, field)
    assert copied.allclose(field)
    if comm.rank == 0:
        copied.f.x.array[0] += 1.0
    copied.f.x.scatter_forward()
    assert not copied.allclose(field)

    index_map = vector_space.dofmap.index_map
    local_blocks = np.arange(owned_blocks + ghost_blocks, dtype=np.int32)
    global_blocks = index_map.local_to_global(local_blocks)
    raw = np.column_stack(
        tuple(10.0 * global_blocks[:owned_blocks] + component for component in range(3))
    ).reshape(-1)
    field.from_array(raw)
    expected_local = np.column_stack(
        tuple(10.0 * global_blocks + component for component in range(3))
    )
    assert np.array_equal(
        field.local_array_with_ghosts().reshape((-1, 3)), expected_local
    )

    field.set(
        lambda x: np.vstack((1.0 + x[0], 2.0 + x[1], 3.0 + x[0] + x[1]))
    ).normalise()
    local_coordinates = vector_space.tabulate_dof_coordinates()
    local_unnormalised = np.column_stack(
        (
            1.0 + local_coordinates[:, 0],
            2.0 + local_coordinates[:, 1],
            3.0 + local_coordinates[:, 0] + local_coordinates[:, 1],
        )
    )
    expected_local = local_unnormalised / np.linalg.norm(
        local_unnormalised, axis=1
    )[:, None]
    assert np.allclose(
        field.local_array_with_ghosts().reshape((-1, 3)), expected_local
    )
    coordinates, values = field.coords_and_values()
    unnormalised = np.column_stack(
        (
            1.0 + coordinates[:, 0],
            2.0 + coordinates[:, 1],
            3.0 + coordinates[:, 0] + coordinates[:, 1],
        )
    )
    assert np.allclose(
        values, unnormalised / np.linalg.norm(unnormalised, axis=1)[:, None]
    )

    zero = Field(vector_space, (1.0, 0.0, 0.0))
    if comm.rank == 0:
        zero.f.x.array[:3] = 0.0
    caught = False
    try:
        zero.normalise()
    except ValueError as error:
        caught = "zero vector" in str(error)
    assert comm.allreduce(caught, op=MPI.LAND)

    gathered_coordinates = comm.gather(coordinates, root=0)
    gathered_owned = comm.gather(owned_blocks, root=0)
    gathered_ghosts = comm.gather(ghost_blocks, root=0)
    if comm.rank == 0:
        all_coordinates = np.vstack(gathered_coordinates)
        unique_coordinates = np.unique(np.round(all_coordinates[:, :2], 12), axis=0)
        assert unique_coordinates.shape == (16, 2)
        print(
            json.dumps(
                {
                    "global_vertices": 16,
                    "owned_blocks_by_rank": gathered_owned,
                    "ghost_blocks_by_rank": gathered_ghosts,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
