"""Two-rank production gate for DOLFINx energy ownership and assembly."""

import json

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.energies import DMI, Exchange, UniaxialAnisotropy, Zeeman
from finmag.energies.energy_base import mu0
from finmag.field import Field


def _assert_function_ghosts_match_owners(field):
    function_space = field.functionspace
    index_map = function_space.dofmap.index_map
    owned = index_map.size_local
    ghosts = index_map.num_ghosts
    block_size = function_space.dofmap.index_map_bs
    local_indices = np.arange(owned + ghosts, dtype=np.int32)
    global_indices = index_map.local_to_global(local_indices)
    values = field.local_array_with_ghosts().reshape((-1, block_size))

    owned_tables = function_space.mesh.comm.allgather(
        (global_indices[:owned], values[:owned].copy())
    )
    owner_values = {
        int(global_index): value
        for indices, table in owned_tables
        for global_index, value in zip(indices, table)
    }
    for global_index, ghost_value in zip(global_indices[owned:], values[owned:]):
        assert np.allclose(ghost_value, owner_values[int(global_index)])


def run_probe():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        raise RuntimeError("production energy MPI probe requires exactly two ranks")

    domain = mesh.create_unit_square(comm, 3, 3)
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    scalar_space = fem.functionspace(domain, ("DG", 0))
    m = Field(
        vector_space,
        lambda x: np.vstack(
            (x[0], 2.0 * x[1], np.full(x.shape[1], 0.8))
        ),
        name="m",
    )
    Ms = Field(scalar_space, 2.5, name="Ms")

    applied_value = np.array((1.25, -2.5, 3.75))
    zeeman = Zeeman(applied_value)
    zeeman.setup(m, Ms, unit_length=2.0e-9)
    assert np.allclose(zeeman.compute_field().reshape((-1, 3)), applied_value)
    assert np.allclose(zeeman.average_field(), applied_value)
    zeeman_function = Field(vector_space, zeeman.compute_field())
    _assert_function_ghosts_match_owners(zeeman_function)
    energies = comm.allgather(zeeman.compute_energy())
    assert np.allclose(energies, energies[0])
    zeeman.set_value(
        lambda x: np.vstack(
            (1.0 + x[0], np.full(x.shape[1], 2.0), np.full(x.shape[1], 3.0))
        )
    )
    assert np.allclose(zeeman.average_field(), (1.5, 2.0, 3.0))

    exchange_1 = Exchange(5.0)
    exchange_2 = Exchange(5.0)
    exchange_1.setup(m, Ms, unit_length=1.0)
    exchange_2.setup(m, Ms, unit_length=2.0)
    H1 = exchange_1.compute_field()
    H2 = exchange_2.compute_field()
    owned_scalar_dofs = (
        vector_space.dofmap.index_map.size_local
        * vector_space.dofmap.index_map_bs
    )
    assert H1.shape == (owned_scalar_dofs,)
    assert np.max(np.abs(H1)) > 0.0
    assert np.allclose(H2, H1 / 4.0)
    assert np.isclose(exchange_1.compute_energy(), 25.0, rtol=1e-12)
    density = exchange_1.energy_density()
    local_density_energy = np.dot(density, exchange_1.nodal_volume_S1)
    density_energy = comm.allreduce(local_density_energy, op=MPI.SUM)
    assert np.isclose(
        density_energy, exchange_1.compute_energy(), rtol=1e-12, atol=1e-14
    )
    density_function = Field(exchange_1.S1, exchange_1.energy_density_function())
    _assert_function_ghosts_match_owners(density_function)
    exchange_function = Field(vector_space, H1)
    _assert_function_ghosts_match_owners(exchange_function)

    local_component_volume = exchange_1.nodal_volume_S3.reshape((-1, 3)).sum(0)
    global_component_volume = np.zeros(3)
    comm.Allreduce(local_component_volume, global_component_volume, op=MPI.SUM)
    assert np.allclose(global_component_volume, 1.0)

    weighted = (
        H1.reshape((-1, 3))
        * exchange_1.nodal_volume_S3.reshape((-1, 3))
    ).sum(0)
    global_weighted = np.zeros(3)
    comm.Allreduce(weighted, global_weighted, op=MPI.SUM)
    assert np.allclose(global_weighted, 0.0, atol=1e-8)

    # DMI (Task 13): unit_length**-1 field scaling (not the exchange-style
    # unit_length**-2), collective energy agreement across ranks, and an
    # owned/ghost-consistent field, using the same distributed ``m``/``Ms``.
    dmi_1 = DMI(5.0, dmi_type="auto")
    dmi_2 = DMI(5.0, dmi_type="auto")
    dmi_1.setup(m, Ms, unit_length=1.0)
    dmi_2.setup(m, Ms, unit_length=2.0)
    H_dmi_1 = dmi_1.compute_field()
    H_dmi_2 = dmi_2.compute_field()
    assert np.max(np.abs(H_dmi_1)) > 0.0
    assert np.allclose(H_dmi_2, H_dmi_1 / 2.0)
    dmi_energies = comm.allgather(dmi_1.compute_energy())
    assert np.allclose(dmi_energies, dmi_energies[0])
    dmi_function = Field(vector_space, H_dmi_1)
    _assert_function_ghosts_match_owners(dmi_function)

    dmi_reversed = DMI(-5.0, dmi_type="auto")
    dmi_reversed.setup(m, Ms, unit_length=1.0)
    assert np.isclose(
        dmi_reversed.compute_energy(), -dmi_1.compute_energy(), rtol=1e-12
    )

    try:
        DMI(1.0, dmi_type="D2D")
    except NotImplementedError as error:
        assert "D2D" in str(error)
    else:
        raise AssertionError("dmi_type='D2D' was not rejected by name")

    anisotropy = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0), K2=1.5)
    anisotropy.setup(m, Ms)
    anisotropy_values = anisotropy.compute_field().reshape((-1, 3))
    expected_z = (2.0 * 4.0 * 0.8 + 4.0 * 1.5 * 0.8**3) / (mu0 * 2.5)
    assert np.allclose(
        anisotropy_values, (0.0, 0.0, expected_z), rtol=1e-13, atol=1e-9
    )
    expected_anisotropy_energy = 4.0 * (1.0 - 0.8**2) - 1.5 * 0.8**4
    assert np.isclose(
        anisotropy.compute_energy(), expected_anisotropy_energy, rtol=1e-12
    )
    anisotropy_function = Field(vector_space, anisotropy.compute_field())
    _assert_function_ghosts_match_owners(anisotropy_function)

    invalid_ms = Field(scalar_space, 2.5)
    if comm.rank == 0:
        invalid_ms.f.x.array[0] = 0.0
    invalid_ms.f.x.scatter_forward()
    try:
        Exchange(5.0).setup(m, invalid_ms)
    except ValueError as error:
        assert "Ms must be positive" in str(error)
    else:
        raise AssertionError("a rank-local zero Ms value was not rejected")

    owned_blocks = vector_space.dofmap.index_map.size_local
    ghost_blocks = vector_space.dofmap.index_map.num_ghosts
    owned_by_rank = comm.gather(owned_blocks, root=0)
    ghosts_by_rank = comm.gather(ghost_blocks, root=0)
    if comm.rank == 0:
        print(
            json.dumps(
                {
                    "ghost_blocks_by_rank": ghosts_by_rank,
                    "mpi_size": comm.size,
                    "owned_blocks_by_rank": owned_by_rank,
                    "status": "ok",
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    run_probe()
