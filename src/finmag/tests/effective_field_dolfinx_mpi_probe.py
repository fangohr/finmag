"""Two-rank production gate for the DOLFINx EffectiveField registry."""

import json

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.field import Field
from finmag.physics.effective_field import EffectiveField


def run_probe():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        raise RuntimeError(
            "production EffectiveField MPI probe requires exactly two ranks"
        )

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

    effective_field = EffectiveField(m, Ms, unit_length=2.0e-9)
    assert effective_field.output_size == m.as_array().size

    applied_value = np.array((1.25, -2.5, 3.75))
    zeeman = Zeeman(applied_value, name="Zeeman")
    exchange = Exchange(5.0, name="Exchange")
    anisotropy = UniaxialAnisotropy(4.0, (0.0, 0.0, 1.0), K2=1.5, name="Anisotropy")

    effective_field.add(zeeman)
    effective_field.add(exchange)
    effective_field.add(anisotropy)

    assert effective_field.all() == ["Anisotropy", "Exchange", "Zeeman"]

    H_total = effective_field.compute()
    expected_total = (
        zeeman.compute_field() + exchange.compute_field() + anisotropy.compute_field()
    )
    assert H_total.shape == (m.as_array().size,)
    assert np.allclose(H_total, expected_total)

    # compute() must hand back an owned-only copy: mutating it and
    # recomputing must not disturb the accumulator or the next result.
    H_total_copy = H_total.copy()
    H_total[:] = -1.0
    H_total_again = effective_field.compute()
    assert np.allclose(H_total_again, H_total_copy)

    energy_total = effective_field.total_energy()
    expected_energy = (
        zeeman.compute_energy() + exchange.compute_energy() + anisotropy.compute_energy()
    )
    assert np.isclose(energy_total, expected_energy, rtol=1e-12)
    # total_energy() collectively reduces over both ranks, so every rank must
    # observe the same global value.
    energies_by_rank = comm.allgather(energy_total)
    assert np.allclose(energies_by_rank, energies_by_rank[0])

    jacobian_only = effective_field.compute_jacobian_only(t=None)
    expected_jacobian = exchange.compute_field() + anisotropy.compute_field()
    assert np.allclose(jacobian_only, expected_jacobian)
    assert not np.allclose(jacobian_only, H_total_copy)

    # Removing an interaction changes the accumulated field, not only energy.
    effective_field.remove("Exchange")
    H_after = effective_field.compute()
    assert not np.allclose(H_after, H_total_copy)
    assert np.allclose(H_after, zeeman.compute_field() + anisotropy.compute_field())

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
