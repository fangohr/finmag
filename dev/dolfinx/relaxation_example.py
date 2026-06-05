"""Runnable M4 DOLFINx relaxation example with JSON output.

This example is deliberately small and lives under ``dev/dolfinx``. It combines
the current prototype pieces into one checked workflow: mesh creation,
magnetisation setup, exchange/Zeeman/anisotropy energy evaluation, explicit LLG
time stepping, and basic machine-readable output. [Codex gpt-5.5 high]
"""

import argparse
import json
from pathlib import Path

import dolfinx
from dolfinx import mesh
from mpi4py import MPI

from dev.dolfinx.prototype import average_nodal_vector
from dev.dolfinx.prototype import constant_vector_function
from dev.dolfinx.prototype import exchange_energy
from dev.dolfinx.prototype import explicit_llg_step
from dev.dolfinx.prototype import uniaxial_anisotropy_energy
from dev.dolfinx.prototype import vector_function_space
from dev.dolfinx.prototype import zeeman_energy


def total_energy(magnetisation, parameters):
    """Compute the reduced prototype energy used by the example."""
    return (
        exchange_energy(
            magnetisation,
            exchange_constant=parameters["exchange_constant"],
            unit_length=parameters["unit_length"],
        )
        + zeeman_energy(
            magnetisation,
            field=parameters["field"],
            saturation_magnetisation=parameters["saturation_magnetisation"],
            unit_length=parameters["unit_length"],
        )
        + uniaxial_anisotropy_energy(
            magnetisation,
            axis=parameters["anisotropy_axis"],
            anisotropy_constant=parameters["anisotropy_constant"],
            unit_length=parameters["unit_length"],
        )
    )


def run_relaxation_example(output_path=None, steps=5, dt=1e-2):
    """Run a tiny deterministic relaxation and optionally write JSON output."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))
    parameters = {
        "anisotropy_axis": (0.0, 0.0, 1.0),
        "anisotropy_constant": 0.25,
        "exchange_constant": 1.0,
        "field": (0.0, 0.0, 1.0),
        "saturation_magnetisation": 1.0,
        "unit_length": 1.0,
    }

    energy_history = [float(total_energy(magnetisation, parameters))]
    for _ in range(steps):
        explicit_llg_step(
            magnetisation,
            effective_field=parameters["field"],
            dt=dt,
            gamma=1.0,
            alpha=1.0,
        )
        energy_history.append(float(total_energy(magnetisation, parameters)))

    summary = {
        "dolfinx_version": dolfinx.__version__,
        "dt": float(dt),
        "energy_history": energy_history,
        "final_average_m": average_nodal_vector(magnetisation).tolist(),
        "initial_energy": energy_history[0],
        "final_energy": energy_history[-1],
        "mesh": "unit_square_2x2",
        "parameters": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in parameters.items()
        },
        "steps": int(steps),
    }

    if output_path is not None and domain.comm.rank == 0:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    domain.comm.Barrier()

    return summary


def main():
    """Command-line entry point for the checked M4 relaxation example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="dolfinx-relaxation-summary.json",
        help="Path for the JSON summary written by rank 0.",
    )
    parser.add_argument("--steps", default=5, type=int, help="Number of LLG steps.")
    parser.add_argument("--dt", default=1e-2, type=float, help="Explicit step size.")
    args = parser.parse_args()

    summary = run_relaxation_example(args.output, steps=args.steps, dt=args.dt)
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
