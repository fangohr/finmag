"""Runnable M4 DOLFINx relaxation example with JSON output.

This example is deliberately small and lives under ``dev/dolfinx``. It combines
the current prototype pieces into one checked workflow: mesh creation,
magnetisation setup, exchange/Zeeman/anisotropy energy evaluation, explicit LLG
time stepping, and basic machine-readable output. [Codex gpt-5.5 high]
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import dolfinx
from dolfinx import mesh
from mpi4py import MPI
import numpy as np

from dev.dolfinx.prototype import average_nodal_vector
from dev.dolfinx.prototype import constant_vector_function
from dev.dolfinx.prototype import exchange_energy
from dev.dolfinx.prototype import explicit_llg_step
from dev.dolfinx.prototype import uniaxial_anisotropy_energy
from dev.dolfinx.prototype import vector_function_space
from dev.dolfinx.prototype import zeeman_energy


SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RelaxationParameters:
    """Configuration for the deliberately small M4 relaxation example.

    The dataclass is a prototype API shape only. It makes the example easier to
    review and test without implying compatibility with legacy Finmag
    configuration objects. [Codex gpt-5.5 high]
    """

    anisotropy_axis: tuple = (0.0, 0.0, 1.0)
    anisotropy_constant: float = 0.25
    exchange_constant: float = 1.0
    field: tuple = (0.0, 0.0, 1.0)
    saturation_magnetisation: float = 1.0
    unit_length: float = 1.0

    def __post_init__(self):
        """Reject invalid prototype parameters before DOLFINx form assembly."""
        _validate_vector("field", self.field)
        _validate_vector("anisotropy_axis", self.anisotropy_axis)
        if self.anisotropy_constant < 0:
            raise ValueError("anisotropy_constant must be non-negative")
        if self.exchange_constant < 0:
            raise ValueError("exchange_constant must be non-negative")
        if self.saturation_magnetisation <= 0:
            raise ValueError("saturation_magnetisation must be positive")
        if self.unit_length <= 0:
            raise ValueError("unit_length must be positive")


def _validate_vector(name, values):
    """Validate the small three-component vectors used by the M4 example."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("%s must contain numeric values" % name)
    if array.shape != (3,):
        raise ValueError("%s must have three components" % name)


@dataclass(frozen=True)
class RelaxationResult:
    """JSON-compatible result contract for the M4 relaxation example."""

    dolfinx_version: str
    dt: float
    energy_history: list
    final_average_m: list
    final_energy: float
    initial_energy: float
    mesh: str
    parameters: dict
    schema_version: int
    steps: int

    def as_summary(self):
        """Return the JSON-compatible dictionary validated by CI."""
        return asdict(self)


def parameters_as_summary(parameters):
    """Convert prototype parameters to JSON-compatible scalar/list values."""
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(parameters).items()
    }


def total_energy(magnetisation, parameters):
    """Compute the reduced prototype energy used by the example."""
    return (
        exchange_energy(
            magnetisation,
            exchange_constant=parameters.exchange_constant,
            unit_length=parameters.unit_length,
        )
        + zeeman_energy(
            magnetisation,
            field=parameters.field,
            saturation_magnetisation=parameters.saturation_magnetisation,
            unit_length=parameters.unit_length,
        )
        + uniaxial_anisotropy_energy(
            magnetisation,
            axis=parameters.anisotropy_axis,
            anisotropy_constant=parameters.anisotropy_constant,
            unit_length=parameters.unit_length,
        )
    )


def run_relaxation_example(output_path=None, steps=5, dt=1e-2, parameters=None):
    """Run a tiny deterministic relaxation and optionally write JSON output."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if parameters is None:
        parameters = RelaxationParameters()

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    function_space = vector_function_space(domain)
    magnetisation = constant_vector_function(function_space, (1.0, 0.0, 0.0))

    energy_history = [float(total_energy(magnetisation, parameters))]
    for _ in range(steps):
        explicit_llg_step(
            magnetisation,
            effective_field=parameters.field,
            dt=dt,
            gamma=1.0,
            alpha=1.0,
        )
        energy_history.append(float(total_energy(magnetisation, parameters)))

    result = RelaxationResult(
        dolfinx_version=dolfinx.__version__,
        dt=float(dt),
        energy_history=energy_history,
        final_average_m=average_nodal_vector(magnetisation).tolist(),
        final_energy=energy_history[-1],
        initial_energy=energy_history[0],
        mesh="unit_square_2x2",
        parameters=parameters_as_summary(parameters),
        schema_version=SUMMARY_SCHEMA_VERSION,
        steps=int(steps),
    )
    summary = result.as_summary()

    if output_path is not None and domain.comm.rank == 0:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    domain.comm.Barrier()

    return summary


def validate_summary(summary):
    """Validate the JSON-compatible output contract for the M4 example.

    The schema is intentionally small and hand-written. It gives CI a stable
    contract for the first M4 output artefact without adding a JSON-schema
    dependency to the isolated DOLFINx environment. [Codex gpt-5.5 high]
    """
    required_keys = {
        "dolfinx_version",
        "dt",
        "energy_history",
        "final_average_m",
        "final_energy",
        "initial_energy",
        "mesh",
        "parameters",
        "schema_version",
        "steps",
    }
    missing = required_keys.difference(summary)
    if missing:
        raise ValueError("summary is missing keys: %s" % ", ".join(sorted(missing)))

    if summary["schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported summary schema version")
    if summary["mesh"] != "unit_square_2x2":
        raise ValueError("unexpected mesh label")
    if summary["steps"] < 1:
        raise ValueError("steps must be positive")
    if summary["dt"] <= 0:
        raise ValueError("dt must be positive")
    if len(summary["energy_history"]) != summary["steps"] + 1:
        raise ValueError("energy history length does not match steps")
    if summary["initial_energy"] != summary["energy_history"][0]:
        raise ValueError("initial energy does not match energy history")
    if summary["final_energy"] != summary["energy_history"][-1]:
        raise ValueError("final energy does not match energy history")
    if len(summary["final_average_m"]) != 3:
        raise ValueError("final average magnetisation must have three components")
    if summary["final_energy"] >= summary["initial_energy"]:
        raise ValueError("example did not reduce energy")

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
    validate_summary(summary)
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
