"""Runnable M5 DOLFINx restart-state round-trip example.

This example exercises the reduced ``PrototypeSimulation`` restart-state path
without claiming compatibility with legacy Finmag restart files. It creates a
tiny unit-square simulation, advances it, writes the narrow JSON restart state,
reloads it, and reports whether the reconstructed state matches the original.
[Codex gpt-5.5 high]
"""

import argparse
import json
from pathlib import Path

from mpi4py import MPI
import numpy as np

from dev.dolfinx.simulation import PrototypeSimulation


def run_restart_example(output_path=None, steps=2, dt=1e-2):
    """Run the reduced restart-state round trip and return a summary."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    sim = PrototypeSimulation.unit_square()
    sim.relax(steps=steps, dt=dt)

    if output_path is None:
        state = sim.restart_state()
        restarted = PrototypeSimulation.from_restart_state(state)
        restart_path = None
    else:
        output_path = Path(output_path)
        state = sim.write_restart_state(output_path)
        restarted = PrototypeSimulation.read_restart_state(output_path)
        restart_path = str(output_path)

    average_m_error = float(np.max(np.abs(sim.average_m() - restarted.average_m())))
    energy_error = _max_energy_error(sim.energy_terms(), restarted.energy_terms())
    return {
        "dt": float(dt),
        "max_average_m_error": average_m_error,
        "max_energy_error": energy_error,
        "mesh": state["mesh"],
        "restart_path": restart_path,
        "restart_schema_version": state["schema_version"],
        "roundtrip_matches": average_m_error == 0.0 and energy_error == 0.0,
        "steps": int(steps),
    }


def _max_energy_error(left, right):
    """Return the largest absolute energy-term difference."""
    return float(max(abs(left[key] - right[key]) for key in sorted(left)))


def main():
    """Command-line entry point for the M5 restart-state example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/tmp/finmag-dolfinx-restart-state.json",
        help="Path for the reduced restart JSON state written by rank 0.",
    )
    parser.add_argument("--steps", default=2, type=int, help="Number of LLG steps.")
    parser.add_argument("--dt", default=1e-2, type=float, help="Explicit step size.")
    args = parser.parse_args()

    summary = run_restart_example(args.output, steps=args.steps, dt=args.dt)
    if not summary["roundtrip_matches"]:
        raise RuntimeError("restart round trip did not reproduce the prototype state")
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
